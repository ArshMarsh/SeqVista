# This script is for the creation of Vision-Language Navigation Agent
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.distributions as D
import numpy as np
import os
import random 
from env import R2RBatch # import file missing for now!
from utils import padding_idx # import file missing for now!

# For the encoder class, the orginal Attention-based Seq-Seq model computes the encoder context
# or enc_ctx by taking the 'outputs' part of the output as well and performing some procssing on it, 
# but here we are just considering the final hidden and cell states --> No Attention

# One more notable difference here is that we do not put here the lengths of batches for dynamic
# batching (inputs in the forward function) --> To be looked into later
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, p, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions
        )
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    def forward(self, x, lengths):
        # x shape: (seq_length, N)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N)
        packed_embeds = pack_padded_sequence(embedding, lengths, batch_first=True)
        # Outputs show the hidden states at each time step. While hidden, cell show only the final
        # cell and hidden states!
        enc_h, (h_t, c_t) = self.rnn(packed_embeds)
        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        ## This decoder_init and some stuff is still not clear to me. Maybe, would require some
        ## clarification afterwards. Okay?
        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.dropout(ctx)
        return ctx, decoder_init, c_t

# Added the feature size as an input
# Added the action as an input to the forward method
class Decoder(nn.Module):
    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size, num_layers,
                  p, feature_size=2048):

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size+feature_size, hidden_size, num_layers, dropout=p)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)

    def forward(self, action, feature, hidden, cell, ctx, ctx_mask=None):
        # shape of x: (N) but we want (1,N)
        action_embeds = self.embedding(action)  # (batch, 1, embedding_size)
        #print(action_embeds.size())
        action_embeds = action_embeds.squeeze(1)
        #print(action_embeds.size())
        concat_input = torch.cat((action_embeds, feature), 1) # (batch, embedding_size+feature_size)
        #print(concat_input.size())
        ## There were some issues of dimensions over here!
        drop = self.dropout(concat_input.unsqueeze(0))
        outputs, (h_1, c_1) = self.lstm(drop, (hidden, cell))
        logit = self.decoder2action(outputs.squeeze(0))
        #print(logit.size())

        return h_1, c_1, logit # Not returning all of h_1,c_1,alpha,logit ?


# This is the base class for an R2R agent to generate and save trajectories!
class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print('Testing %s' % self.__class__.__name__)
        looped = False
        while True:
            for traj in self.rollout():
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']
            if looped:
                break

# Just see that there is a logit thing (from the decoder in model.py file) inside this class in
# rollout function definition
# --- more investigation required on this as well!
class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
      (0,-1, 0), # left
      (0, 1, 0), # right
      (0, 0, 1), # up
      (0, 0,-1), # down
      (1, 0, 0), # forward
      (0, 0, 0), # <end>
      (0, 0, 0), # <start>
      (0, 0, 0)  # <ignore>
    ]
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.model_actions.index('<ignore>'))
        ## Here, the criterion takes in the model_actions --> Somewhat solves the paradox of where
        ## these model/env_actions are actually used!
        ## This loss is further used in the rollout function definition
    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]

        return Variable(sorted_tensor, requires_grad=False).long(), \
               mask.byte(), \
               list(seq_lengths), list(perm_idx) # Two cudas removed here!

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        feature_size = obs[0]['feature'].shape[0]
        features = np.empty((len(obs),feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False)#.cuda()

    ## How is this teacher getting these actions? Are these from the dataset? But what about right
    ## left stuff etc.. ?
    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = self.model_actions.index('right')
            elif heading_chg < 0:
                a[i] = self.model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = self.model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = self.model_actions.index('down')
            elif ix > 0:
                a[i] = self.model_actions.index('forward')
            elif ended[i]:
                a[i] = self.model_actions.index('<ignore>')
            else:
                a[i] = self.model_actions.index('<end>')
        return Variable(a, requires_grad=False)#.cuda()

    def rollout(self):
        # This line gives you the observations, stored in the obs variable in form of
        # [ix, heading, elvation] kind of a thing! (reset fn called from the env.py file
        # to give you the described result)  
        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False)#.cuda()
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        for t in range(self.episode_len):

            f_t = self._feature_variable(perm_obs) # Image features from obs

            ## Somehow, here we are passing these arguments to the forward method of decoder class
            ## which I literally think I don't understand at the moment. Shouldn't we always pass the
            ## input arguments as given in the def__init(--) method? 
            h_t,c_t,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(perm_obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit, dim=1)
                m = D.Categorical(probs)
                a_t = m.sample()            # sampling an action from model
            else:
                sys.exit('Invalid feedback option')

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True
                env_action[idx] = self.env_actions[action_idx]

            obs = np.array(self.env.step(env_action))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            # Early exit if all ended
            if ended.all():
                break

        self.losses.append(self.loss.item() / self.episode_len)
        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))






