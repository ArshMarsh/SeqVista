# SeqVista 

## Overview

The SeqVista project focuses on advancing Vision-Language Navigation (VLN) by developing a robust Sequence-to-Sequence (Seq2Seq) model. The objective is to train an autonomous agent to interpret and execute natural language instructions in unseen environments..

## Method

### Key Components:
- **Seq2Seq Model:** Core model for training the autonomous agent.
- **Matterport 3D Simulator:** Provides a graph-based environment with real-world imagery.
- **Room-to-Room (R2R) Dataset:** Offers diverse instruction-trajectory pairs for training and evaluation.
- **Removal of attention mechanism**

### Techniques Used:
- **Natural Language Processing (NLP):** To understand and process instructions.
- **Computer Vision:** For interpreting visual inputs.
- **Attention Mechanism:** Enhances the model's ability to focus on relevant parts of the input.
- **LSTM and RESNET-152:** Utilized within the Seq2Seq architecture for processing sequential data and visual features, respectively.

## Results

We observed that the removal of attention mechanism heavily impacts the model's ability to accurately navigate rooms.

## Acknowledgments
We used identical settings to [Matterport3D and their models](https://github.com/peteanderson80/Matterport3DSimulator) and removed the attention mechanism.
