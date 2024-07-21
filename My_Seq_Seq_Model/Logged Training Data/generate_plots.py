import matplotlib.pyplot as plt
import re

def extract_data_from_line(line):
    # Extract relevant information using regular expressions
    match = re.search(r'train loss: (.*?), val_seen loss: (.*?), success_rate: (.*?), val_unseen loss: (.*?), success_rate: (.*?)$', line)
    if match:
        train_loss, val_seen_loss, val_seen_success_rate, val_unseen_loss, val_unseen_success_rate = match.groups()
        return float(train_loss), float(val_seen_loss), float(val_seen_success_rate), float(val_unseen_loss), float(val_unseen_success_rate)
    else:
        return None

def plot_losses_and_success_rates(log_file):
    # Read lines from the log file
    with open(log_file, 'r') as file:
        lines = file.readlines()

    # Extract data from each line
    data = [extract_data_from_line(line) for line in lines]

    # Separate data for plotting
    train_losses = [entry[0] for entry in data if entry is not None]
    val_seen_losses = [entry[1] for entry in data if entry is not None]
    val_seen_success_rates = [entry[2] for entry in data if entry is not None]
    val_unseen_losses = [entry[3] for entry in data if entry is not None]
    val_unseen_success_rates = [entry[4] for entry in data if entry is not None]

    
    # Loss plots
    plt.figure(figsize=(8, 12))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_seen_losses, label='Validation Seen Loss')
    plt.plot(val_unseen_losses, label='Validation Unseen Loss')
    plt.title('Training and Validation Losses', fontsize=50)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.ylim(0, 3.0)
    plt.xlabel('No. of Iterations (in 100s)', fontsize=23)
    plt.ylabel('Loss', fontsize=30)
    plt.legend()
    plt.show()
    
    # Success rate plots
    plt.figure(figsize=(8, 12))
    plt.plot(val_seen_success_rates, label='Validation Seen Success Rate')
    plt.plot(val_unseen_success_rates, label='Validation Unseen Success Rate')
    plt.title('Validation Success Rates', fontsize=50)
    plt.xlabel('No. of Iterations (in 100s)', fontsize=23)
    plt.ylabel('Success Rate', fontsize=30)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.ylim(0, 0.4)
    plt.legend()
    plt.show()




# Replace 'your_log_file.txt' with the actual path to your log file
log_file_path = 'masti.txt'
plot_losses_and_success_rates(log_file_path)
