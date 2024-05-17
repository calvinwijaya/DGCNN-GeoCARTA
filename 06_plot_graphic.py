import pandas as pd
import re
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Function to extract numbers from a string
def extract_numbers(text):
    return [float(num) for num in re.findall(r'[-+]?\d*\.\d+|\d+', text)]

def parse_args():
    parser = argparse.ArgumentParser('Plot Graphics')
    parser.add_argument('--folder', type=str, required=True, help='Name of Folder Directory')
    return parser.parse_args()

# Call parse_args() to get the command-line arguments
args = parse_args()

log_dir = 'log/sem_seg/' + args.folder
run_dir = log_dir + '/run.log'
run_dir = Path(run_dir)

# Read the log file skipping the first three rows
with open(run_dir, 'r') as file:
    lines = file.readlines()[3:]

# Initialize empty lists to store the extracted information
train_data = []
test_data = []

# Extract information from each line
for line in lines:
    if line.startswith('Train') or line.startswith('Test'):
        values = extract_numbers(line)
        
        if line.startswith('Train'):
            train_data.append(values)
        elif line.startswith('Test'):
            test_data.append(values)

# Create dataframes for training and testing data
columns = ['epoch', 'loss', 'acc', 'avg_acc', 'iou']
train_df = pd.DataFrame(train_data, columns=['epoch'] + columns[1:])
test_df = pd.DataFrame(test_data, columns=['epoch'] + columns[1:])

# Save dataframes to CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

# Plot and save figures
plt.figure(figsize=(16, 8))

# Figure 1: Train accuracy and train loss
plt.subplot(1, 3, 1)
plt.plot(train_df.index, train_df['acc'], label='Train Accuracy')
plt.plot(train_df.index, train_df['loss'], label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('Train Accuracy and Train Loss')

# Figure 2: Test accuracy and test loss
plt.subplot(1, 3, 2)
plt.plot(test_df.index, test_df['acc'], label='Test Accuracy')
plt.plot(test_df.index, test_df['loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('Test Accuracy and Test Loss')

# Figure 3: Train loss and test loss
plt.subplot(1, 3, 3)
plt.plot(train_df.index, train_df['loss'], label='Train Loss')
plt.plot(test_df.index, test_df['loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('Train Loss and Test Loss')
plt.savefig('Training Plot.png')

plt.show()