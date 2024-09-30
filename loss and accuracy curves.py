# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 00:16:37 2024

@author: chris
"""
import re
import matplotlib.pyplot as plt
path='C:/Users/chris/OneDrive/Desktop/Training Results/10000, 0.0025 aug.txt'
# Read the content of the text file
with open(path, 'r') as file:
    log_text = file.read()

# Regular expressions to extract the required information
step_pattern = r"Step (\d+):"
train_accuracy_pattern = r"Train accuracy = ([\d.]+)%"
cross_entropy_pattern = r"Cross entropy = ([\d.]+)"
validation_accuracy_pattern = r"Validation accuracy = ([\d.]+)%"

# Extracting the information
steps = re.findall(step_pattern, log_text)
train_accuracies = re.findall(train_accuracy_pattern, log_text)
cross_entropies = re.findall(cross_entropy_pattern, log_text)
validation_accuracies = re.findall(validation_accuracy_pattern, log_text)

# Converting strings to appropriate data types
# Steps list will only contain unique steps
unique_steps = sorted(set(int(step) for step in steps))
train_accuracies = [float(acc) for acc in train_accuracies]
cross_entropies = [float(ent) for ent in cross_entropies]
validation_accuracies = [float(val_acc) for val_acc in validation_accuracies]

# Print the results
print("Steps:", unique_steps)
print("Train Accuracies:", train_accuracies)
print("Cross Entropies:", cross_entropies)
print("Validation Accuracies:", validation_accuracies)

def plot_train():
    plt.scatter(unique_steps,train_accuracies, marker='x')
    plt.xlabel('training steps')
    plt.ylabel('training accuracy')
    
    plt.savefig(path+'training.png')
    plt.show()
    plt.close()

def plot_valid():
    plt.scatter(unique_steps,validation_accuracies, marker='x')
    plt.xlabel('training steps')
    plt.ylabel('validation accuracy')
    plt.savefig(path+'validation.png')
    plt.show()
    plt.close()

def plot_entropy():
    plt.scatter(unique_steps,cross_entropies, marker='x')
    plt.xlabel('training steps')
    plt.ylabel('cross entropy loss')
    plt.savefig(path+'entropy.png')
    plt.show()
    plt.close()
    
plot_train()
plot_valid()
plot_entropy()