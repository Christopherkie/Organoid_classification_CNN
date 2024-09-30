# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:02:27 2024

@author: chris
"""
#plot confusion matrix for multiclass classification
#I need the TP and FP for all samples

#read the file and create counts for all the differetn tp and fp

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def check_classifications(file_path):
    count_8028_8028=0
    count_8028_9591=0
    count_8028_16992=0
    count_9591_8028=0
    count_9591_9591=0
    count_9591_16992=0
    count_16992_8028=0
    count_16992_9591=0
    count_16992_16992=0

    # Open the CSV file
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        
        # Skip the header row if your CSV file has one
        next(csv_reader, None)
        
        for row in csv_reader:
            numerator = row[0]
            guess = row[1]
            correct_classification = row[2]
            # Compare guess with correct classification
            if correct_classification=='8028':
                if int(guess)==8028:
                    count_8028_8028+=1
                elif int(guess)==9591:
                    count_8028_9591+=1
                elif int(guess)==16992:
                    count_8028_16992+=1
                else:
                    print(f'value error in csv at line {row}')
            elif correct_classification=='9591':
                if int(guess)==8028:
                    count_9591_8028+=1
                elif int(guess)==9591:
                    count_9591_9591+=1
                elif int(guess)==16992:
                    count_9591_16992+=1
                else:
                    print(f'value error in csv at line {row}')
            elif correct_classification=='16992':
                if int(guess)==8028:
                    count_16992_8028+=1
                elif int(guess)==9591:
                    count_16992_9591+=1
                elif int(guess)==16992:
                    count_16992_16992+=1
                else:
                    print(f'value error in csv at line {row}')
            else:
                continue
                    
    
    print(f'8028:\n 8028:{count_8028_8028}\n 9591:{count_8028_9591}\n 16992:{count_8028_16992}')
    print(f'9591:\n 8028:{count_9591_8028}\n 9591:{count_9591_9591}\n 16992:{count_9591_16992}')
    print(f'16992:\n 8028:{count_16992_8028}\n 9591:{count_16992_9591}\n 16992:{count_16992_16992}')
    return np.array([
        [count_8028_8028, count_8028_9591, count_8028_16992],
        [count_9591_8028, count_9591_9591, count_9591_16992],
        [count_16992_8028, count_16992_9591, count_16992_16992]
        ])
file_path ="C:/Desktop/Leer 2.csv"    

counts = check_classifications(file_path)

# Optionally, you can provide labels for the classes
class_labels = ['8028', '9591', '16992']
def normalize_array(array):
    norm_8028=[]
    norm_9591=[]
    norm_16992=[]
    #8028
    count=0
    i=0
    for j in range(3):
        #print(array[i][j])
        count+=array[i][j]
    for j in range(3):
        #print(count)
        norm_8028.append(np.divide(array[i][j],count))
    #9591
    count=0
    i=1
    for j in range(3):
        #print(array[i][j])
        count+=array[i][j]
    for j in range(3):
        #print(count)
        norm_9591.append(np.divide(array[i][j],count))
    #16992
    count=0
    i=2
    for j in range(3):
        #print(array[i][j])
        count+=array[i][j]
    for j in range(3):
        #print(count)
        norm_16992.append(np.divide(array[i][j],count))
    return np.array([
        norm_8028, norm_9591, norm_16992
        ])
counts=normalize_array(counts)
# Create a heatmap using seaborn
# Alternatively, you can use sklearn's ConfusionMatrixDisplay
# Uncomment the following lines to use sklearn's plot
# Create a heatmap using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(counts, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

#disp = ConfusionMatrixDisplay(confusion_matrix=counts, display_labels=class_labels)
#disp.plot(cmap="Blues")
#plt.show()