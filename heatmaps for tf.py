# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:53:05 2024

@author: chris
"""
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

count_8028_8028=0
count_8028_9591=0
count_8028_16992=0
count_9591_8028=0
count_9591_9591=0
count_9591_16992=0
count_16992_8028=0
count_16992_9591=0
count_16992_16992=0
my_file = open("C:/Users/chris/OneDrive/Desktop/test result/ADAM final.txt", "r") 
i=0 
data_into_list=[]
liststring=""
for line in my_file:
    liststring+=line
    i=i+1
    if (i<5):
            continue
    elif(i==5):
            i=0
            data_into_list.append(liststring)
            liststring=""

for sample_string in data_into_list:
    
    # Regular expression for extracting score for 16992
    regex_16992 = re.compile(r'16992 \(score = (?P<A>\d\.\d+)\)')
    match_16992 = regex_16992.search(sample_string)
    score_16992 = match_16992.group('A') if match_16992 else None
    
    # Regular expression for extracing score for 9591
    regex_9591 = re.compile(r'9591 \(score = (?P<B>\d\.\d+)\)')
    match_9591 = regex_9591.search(sample_string)
    score_9591 = match_9591.group('B') if match_9591 else None
    
    # Regular expression for extracting score for 8028
    regex_8028 = re.compile(r'8028 \(score = (?P<C>\d\.\d+)\)')
    match_8028 = regex_8028.search(sample_string)
    score_8028 = match_8028.group('C') if match_8028 else None
    
    # Regular expression for extracting subtype
    regex_subtype = re.compile(r'\\(?P<subtype>\d+)_')
    match_subtype = regex_subtype.search(sample_string)
    subtype = match_subtype.group('subtype') if match_subtype else None
    
    # Regular expression for extracting subtype and time
    regex_subtype_time = re.compile(r'_(?P<time>\d{3})\.jpg')
    match_subtype_time = regex_subtype_time.search(sample_string)
    time = match_subtype_time.group('time') if match_subtype_time else None
    #print(f"1 (10669) score: {score_10669}")
    #print(f"2 (9591) score: {score_9591}")
    #print(f"3 (8028) score: {score_8028}")
    #print(f"Subtype: {subtype}")
    #print(f"Time: {time}")
    #print(f"1 (10669) score: {score_10669}")
    #print(f"2 (9591) score: {score_9591}")
    #print(f"3 (8028) score: {score_8028}")
    #print(f"Subtype: {subtype}")
    
    if time==None:
        time=0
    if int(time)>70:
        print(sample_string, time)
    if subtype=='8028':
        if  max(score_8028,score_16992,score_9591)==score_8028:    
            count_8028_8028+=1
        elif max(score_8028,score_16992,score_9591)==score_9591:    
            count_8028_9591+=1
        elif max(score_8028,score_16992,score_9591)==score_16992:    
            count_8028_16992+=1
        else:
            print(f'error in {sample_string}, {score_8028},{score_9591}, {score_16992}')
        time=int(time)
        
    elif subtype=='16992':
        if  max(score_8028,score_16992,score_9591)==score_8028:    
            count_16992_8028+=1
        elif max(score_8028,score_16992,score_9591)==score_9591:    
            count_16992_9591+=1
        elif max(score_8028,score_16992,score_9591)==score_16992:    
            count_16992_16992+=1
        else:
            print(f'error in {sample_string}')
        time=int(time)
        time=int(time)
        
    elif subtype=='9591':
        if  max(score_8028,score_16992,score_9591)==score_8028:    
            count_9591_8028+=1
        elif max(score_8028,score_16992,score_9591)==score_9591:    
            count_9591_9591+=1
        elif max(score_8028,score_16992,score_9591)==score_16992:    
            count_9591_16992+=1
        else:
            print(f'error in {sample_string}')
            
        time=int(time)
        
counts=np.array([
        [count_8028_8028, count_8028_9591, count_8028_16992],
        [count_9591_8028, count_9591_9591, count_9591_16992],
        [count_16992_8028, count_16992_9591, count_16992_16992]
        ])
print(counts)
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
print(normalize_array(counts))
counts=normalize_array(counts)
class_labels = ['8028', '9591', '16992']
plt.figure(figsize=(8, 6))
sns.heatmap(counts, annot=True,fmt='.2f', cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()        