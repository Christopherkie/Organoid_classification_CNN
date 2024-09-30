# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:54:16 2024

@author: chris
"""

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def get_average(list):
    score=0
    count=0
    for i in list:
        score+=float(i)
        count+=1
    if count==0:
        count=1    
    average=score/count    
    return average

def calculate_acc_time(list):
    count=0
    correct=0
    for i in list:
        
        if i==1:
            count+=1
            continue
        elif i==0:
            count+=1
            correct+=1
            continue
    if count==len(list) and not count==0:
        return correct/count
    else:
        print(f'ERROR!{list}')
        
            
            



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
dict8028_8028={}
dict8028_9591={}
dict8028_16992={}
dict9591_8028={}
dict9591_9591={}
dict9591_16992={}
dict16992_8028={}
dict16992_9591={}
dict16992_16992={}
for u in range(100):
    dict8028_8028[u]=[]
    dict8028_9591[u]=[]
    dict8028_16992[u]=[]
    dict9591_8028[u]=[]
    dict9591_9591[u]=[]
    dict9591_16992[u]=[]
    dict16992_8028[u]=[]
    dict16992_9591[u]=[]
    dict16992_16992[u]=[]
dict_acc_8028={}
dict_acc_9591={}
dict_acc_16992={}
dict_acc={}
for u in range(100):
    dict_acc_8028[u]=[]
    dict_acc_9591[u]=[]
    dict_acc_16992[u]=[]
    dict_acc[u]=[]
    
        
totcount=0
totcount8028=0
incorrect8028=0
totcount9591=0
incorrect9591=0
totcount16992=0
incorrect16992=0
incorrectcount=0      
          

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
        totcount+=1
        totcount8028+=1
        time=int(time)
        if not (score_8028>score_16992 and score_8028>score_9591):
            print('incorrect:',sample_string)
            incorrect8028+=1
            dict_acc_8028[time].append(1)
            dict_acc[time].append(1)
        if (score_8028>score_16992 and score_8028>score_9591):
            #print('correct:',time, subtype)
            dict_acc_8028[time].append(0)
            dict_acc[time].append(0)
            
        
        dict8028_8028[time].append(score_8028)
        dict8028_9591[time].append(score_9591)
        dict8028_16992[time].append(score_16992)
    elif subtype=='16992':
        time=int(time)
        totcount+=1
        totcount16992+=1
        if not (score_16992>score_8028 and score_16992>score_9591):
            incorrect16992+=1
            #print('incorrect:',time, subtype)
            dict_acc_16992[time].append(1)
            dict_acc[time].append(1)
        if (score_16992>score_8028 and score_16992>score_9591):            
            #print('correct:',time, subtype)
            dict_acc_16992[time].append(0)
            dict_acc[time].append(0)
        
        dict16992_8028[time].append(score_8028)
        dict16992_9591[time].append(score_9591)
        dict16992_16992[time].append(score_16992)
    elif subtype=='9591':
        time=int(time)
        totcount+=1
        totcount9591+=1
        if not (score_9591>score_16992 and score_9591>score_8028):
            incorrect9591+=1
            #print('incorrect:',time, subtype)
            dict_acc_9591[time].append(1)
            dict_acc[time].append(1)
        if (score_9591>score_16992 and score_9591>score_8028):
            #print('correct:',time, subtype)
            dict_acc_9591[time].append(0)
            dict_acc[time].append(0)
        dict9591_8028[time].append(score_8028)
        dict9591_9591[time].append(score_9591)
        dict9591_16992[time].append(score_16992)
  
average_8028_8028=[]
average_8028_9591=[]
average_8028_16992=[]
for i in range(1,71):
   average_8028_8028.append(get_average(dict8028_8028[i]))
   average_8028_9591.append(get_average(dict8028_9591[i]))
   average_8028_16992.append(get_average(dict8028_16992[i]))
print(average_8028_8028) 
print(len(range(70)))

average_9591_8028=[]
average_9591_9591=[]
average_9591_16992=[]
for i in range(1,71):
   average_9591_8028.append(get_average(dict9591_8028[i]))
   average_9591_9591.append(get_average(dict9591_9591[i]))
   average_9591_16992.append(get_average(dict9591_16992[i])) 

average_16992_8028=[]
average_16992_9591=[]
average_16992_16992=[]
for i in range(1,71):
   average_16992_8028.append(get_average(dict16992_8028[i]))
   average_16992_9591.append(get_average(dict16992_9591[i]))
   average_16992_16992.append(get_average(dict16992_16992[i]))       

    
print(dict_acc_8028)    
accuracy_8028=[]
accuracy_9591=[]
accuracy_16992=[]
accuracylist=[]
for i in range(1,71):
    accuracy_8028.append(calculate_acc_time(dict_acc_8028[i]))
    accuracy_9591.append(calculate_acc_time(dict_acc_9591[i]))
    accuracy_16992.append(calculate_acc_time(dict_acc_16992[i]))
    accuracylist.append(calculate_acc_time(dict_acc[i]))
   
#print(accuracy_8028)    






#plt.scatter(range(70),average_8028_8028,marker='x', color='green', label='8028')
#plt.scatter(range(70),average_8028_9591,marker='x', color='red', label='9591')
#plt.scatter(range(70),average_8028_16992,marker='x', color='blue', label='16992')
#plt.ylabel('average certainty for subtype 8028')
#plt.xlabel('time in hours')
#plt.legend()
#plt.show  

def plot_8028():
    plt.scatter(range(70),average_8028_8028,marker='x', color='green', label='8028')
    plt.scatter(range(70),average_8028_9591,marker='x', color='red', label='9591')
    plt.scatter(range(70),average_8028_16992,marker='x', color='blue', label='16992')
    plt.ylabel('average certainty for subtype 8028')
    plt.xlabel('time in hours')
    plt.legend()
    plt.show()
    plt.close()
def plot_9591():  
    plt.scatter(range(70),average_9591_8028,marker='x', color='green', label='8028')
    plt.scatter(range(70),average_9591_9591,marker='x', color='red', label='9591')
    plt.scatter(range(70),average_9591_16992,marker='x', color='blue', label='16992')
    plt.ylabel('average certainty for subtype 9591')
    plt.xlabel('time in hours')
    plt.legend()
    plt.show()
    plt.close()                 

def plot_16992():
    plt.scatter(range(70),average_16992_8028,marker='x', color='green', label='8028')
    plt.scatter(range(70),average_16992_9591,marker='x', color='red', label='9591')
    plt.scatter(range(70),average_16992_16992,marker='x', color='blue', label='16992')
    plt.ylabel('average certainty for subtype 16992')
    plt.xlabel('time in hours')
    plt.legend()
    plt.show()
    plt.close()

def column_plot_8028():
    species = range(70)
    weight_counts = {
        "16992": average_8028_16992,
        "9591": average_8028_9591,
        "8028": average_8028_8028
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(70)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("average score for subtype 8028 over time in hours")
    ax.legend(loc="upper right")

    plt.show()
    
def column_plot_9591():
    species = range(70)
    weight_counts = {
        "16992": average_9591_16992,
        "9591": average_9591_9591,
        "8028": average_9591_8028
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(70)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("average score for subtype 9591 over time in hours")
    ax.legend(loc="upper right")

    plt.show()


def column_plot_16992():
    species = range(70)
    weight_counts = {
        "16992": average_16992_16992,
        "9591": average_16992_9591,
        "8028": average_16992_8028
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(70)

    for boolean, weight_count in weight_counts.items():
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("average score for subtype 16992 over time in hours")
    ax.legend(loc="upper right")

    plt.show()
    

def calculate_acc():
    if(totcount16992+totcount8028+totcount9591==totcount):
        acc_8028=1-incorrect8028/totcount8028
        acc_9591=1-incorrect9591/totcount9591
        acc_16992=1-incorrect16992/totcount16992
        accuracy=1-(incorrect8028+incorrect9591+incorrect16992)/totcount
        return accuracy,acc_8028,acc_9591,acc_16992
    else:
        print('does not add up')

def plot_acc():
    plt.scatter(range(70),accuracylist,marker='x', color='blue',)
    plt.ylabel('accuracy')
    plt.xlabel('time in hours')
    #plt.legend()
    plt.show()
    plt.close()

def plot_8028_acc():
    plt.scatter(range(70),accuracy_8028,marker='x', color='blue',)
    plt.ylabel('accuracy for subtype 8028')
    plt.xlabel('time in hours')
    #plt.legend()
    plt.show()
    plt.close()

def plot_9591_acc():
    plt.scatter(range(70),accuracy_9591,marker='x', color='blue',)
    plt.ylabel('accuracy for subtype 9591')
    plt.xlabel('time in hours')
    #plt.legend()
    plt.show()
    plt.close()               

def plot_16992_acc():
    plt.scatter(range(70),accuracy_16992,marker='x', color='blue',)
    plt.ylabel('accuracy for subtype 16992')
    plt.xlabel('time in hours')
    #plt.legend()
    plt.show()
    plt.close()
    

plot_16992()
plot_9591()
plot_8028()     
#column_plot_8028()   
#column_plot_9591()
#column_plot_16992()
plot_16992_acc()
plot_8028_acc()
plot_9591_acc()
plot_acc()
        
        
accuracy,acc_8028,acc_9591,acc_16992=calculate_acc()
print(f'accuracy={accuracy},\n accuracy 8028={acc_8028},\n accuracy 9591={acc_9591},\n accuracy 16992={acc_16992}')    
print((acc_16992+acc_8028+acc_9591)/3)   
                
   # Sample input strings
