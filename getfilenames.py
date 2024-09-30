# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:10:12 2024

@author: chris
"""
import os

def get_file_names_in_folder(folder_path):
    # List to store file names
    file_names = []

    # Iterate over all the files in the given folder
    for file_name in os.listdir(folder_path):
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)

    return file_names

# Example usage
folder_path = 'C:/training_dataset - Kopie/training 2/testing'  # Replace with your folder path
file_list = get_file_names_in_folder(folder_path)
print(file_list)
print(len(file_list))
