# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:43:00 2024

@author: chris
"""
import os
import shutil

# Paths to your folder and text file
folder_path = 'C:/Users/chris/OneDrive/Desktop/Sophies_test_data'
text_file_path = 'C:/Users/chris/OneDrive/Desktop/filenames.txt'
#"C:\Users\chris\OneDrive\Desktop\filenames.txt"
# Read the mapping from the text file
with open(text_file_path, 'r') as f:
    mappings = f.readlines()

# Process each line in the text file
for line in mappings:
    # Split the line into the old filename and the new filename
    old_name, new_name = line.strip().split(': ')
    old_name = old_name.strip()  # Remove any extra spaces
    new_name = new_name.strip("'")  # Remove the quotes around the new filename

    # Create the full path to the old file
    old_file_path = os.path.join(folder_path, old_name)

    # Create the new filename with the .jpg extension
    new_file_name = new_name + '.jpg'

    # Create the full path for the new file
    new_file_path = os.path.join(folder_path, new_file_name)

    # Rename the file
    shutil.move(old_file_path, new_file_path)

print("Files renamed successfully.")
