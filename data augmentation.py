# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:52:49 2024

@author: chris
"""
import os
from PIL import Image
#import cv2

# Pfade festlegen
input_dir = "C:/training_dataset/8028/"
output_dir = "C:/training_datasetaug/8028/"

# Sicherstellen, dass der Ausgabeordner existiert
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Alle JPEG-Dateien im Verzeichnis durchgehen
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)

        # Original speichern
        img.save(os.path.join(output_dir, filename))
counter=0
n=len(os.listdir(input_dir))
nmax=10103-n
for filename in os.listdir(input_dir):
    if counter<=nmax:
        if filename.endswith(".jpg"):
            # Spiegeln
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
            mirrored.save(os.path.join(output_dir, f"{filename[:-4]}_mirrored.jpg"))
            counter+=1
print("mirrored complete")
#for filename in os.listdir(input_dir):
#    if filename.endswith(".jpg"):
#        img_path = os.path.join(input_dir, filename)
#        img = Image.open(img_path)
#        # 90° Drehungen
#        for i, angle in enumerate([90, 180, 270], start=1):
#            rotated = img.rotate(angle)
#            rotated.save(os.path.join(output_dir, f"{filename[:-4]}_rotated_{angle}.jpg"))
print("angled complete")
#for filename in os.listdir(input_dir):
#    if filename.endswith(".jpg"):
#        img_path = os.path.join(input_dir, filename)
#        img = Image.open(img_path)
#        # Cropping
#        width, height = img.size
#        crop_size = (int(width * 0.8), int(height * 0.8))  # 80% des Originalbildes
#        left = (width - crop_size[0]) // 2
#        top = (height - crop_size[1]) // 2
#        right = (width + crop_size[0]) // 2
#        bottom = (height + crop_size[1]) // 2
#        cropped = img.crop((left, top, right, bottom))
#        cropped.save(os.path.join(output_dir, f"{filename[:-4]}_cropped.jpg"))

        # Optional: Weitere Kombinationen
        #mirrored_rotated = mirrored.rotate(90)
        #mirrored_rotated.save(os.path.join(output_dir, f"{filename[:-4]}_mirrored_rotated_90.jpg"))

print("Data Augmentation abgeschlossen!")
print(len(os.listdir(output_dir)))
