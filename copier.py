# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 00:09:54 2024

@author: chris
"""
import os
import shutil
import re

source_directory='Z:/Bausch_Group/Sophie_Dataserver/2024_03_26_11_07_31--Patternoids_PDAC-subtypes 8028_WT1_5mgmL_dt1h_3days_II/Sequence 001/1'
#source_directory='D:/9591'
destination_directory='D:/Sequence001'
folder_name=''
for filename in os.listdir(source_directory):
    if filename.endswith(".tiff") or filename.endswith(".tif"):
        # Suche nach dem Muster im Dateinamen
        
        folder_path = os.path.join(destination_directory, folder_name)
            
           # Erstelle den Ordner, falls er noch nicht existiert
        if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # Pfade zur Quelle und zum Ziel
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(folder_path, filename)
            
            # Verschiebe die Datei
        shutil.copy(source_path, destination_path)
            
        print(f"copied {filename} to {folder_path}")

print("Sortierung abgeschlossen.")
        