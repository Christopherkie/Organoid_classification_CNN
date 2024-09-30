# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:13:11 2024

@author: chris
"""

import os
import re
import shutil

# Pfad zu deinem Verzeichnis mit den TIFF-Dateien
source_directory = "/path/to/your/tiff/folder"
# Pfad zu deinem Zielverzeichnis, wo die sortierten Unterordner erstellt werden
destination_directory = "/path/to/your/sorted/folder"

# Dein Regex-Kriterium
pattern = r"your_regex_here"  # Ersetze 'your_regex_here' durch dein tatsächliches Regex-Muster

# Überprüfe, ob das Zielverzeichnis existiert, und erstelle es gegebenenfalls
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Durchlaufe alle Dateien im Quellverzeichnis
for filename in os.listdir(source_directory):
    if filename.endswith(".tiff") or filename.endswith(".tif"):
        # Suche nach dem Muster im Dateinamen
        match = re.search(pattern, filename)
        if match:
            # Verwende den gefundenen Text als Ordnername
            folder_name = match.group(0)
            # Erstelle den Pfad zum Zielordner
            folder_path = os.path.join(destination_directory, folder_name)
            
            # Erstelle den Ordner, falls er noch nicht existiert
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # Pfade zur Quelle und zum Ziel
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(folder_path, filename)
            
            # Verschiebe die Datei
            shutil.move(source_path, destination_path)
            
            print(f"Moved {filename} to {folder_path}")

print("Sortierung abgeschlossen.")