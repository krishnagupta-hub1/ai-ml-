import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import webcolors

# Function to find the closest color name
def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# Function to detect dominant color
def get_dominant_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    pixels = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    return closest_color(dominant_color)

# Folder containing Pokémon images
image_folder = 'C:\Users\Krishna Gupta\Documents\PokemonDataset\images'

# Data storage
pokemon_data = []

# Loop through images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        pokemon_name = os.path.splitext(filename)[0]  # Filename without extension
        image_path = os.path.join(image_folder, filename)
        
        # Get dominant color
        dominant_color = get_dominant_color(image_path)
        
        # Append to data
        pokemon_data.append({'Pokemon': pokemon_name, 'Color': dominant_color})

# Save to CSV
output_df = pd.DataFrame(pokemon_data)
output_df.to_csv('pokemon_classification.csv', index=False)

print("Classification completed. Output saved to 'pokemon_classification.csv'.")
