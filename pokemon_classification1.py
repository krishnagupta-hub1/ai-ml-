import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

# Define Pokémon color categories
POKEMON_COLORS = {
    'Red': [(200, 0, 0), (255, 0, 0)],
    'Blue': [(0, 0, 200), (0, 0, 255)],
    'Green': [(0, 200, 0), (0, 255, 0)],
    'Yellow': [(255, 255, 0), (200, 200, 0)],
    'Purple': [(128, 0, 128), (75, 0, 130)],
    'Pink': [(255, 192, 203), (255, 105, 180)],
    'Gray': [(128, 128, 128), (169, 169, 169)],
    'Black': [(0, 0, 0), (20, 20, 20)],
    'White': [(255, 255, 255), (240, 240, 240)]
}

# Function to determine the closest Pokémon color
def closest_pokemon_color(requested_color):
    min_dist = float('inf')
    closest_color = 'Unknown'
    for color_name, shades in POKEMON_COLORS.items():
        for shade in shades:
            distance = np.linalg.norm(np.array(requested_color) - np.array(shade))
            if distance < min_dist:
                min_dist = distance
                closest_color = color_name
    return closest_color

# Function to detect dominant color
def get_dominant_color(image_path, k=3):
    image = cv2.imread(image_path)
    if image is None:
        return 'Unknown'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (100, 100))
    pixels = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    # Count occurrences of each color
    color_counts = Counter(kmeans.labels_)
    dominant_color = colors[color_counts.most_common(1)[0][0]]
    
    return closest_pokemon_color(dominant_color)

# Folder containing Pokémon images
image_folder = r'C:\Users\Krishna Gupta\Documents\PokemonDataset\images'
  # Update as needed

# Data storage
pokemon_data = []

# Loop through images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        pokemon_name = os.path.splitext(filename)[0]  # Filename without extension
        image_path = os.path.join(image_folder, filename)
        
        try:
            dominant_color = get_dominant_color(image_path)
            pokemon_data.append({'Pokemon': pokemon_name, 'Color': dominant_color})
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save to CSV
output_df = pd.DataFrame(pokemon_data)
output_df.to_csv('pokemon_classification1.csv', index=False)

print("Classification completed. Output saved to 'pokemon_classification1.csv'.")
