import os
import requests
import pandas as pd

# Function to download images
def download_images(image_links, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, link in enumerate(image_links, 1):
        try:
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(output_folder, f'image_{idx}.jpg')
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Downloaded: {file_path}")
            else:
                print(f"Failed to download {link}")
        except Exception as e:
            print(f"Error downloading {link}: {e}")

# Google Sheet ID (extracted from the URL)
SHEET_ID = '121aV7BjJqCRlFcVegbbhI1Zmt67wG61ayRiFtDnafKY'

# Load Google Sheet as a DataFrame
# Construct the URL for CSV export:
sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
df = pd.read_csv(sheet_url)

# Extract image links from Column B (adjust column name or index if necessary)
image_links = df.iloc[:, 1].dropna()  # Assuming links are in the second column (Column B)

# Download images to dataset folder
output_folder = "dataset"
download_images(image_links, output_folder)
import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()
    preprocessed_img = preprocess_input(expanded_img_array)
# ... (extract_features function remains the same) ...

dataset_folder = "/content/dataset"
feature_list = []

for filename in os.listdir(dataset_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(dataset_folder, filename)

        # Extract features and store them
        features = extract_features(img_path)  # Call extract_features here
        feature_list.append(features)  # Append to the feature list
# Function to find the top N similar images
def find_similar_images(new_img_path, feature_list, image_paths, top_n=5):
    # Extract features from the new image
    new_image_features = extract_features(new_img_path)

    # Compute cosine similarity between the new image features and all dataset features
    similarities = cosine_similarity(
        [new_image_features],  # Reshape to 2D array for compatibility
        feature_list
    )[0]  # Get the first row of similarity scores

    # Get the indices of the top N most similar images
    top_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort in descending order

    # Retrieve the paths of the top N similar images
    similar_image_paths = [image_paths[i] for i in top_indices]

    return similar_image_paths, similarities[top_indices]

# Create a list of image paths corresponding to your dataset images
image_paths = [
    os.path.join(dataset_folder, filename)
    for filename in os.listdir(dataset_folder)
    if filename.endswith((".jpg", ".jpeg", ".png"))
]

# Path to the new image
new_image_path = "/content/new_image.jpg"

# Find the 5 most similar images
similar_images, scores = find_similar_images(new_image_path, feature_list, image_paths, top_n=5)

# Print results
print("Top 5 similar images:")
for i, (path, score) in enumerate(zip(similar_images, scores)):
    print(f"{i+1}. Path: {path}, Similarity: {score:.4f}")
import matplotlib.pyplot as plt
def find_similar_images(new_img_path):
    """
    Find and print the top 5 most similar images in the dataset based on cosine similarity.

    Args:
    - new_img_path (str): Path to the new image.

    Returns:
    - list of str: Paths of the 5 most similar images.
    """
    # Extract features of the new image
    new_image_features = extract_features(new_img_path)

    # Compute cosine similarity between the new image and the dataset images
    similarities = cosine_similarity(
        [new_image_features],  # Reshape to 2D array for compatibility
        feature_list
    )[0]  # Get the first row of similarity scores

    # Get the indices of the top 5 most similar images
    top_indices = np.argsort(similarities)[-5:][::-1]  # Sort in descending order

    # Retrieve the paths of the top 5 similar images
    top_paths = [os.path.join(dataset_folder, os.listdir(dataset_folder)[i]) for i in top_indices]

    # Print the top 5 similar image paths
    print("Top 5 similar images:")
    for i, path in enumerate(top_paths):
        print(f"{i+1}. {path}")
    for i, path in enumerate(top_paths):
      img = cv2.imread(path)  # Read the image from the path
      if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display in matplotlib
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Similar Image {i+1}")
        plt.show()
      else:
        print(f"Unable to open image: {path}")
