import os
import requests
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import streamlit as st

# Streamlit UI
st.title("Image Similarity Finder")

# Define constants
DATASET_FOLDER = "dataset"

# Function to download images from a Google Sheet
def download_images_from_sheet(sheet_id, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load Google Sheet as a DataFrame
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(sheet_url)

    # Extract image links from the second column (adjust if necessary)
    image_links = df.iloc[:, 1].dropna()

    for idx, link in enumerate(image_links, 1):
        try:
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(output_folder, f'image_{idx}.jpg')
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                st.write(f"Downloaded: {file_path}")
            else:
                st.warning(f"Failed to download: {link}")
        except Exception as e:
            st.error(f"Error downloading {link}: {e}")

# Function to extract features using VGG16
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Pre-load VGG16 model
@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

model = load_model()

# Load features and image paths
@st.cache_data
def prepare_dataset(folder):
    feature_list = []
    image_paths = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder, filename)
            features = extract_features(img_path)
            feature_list.append(features)
            image_paths.append(img_path)
    return np.array(feature_list), image_paths

# Main Functionality
# Download dataset
SHEET_ID = '121aV7BjJqCRlFcVegbbhI1Zmt67wG61ayRiFtDnafKY'
download_images_from_sheet(SHEET_ID, DATASET_FOLDER)

# Prepare dataset
st.write("Preparing dataset...")
feature_list, image_paths = prepare_dataset(DATASET_FOLDER)

from sklearn.cluster import KMeans
import pickle

# Function to create clusters
@st.cache_data
def create_clusters(features, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels

# Cluster the dataset and save cluster information
st.write("Clustering dataset...")
kmeans, cluster_labels = create_clusters(feature_list)

# Save clusters and cluster labels for efficient access
cluster_data = {i: [] for i in range(kmeans.n_clusters)}
for idx, label in enumerate(cluster_labels):
    cluster_data[label].append(image_paths[idx])

# Upload image and find similar images within its cluster
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded image locally
    uploaded_image_path = "uploaded_image.jpg"
    with open(uploaded_image_path, "wb") as file:
        file.write(uploaded_file.read())

    # Extract features of the uploaded image
    new_image_features = extract_features(uploaded_image_path).reshape(1, -1)

    # Assign uploaded image to a cluster
    query_cluster = kmeans.predict(new_image_features)[0]
    st.write(f"Uploaded image belongs to Cluster {query_cluster}")

    # Get images and features from the same cluster
    cluster_image_paths = cluster_data[query_cluster]
    cluster_features = np.array([feature_list[image_paths.index(p)] for p in cluster_image_paths])

    # Compute cosine similarity within the cluster
    similarities = cosine_similarity(new_image_features, cluster_features)[0]

    # Get indices of top 5 similar images
    top_indices = np.argsort(similarities)[-5:][::-1]

    # Display results
    st.image(uploaded_image_path, caption="Uploaded Image", use_column_width=True)
    st.write("Top 5 similar images from the cluster:")
    for idx in top_indices:
        similar_image_path = cluster_image_paths[idx]
        similarity_score = similarities[idx]
        img = cv2.imread(similar_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption=f"Score: {similarity_score:.4f}", use_column_width=True)
