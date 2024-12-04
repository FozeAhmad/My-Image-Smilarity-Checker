import os
import requests
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import cv2
import pickle
import streamlit as st

# Streamlit UI
st.title("Clustered Image Similarity Finder")

DATASET_FOLDER = "dataset"
CLUSTER_MODEL_FILE = "clusters.pkl"
NUM_CLUSTERS = 20  # Adjust as needed

# Function to download images
def download_images_from_sheet(sheet_id, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(sheet_url)

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

# Pre-load VGG16 model
@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

model = load_model()

# Function to extract features
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Precompute features and clusters
@st.cache_data
def prepare_clusters(folder, num_clusters):
    feature_list = []
    image_paths = []

    # Extract features for all images
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder, filename)
            features = extract_features(img_path)
            feature_list.append(features)
            image_paths.append(img_path)

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(feature_list)

    # Save cluster assignments, features, and image paths
    clusters = {
        "kmeans": kmeans,
        "features": np.array(feature_list),
        "image_paths": image_paths,
        "assignments": cluster_assignments,
    }
    with open(CLUSTER_MODEL_FILE, "wb") as file:
        pickle.dump(clusters, file)

    return clusters

# Load precomputed clusters
@st.cache_data
def load_clusters():
    if os.path.exists(CLUSTER_MODEL_FILE):
        with open(CLUSTER_MODEL_FILE, "rb") as file:
            return pickle.load(file)
    else:
        return prepare_clusters(DATASET_FOLDER, NUM_CLUSTERS)

clusters = load_clusters()

# Main functionality
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded image locally
    uploaded_image_path = "uploaded_image.jpg"
    with open(uploaded_image_path, "wb") as file:
        file.write(uploaded_file.read())

    # Extract features of the uploaded image
    uploaded_features = extract_features(uploaded_image_path).reshape(1, -1)

    # Assign to nearest cluster
    assigned_cluster = clusters["kmeans"].predict(uploaded_features)[0]

    # Get images in the same cluster
    cluster_indices = np.where(clusters["assignments"] == assigned_cluster)[0]
    cluster_features = clusters["features"][cluster_indices]
    cluster_image_paths = [clusters["image_paths"][i] for i in cluster_indices]

    # Compute similarity within the cluster
    similarities = cosine_similarity(uploaded_features, cluster_features)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]  # Top 5 similar images

    # Display results
    st.image(uploaded_image_path, caption="Uploaded Image", use_column_width=True)
    st.write("Top 5 similar images:")
    for idx in top_indices:
        similar_image_path = cluster_image_paths[idx]
        similarity_score = similarities[idx]
        img = cv2.imread(similar_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption=f"Score: {similarity_score:.4f}", use_column_width=True)
