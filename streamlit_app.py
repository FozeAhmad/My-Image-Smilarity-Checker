import streamlit as st
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Initialize the model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Feature extraction function
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

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
                st.write(f"Downloaded: {file_path}")
            else:
                st.warning(f"Failed to download {link}")
        except Exception as e:
            st.error(f"Error downloading {link}: {e}")

# Function to find similar images
def find_similar_images(new_img_path, feature_list, image_paths, top_n=5):
    new_image_features = extract_features(new_img_path)
    similarities = cosine_similarity(
        [new_image_features], feature_list
    )[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    similar_image_paths = [image_paths[i] for i in top_indices]
    return similar_image_paths, similarities[top_indices]

# Streamlit UI
st.title("Image Similarity Search")
st.sidebar.header("Upload and Dataset Options")

dataset_folder = "dataset"
feature_list = []
image_paths = []

# Step 1: Upload or Download Dataset
st.sidebar.subheader("Step 1: Load Dataset")
use_google_sheet = st.sidebar.checkbox("Use Google Sheet for image links", value=True)
if use_google_sheet:
    sheet_url = st.sidebar.text_input("Google Sheet CSV URL", value="")
    if st.sidebar.button("Download Images"):
        if sheet_url:
            df = pd.read_csv(sheet_url)
            image_links = df.iloc[:, 1].dropna()
            download_images(image_links, dataset_folder)
        else:
            st.warning("Please provide a valid Google Sheet CSV URL.")

# Step 2: Extract Features
st.sidebar.subheader("Step 2: Extract Features")
if st.sidebar.button("Extract Features"):
    if os.path.exists(dataset_folder):
        for filename in os.listdir(dataset_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(dataset_folder, filename)
                features = extract_features(img_path)
                feature_list.append(features)
                image_paths.append(img_path)
        st.success("Feature extraction completed.")
    else:
        st.warning("Dataset folder does not exist. Please load dataset first.")

# Step 3: Upload Image and Search
st.sidebar.subheader("Step 3: Find Similar Images")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    uploaded_img_path = os.path.join("temp", uploaded_file.name)
    with open(uploaded_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.sidebar.button("Find Similar Images"):
        if feature_list and image_paths:
            similar_images, scores = find_similar_images(
                uploaded_img_path, feature_list, image_paths, top_n=5
            )
            st.subheader("Top 5 Similar Images:")
            for i, (path, score) in enumerate(zip(similar_images, scores)):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, caption=f"Similarity: {score:.4f}", use_column_width=True)
        else:
            st.warning("Please load the dataset and extract features first.")
