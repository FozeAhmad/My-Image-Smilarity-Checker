import streamlit as st
import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd

# Initialize the model
@st.cache_resource
def load_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

model = load_model()

# Function to extract features
@st.cache_data
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Load and preprocess dataset (run once)
@st.cache_data
def prepare_dataset():
    # Google Sheet ID
    SHEET_ID = '121aV7BjJqCRlFcVegbbhI1Zmt67wG61ayRiFtDnafKY'
    sheet_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
    df = pd.read_csv(sheet_url)

    # Extract image links
    image_links = df.iloc[:, 1].dropna()

    # Download images
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    for idx, link in enumerate(image_links, 1):
        try:
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(dataset_folder, f"image_{idx}.jpg")
                with open(file_path, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
            else:
                print(f"Failed to download {link}")
        except Exception as e:
            print(f"Error downloading {link}: {e}")

    # Extract features and create feature list
feature_list = []
image_paths = []

for filename in os.listdir(dataset_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(dataset_folder, filename)

        # Extract features
        try:
            features = extract_features(img_path)
            feature_list.append(features)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Ensure feature_list is a 2D array
feature_list = np.array(feature_list)
if feature_list.ndim == 1:
    feature_list = feature_list.reshape(1, -1)  # Single feature reshaped to 2D

def find_similar_images(uploaded_img_path, feature_list, image_paths, top_n=5):
    if len(feature_list) == 0:
        st.error("Feature list is empty. Check dataset preparation.")
        return [], []

    # Extract features for the uploaded image
    new_image_features = extract_features(uploaded_img_path).reshape(1, -1)  # Ensure 2D

    # Compute cosine similarity
    similarities = cosine_similarity(new_image_features, feature_list)[0]

    # Get indices of top-N similar images
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    similar_image_paths = [image_paths[i] for i in top_indices]
    similar_scores = [similarities[i] for i in top_indices]

    return similar_image_paths, similar_scores
if uploaded_file is not None:
    try:
        # Save uploaded image locally
        with open("uploaded_image.jpg", "wb") as file:
            file.write(uploaded_file.getbuffer())

        # Perform similarity search
        similar_images, scores = find_similar_images(
            "uploaded_image.jpg", feature_list, image_paths
        )

        # Display results
        if similar_images:
            st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)
            st.write("Top 5 similar images:")
            for i, (path, score) in enumerate(zip(similar_images, scores)):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, caption=f"Similar Image {i+1} (Score: {score:.4f})", use_column_width=True)
        else:
            st.warning("No similar images found.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit App Interface
st.title("Image Similarity Search")
st.write("Upload an image to find the most similar images in the dataset.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("uploaded_image.jpg", "wb") as file:
        file.write(uploaded_file.getbuffer())

    # Find similar images
    similar_images, scores = find_similar_images("uploaded_image.jpg", feature_list, image_paths)

    # Display results
    st.image("uploaded_image.jpg", caption="Uploaded Image", use_column_width=True)
    st.write("Top 5 similar images:")
    for i, (path, score) in enumerate(zip(similar_images, scores)):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption=f"Similar Image {i+1} (Score: {score:.4f})", use_column_width=True)
