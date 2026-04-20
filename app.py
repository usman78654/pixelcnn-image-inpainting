import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from model import PixelCNN

# --- Configuration ---
MODEL_PATH = './saved_models/pixelcnn_bedroom.pth'
DATA_DIR = './data'
IMAGE_SIZE = 64

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """Load the pre-trained PixelCNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PixelCNN(in_channels=3, n_filters=128, n_blocks=5, output_bins=256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image):
    """Preprocess the uploaded image for the model."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def tensor_to_pil(tensor):
    """Convert a PyTorch tensor to a PIL image."""
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🖼️ PixelRNN/CNN Image Completion")

st.info(
    "Upload an occluded image to see the model reconstruct the missing parts. "
    "The model uses a PixelCNN architecture trained on a bedroom dataset."
)

# --- Load Model ---
with st.spinner('Loading the generative model...'):
    model, device = load_model()

# --- Image Uploader ---
uploaded_file = st.file_uploader(
    "Choose an occluded image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # --- Display Images ---
    col1, col2, col3 = st.columns(3)

    # Display Occluded Image
    with col1:
        st.header("1. Occluded Input")
        occluded_pil = Image.open(uploaded_file).convert('RGB')
        st.image(occluded_pil, caption='Uploaded Occluded Image', use_column_width=True)

    # --- Model Inference ---
    with col2:
        st.header("2. Model Output")
        with st.spinner('Reconstructing the image...'):
            occluded_tensor = preprocess_image(occluded_pil).to(device)
            
            # The .sample() method is very slow. A faster way for a quick demo
            # is to do a single forward pass and take the argmax.
            # For this app, we will use the slow but more accurate sampling method.
            reconstructed_tensor = model.sample(occluded_tensor)

            reconstructed_pil = tensor_to_pil(reconstructed_tensor)
        st.image(reconstructed_pil, caption='Reconstructed by PixelCNN', use_column_width=True)

    # --- Display Ground Truth ---
    with col3:
        st.header("3. Ground Truth")
        # Try to find the corresponding original image by filename
        original_img_path = os.path.join(DATA_DIR, 'train', uploaded_file.name)
        if os.path.exists(original_img_path):
            original_pil = Image.open(original_img_path)
            st.image(original_pil, caption='Original Image', use_column_width=True)
        else:
            st.warning("Could not find the corresponding original image in the dataset.")

else:
    st.info("Please upload an image to begin.")