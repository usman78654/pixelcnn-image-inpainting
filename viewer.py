import streamlit as st
import os
from PIL import Image
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = './data'
RESULTS_DIR = './results'

# Page config
st.set_page_config(page_title="PixelCNN Results Viewer", layout="wide")

st.title("🖼️ PixelCNN Image Completion Results Viewer")
st.markdown("Compare occluded, reconstructed, and original images side-by-side")

# Sidebar for dataset selection
st.sidebar.header("Settings")
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Validation Set", "Training Set", "Test Set (No Ground Truth)"],
    index=0
)

# Map selection to directories
if dataset == "Validation Set":
    occ_dir = os.path.join(DATA_DIR, 'val', 'occluded_images')
    orig_dir = os.path.join(DATA_DIR, 'val', 'original_images')
    has_gt = True
    st.sidebar.info("📊 Validation set has ground truth for comparison")
elif dataset == "Training Set":
    occ_dir = os.path.join(DATA_DIR, 'train', 'occluded_images')
    orig_dir = os.path.join(DATA_DIR, 'train', 'original_images')
    has_gt = True
    st.sidebar.info("📊 Training set has ground truth for comparison")
else:  # Test Set
    occ_dir = os.path.join(DATA_DIR, 'occluded_test')
    orig_dir = None
    has_gt = False
    st.sidebar.warning("⚠️ Test set has no ground truth - visual inspection only")

# Check if directories exist
if not os.path.exists(occ_dir):
    st.error(f"❌ Directory not found: {occ_dir}")
    st.stop()

# Get list of files
occ_files = sorted([f for f in os.listdir(occ_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if len(occ_files) == 0:
    st.error(f"❌ No images found in {occ_dir}")
    st.stop()

# Display dataset stats
st.sidebar.markdown("---")
st.sidebar.header("Dataset Statistics")
st.sidebar.metric("Total Images", len(occ_files))

# Load metrics if available
metrics_file = os.path.join(RESULTS_DIR, f'metrics_{dataset.split()[0].lower()}.txt')
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics_text = f.read()
        # Extract summary metrics
        for line in metrics_text.split('\n'):
            if 'Average MSE:' in line:
                mse = line.split(':')[1].strip()
                st.sidebar.metric("Avg MSE", mse)
            elif 'Average PSNR:' in line:
                psnr = line.split(':')[1].strip().split()[0]
                st.sidebar.metric("Avg PSNR", f"{psnr} dB")
            elif 'Average Pixel Accuracy:' in line:
                acc = line.split(':')[1].strip().split()[0]
                acc_pct = line.split('(')[1].split(')')[0] if '(' in line else ''
                st.sidebar.metric("Avg Pixel Accuracy", acc_pct)

# Image selection
st.sidebar.markdown("---")
st.sidebar.header("Image Selection")

# Add navigation options
nav_mode = st.sidebar.radio("Navigation", ["Select by Index", "Browse Sequentially"])

if nav_mode == "Select by Index":
    selected_idx = st.sidebar.number_input(
        "Image Index",
        min_value=0,
        max_value=len(occ_files)-1,
        value=0,
        step=1
    )
else:
    # Sequential browsing with prev/next buttons
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("⬅️ Prev"):
        st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
    if col3.button("Next ➡️"):
        st.session_state.current_idx = min(len(occ_files)-1, st.session_state.current_idx + 1)
    
    col2.markdown(f"**{st.session_state.current_idx + 1}/{len(occ_files)}**")
    selected_idx = st.session_state.current_idx

# Get filenames
occ_filename = occ_files[selected_idx]
if occ_filename.startswith('occluded_'):
    orig_filename = occ_filename.replace('occluded_', '', 1)
    recon_filename = occ_filename.replace('occluded_', 'recon_')
else:
    orig_filename = occ_filename
    recon_filename = 'recon_' + occ_filename

# Display filename
st.markdown(f"### Viewing: `{occ_filename}` (Image {selected_idx + 1} of {len(occ_files)})")

# Load images
occ_path = os.path.join(occ_dir, occ_filename)
recon_path = os.path.join(RESULTS_DIR, recon_filename)

occ_img = Image.open(occ_path).convert('RGB')
recon_exists = os.path.exists(recon_path)

if recon_exists:
    recon_img = Image.open(recon_path).convert('RGB')
else:
    st.warning(f"⚠️ Reconstruction not found at {recon_path}. Run evaluate.py first!")
    recon_img = None

if has_gt and orig_dir:
    orig_path = os.path.join(orig_dir, orig_filename)
    if os.path.exists(orig_path):
        orig_img = Image.open(orig_path).convert('RGB')
        # Resize original to match reconstruction size for comparison
        if recon_img:
            orig_img = orig_img.resize(recon_img.size, Image.Resampling.LANCZOS)
    else:
        st.warning(f"⚠️ Original not found: {orig_filename}")
        orig_img = None
else:
    orig_img = None

# Display images in columns
if has_gt and orig_img:
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("#### 🔲 Occluded Input")
        st.image(occ_img, use_column_width=True)
        st.caption(f"Filename: {occ_filename}")
    
    with cols[1]:
        st.markdown("#### ✨ Reconstruction")
        if recon_img:
            st.image(recon_img, use_column_width=True)
            st.caption(f"Generated by PixelCNN")
        else:
            st.info("Run evaluation first")
    
    with cols[2]:
        st.markdown("#### ✅ Ground Truth")
        st.image(orig_img, use_column_width=True)
        st.caption(f"Filename: {orig_filename}")
    
    # Compute and display pixel-level metrics
    if recon_img:
        st.markdown("---")
        st.subheader("📊 Image-Level Metrics")
        
        occ_arr = np.array(occ_img).astype(float) / 255.0
        recon_arr = np.array(recon_img).astype(float) / 255.0
        orig_arr = np.array(orig_img).astype(float) / 255.0
        
        # MSE
        mse = np.mean((recon_arr - orig_arr) ** 2)
        # PSNR
        if mse > 0:
            psnr = 20 * np.log10(1.0) - 10 * np.log10(mse)
        else:
            psnr = float('inf')
        # Pixel accuracy (exact match after quantization)
        recon_quant = (recon_arr * 255).round().astype(int)
        orig_quant = (orig_arr * 255).round().astype(int)
        pixel_acc = np.mean(recon_quant == orig_quant)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.6f}")
        col2.metric("PSNR", f"{psnr:.2f} dB")
        col3.metric("Pixel Accuracy", f"{pixel_acc*100:.2f}%")
        
        # Show difference heatmap
        st.markdown("#### 🔥 Difference Heatmap (Reconstruction vs Ground Truth)")
        diff = np.abs(recon_arr - orig_arr)
        diff_gray = np.mean(diff, axis=2)  # Average across RGB channels
        
        # Normalize to 0-255 for display
        diff_display = (diff_gray * 255).astype(np.uint8)
        
        # Create heatmap
        from PIL import Image as PILImage
        diff_img = PILImage.fromarray(diff_display, mode='L')
        
        st.image(diff_img, use_column_width=True, caption="Brighter = larger difference", clamp=True)

else:
    # Test set or no ground truth - just show occluded and reconstruction
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("#### 🔲 Occluded Input")
        st.image(occ_img, use_column_width=True)
        st.caption(f"Filename: {occ_filename}")
    
    with cols[1]:
        st.markdown("#### ✨ Reconstruction")
        if recon_img:
            st.image(recon_img, use_column_width=True)
            st.caption(f"Generated by PixelCNN")
        else:
            st.info("Run evaluation first")

# Add download buttons
st.markdown("---")
st.subheader("💾 Download Images")
dl_cols = st.columns(3 if has_gt and orig_img else 2)

with dl_cols[0]:
    with open(occ_path, 'rb') as f:
        st.download_button("⬇️ Download Occluded", f.read(), file_name=f"occluded_{occ_filename}", mime="image/jpeg")

if recon_exists:
    with dl_cols[1]:
        with open(recon_path, 'rb') as f:
            st.download_button("⬇️ Download Reconstruction", f.read(), file_name=f"recon_{occ_filename}", mime="image/jpeg")

if has_gt and orig_img:
    orig_path = os.path.join(orig_dir, orig_filename)
    with dl_cols[2]:
        with open(orig_path, 'rb') as f:
            st.download_button("⬇️ Download Ground Truth", f.read(), file_name=f"original_{orig_filename}", mime="image/jpeg")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>PixelCNN Image Completion Viewer | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
