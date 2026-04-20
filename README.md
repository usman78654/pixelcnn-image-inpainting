# PixelCNN Image Inpainting

A generative AI model for bedroom image inpainting using PixelCNN. This project demonstrates how to complete partially occluded images using a trained PixelCNN model.

## Features

- 🎨 Image inpainting for bedroom scenes
- 🧠 PixelCNN-based generative model
- 📊 Model evaluation with MSE, PSNR, and Pixel Accuracy metrics
- 🎯 Interactive Streamlit application for real-time inference
- 📈 Validation set with ground truth comparisons

## Model Performance

| Metric | Value | Rating |
|--------|-------|--------|
| MSE | 0.0217 | ⭐⭐⭐ Good |
| PSNR | 16.82 dB | ⭐⭐⭐ Moderate |
| Pixel Accuracy | 7.17% | ⭐⭐ Normal |

**Dataset**: Validation Set (158 images)

## Project Structure

```
├── app.py                      # Main Streamlit application
├── model.py                    # PixelCNN model architecture
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── viewer.py                   # Interactive image viewer
├── dataset.py                  # Dataset utilities
├── create_validation_split.py  # Validation set creation
├── requirements.txt            # Project dependencies
├── data/
│   ├── train/
│   │   ├── original_images/
│   │   └── occluded_images/
│   ├── val/
│   │   ├── original_images/
│   │   └── occluded_images/
│   └── occluded_test/
├── saved_models/
│   └── pixelcnn_bedroom.pth   # Pre-trained model
├── results/
│   └── metrics_val.txt        # Evaluation metrics
└── docs/
    ├── ACCURACY_SUMMARY.md    # Model accuracy report
    ├── EVALUATION_GUIDE.md    # Evaluation instructions
    └── QUICK_REFERENCE.md     # Quick command reference
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/usman78654/pixelcnn-image-inpainting.git
cd pixelcnn-image-inpainting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Application (Recommended)

Run the Streamlit app for an interactive experience:
```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Image Viewer

Inspect validation results with the interactive viewer:
```bash
python -m streamlit run viewer.py
```

### Evaluation

Evaluate the model on different datasets:
```bash
# Validation set (with ground truth)
python evaluate.py --dataset val

# Training set
python evaluate.py --dataset train

# Test set (visual only)
python evaluate.py --dataset test
```

### Training

To retrain the model:
```bash
python train.py
```

## Model Architecture

PixelCNN is an autoregressive generative model that learns to predict pixel values sequentially, conditioned on previously generated pixels. The architecture includes:

- Multiple masked convolutional layers for autoregressive generation
- 128 filters and 5 residual blocks
- 256 output bins for pixel value distribution

## Requirements

See `requirements.txt` for full dependencies. Key requirements:

- Python 3.8+
- PyTorch
- Streamlit
- Torchvision
- NumPy
- Pillow

## Results

The trained model achieves:
- **MSE**: 0.0217 - Good reconstruction quality
- **PSNR**: 16.82 dB - Moderate image quality
- **Pixel Accuracy**: 7.17% - Normal for generative models

Note: Pixel accuracy is strict (requires exact RGB match). Visual quality is more important than pixel-level metrics for generative tasks.

## Dataset

The project uses a bedroom image dataset with occluded regions. The dataset is split into:
- **Training**: 898 image pairs
- **Validation**: 158 image pairs
- **Test**: 195 images (visual evaluation only)

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License

This project is open source and available under the MIT License.

## Author

Created as part of a Generative AI assignment.

## Quick Commands

```powershell
# View model metrics
cat results/metrics_val.txt

# Run the web app
python -m streamlit run app.py

# View reconstructed images
python -m streamlit run viewer.py
```

## Documentation

For more detailed information, see:
- [Accuracy Summary](./ACCURACY_SUMMARY.md)
- [Evaluation Guide](./EVALUATION_GUIDE.md)
- [Quick Reference](./QUICK_REFERENCE.md)
