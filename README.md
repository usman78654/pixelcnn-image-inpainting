# PixelCNN Image Inpainting

A generative AI model for bedroom image inpainting using PixelCNN. This project demonstrates how to complete partially occluded images using a trained PixelCNN-based autoregressive model. The implementation includes a fully functional Streamlit web application for interactive image completion.

**Course:** Generative AI - Semester 7  
**Author:** Usman Tariq (22i-2459)  
**Assignment:** Image Completion using Autoregressive Models

## Features

- 🎨 Image inpainting for bedroom scenes using autoregressive generation
- 🧠 PixelCNN-based generative model with masked convolutions
- 📊 Comprehensive evaluation with MSE, PSNR, and Pixel Accuracy metrics
- 🎯 Interactive Streamlit web application for real-time inference
- 📈 Validation set with ground truth comparisons
- 📋 Detailed academic report with methodology and analysis

## Model Performance

| Metric | Value | Rating |
|--------|-------|--------|
| MSE | 0.0217 | ⭐⭐⭐ Good |
| PSNR | 16.82 dB | ⭐⭐⭐ Moderate |
| Pixel Accuracy | 7.17% | ⭐⭐ Normal |

**Dataset:** Validation Set (158 images from bedroom dataset)

**Note:** Pixel accuracy is a strict metric requiring exact RGB matches. The 7.17% is normal for generative models. Visual quality (reflected by PSNR) provides a better assessment of reconstruction quality. For detailed analysis, see [REPORT.md](./REPORT.md).

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

- **Input Layer:** Masked Convolution (Type A) - 7×7 kernel, 3→128 channels
- **Residual Blocks:** 5 blocks with bottleneck architecture
- **Intermediate Layer:** 1×1 convolution, 128→1024 channels
- **Output Layer:** 1×1 convolution, 1024→768 channels (3 RGB channels × 256 intensity bins)

**Total Parameters:** ~2.5M trainable parameters

**Key Components:**
- **Masked Convolutions:** Enforce autoregressive property (Type A prevents current pixel visibility, Type B allows it for deeper layers)
- **Residual Blocks:** Enable deeper architecture with skip connections
- **Sequential Generation:** Each pixel is sampled from the predicted categorical distribution

For detailed architecture explanation, see [REPORT.md - Section 2.3](./REPORT.md#23-model-architecture).

## Requirements

See `requirements.txt` for full dependencies. Key requirements:

- Python 3.8+
- PyTorch
- Streamlit
- Torchvision
- NumPy
- Pillow

**Training Configuration:**
- Optimizer: Adam (lr=1×10⁻⁴)
- Batch Size: 32
- Epochs: 50
- Loss Function: Cross-Entropy (averaged over RGB channels)
- Image Size: 64×64 pixels

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

### Full Assignment Report
For comprehensive methodology, results, and analysis, see **[REPORT.md](./REPORT.md)** which includes:
- Detailed problem formulation and background
- Complete methodology and dataset description
- Thorough model architecture explanation with code snippets
- Quantitative and qualitative results analysis
- Discussion of challenges and comparisons with alternative approaches
- Comprehensive references and appendices

### Quick Reference Files
- [ACCURACY_SUMMARY.md](./ACCURACY_SUMMARY.md) - Model accuracy report
- [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) - Evaluation instructions
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - Quick command reference
