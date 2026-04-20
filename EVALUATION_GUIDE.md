# PixelCNN Image Completion - Evaluation Guide

## 📊 Model Accuracy Results

### Validation Set Performance
After training your PixelCNN model, here are the accuracy metrics on the validation set:

- **Images Evaluated**: 158
- **Average MSE**: 0.021737
- **Average PSNR**: 16.82 dB
- **Average Pixel Accuracy**: 7.17%

### What These Metrics Mean

1. **MSE (Mean Squared Error)**: Measures the average squared difference between reconstructed and original pixels. Lower is better. Your value of 0.0217 is relatively low.

2. **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality in decibels. Higher is better. Your 16.82 dB indicates moderate quality reconstruction.

3. **Pixel Accuracy**: Percentage of pixels that exactly match the ground truth after quantization. Your 7.17% shows the model captures some exact pixels, but image completion is challenging and many plausible solutions exist.

### Important Notes

- **This is an image generation task**, not classification, so "accuracy" is just one metric
- Visual quality is often more important than exact pixel matches
- The model is designed to fill in missing regions, and there can be multiple valid completions
- PSNR and visual inspection are typically better indicators of performance for image completion

---

## 🚀 How to Use the Evaluation Tools

### 1. Create Validation Set (Done ✓)

The validation set has already been created:
```powershell
python create_validation_split.py
```

This split 15% of training data (158 pairs) into a validation set.

### 2. Run Evaluation

Evaluate on different datasets:

**Validation Set (recommended for metrics):**
```powershell
python evaluate.py --dataset val
```

**Training Set (sanity check):**
```powershell
python evaluate.py --dataset train
```

**Test Set (no ground truth, visual only):**
```powershell
python evaluate.py --dataset test
```

**Limit to first N images:**
```powershell
python evaluate.py --dataset val --limit 10
```

### 3. View Results Visually

Launch the interactive viewer:
```powershell
streamlit run viewer.py
```

This opens a web interface where you can:
- Browse all images with prev/next buttons
- Compare occluded, reconstructed, and original images side-by-side
- View per-image metrics (MSE, PSNR, pixel accuracy)
- See difference heatmaps
- Download images
- Switch between validation/training/test sets

---

## 📁 Project Structure

```
Assignment2/
├── data/
│   ├── train/              # Training data (after split)
│   │   ├── occluded_images/
│   │   └── original_images/
│   ├── val/                # Validation data (created by split)
│   │   ├── occluded_images/
│   │   └── original_images/
│   └── occluded_test/      # Test data (no ground truth)
├── results/                # Reconstructed images and metrics
│   ├── recon_*.jpg         # Reconstructed images
│   └── metrics_*.txt       # Evaluation metrics
├── saved_models/
│   └── pixelcnn_bedroom.pth  # Trained model
├── model.py                # PixelCNN architecture
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── viewer.py               # Streamlit visualization app
└── create_validation_split.py  # Create val set
```

---

## 🎯 Quick Start Guide

### To see your model's accuracy:

1. **Run evaluation on validation set:**
   ```powershell
   python evaluate.py --dataset val
   ```

2. **Check the summary at the end of the output:**
   ```
   ============================================================
   EVALUATION SUMMARY
   ============================================================
   Dataset: val
   Images evaluated: 158
   Average MSE: 0.021737
   Average PSNR: 16.82 dB
   Average Pixel Accuracy: 7.17% (0.0717)
   ============================================================
   ```

3. **View detailed metrics:**
   ```powershell
   cat results/metrics_val.txt
   ```

4. **Visualize results:**
   ```powershell
   streamlit run viewer.py
   ```

---

## 💡 Tips for Improving Accuracy

1. **Train for more epochs** - You trained for 50 epochs; try 100+
2. **Increase model capacity** - Use more filters or residual blocks
3. **Data augmentation** - Add random flips, crops during training
4. **Different loss functions** - Try perceptual loss or adversarial training
5. **Larger images** - Train on 128x128 instead of 64x64
6. **Better sampling** - Use temperature sampling instead of argmax

---

## 🔍 Understanding Your Results

### Why is pixel accuracy only 7.17%?

This is **normal and expected** for image completion tasks because:

1. **Image completion is ambiguous** - Many valid ways to fill occluded regions
2. **Colors vary slightly** - A pixel can be RGB(100,120,140) or RGB(101,121,139) and look identical but count as "wrong"
3. **The metric is strict** - Requires exact match of all 3 color channels
4. **PSNR is more meaningful** - 16.82 dB shows reasonable reconstruction quality

### Better ways to evaluate:

1. **Visual inspection** - Use the Streamlit viewer
2. **PSNR** - Industry standard for image quality
3. **SSIM** - Structural similarity (can be added)
4. **User studies** - Ask people which looks better

---

## 📝 Generated Files

After running evaluation, you'll find:

- `results/recon_*.jpg` - All reconstructed images
- `results/metrics_val.txt` - Detailed per-image metrics
- `results/metrics_train.txt` - Training set metrics (if evaluated)

---

## 🐛 Troubleshooting

**"No images found in directory"**
- Make sure you ran `create_validation_split.py` first

**"Reconstruction not found"**
- Run `python evaluate.py --dataset val` before opening the viewer

**Low accuracy**
- This is normal! Image completion is hard. Focus on PSNR and visual quality.

**Streamlit errors**
- Make sure streamlit is installed: `pip install streamlit`
- Check you're in the correct directory

---

## 📚 Additional Resources

- [PixelCNN Paper](https://arxiv.org/abs/1606.05328)
- [Image Quality Metrics](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Author**: Auto-generated evaluation suite
**Date**: October 2025
