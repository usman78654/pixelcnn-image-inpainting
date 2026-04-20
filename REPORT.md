# PixelCNN for Image Completion: A Deep Learning Approach

**Course:** Generative AI - Semester 7  
**Assignment:** Image Completion using Autoregressive Models  
**Date:** October 19, 2025  
**Author:** Usman Tariq (22i-2459)

---

## Abstract

This report presents the implementation and evaluation of a PixelCNN-based autoregressive model for image completion tasks. The model was trained on a bedroom image dataset to learn the distribution of pixel values and reconstruct occluded regions. Our implementation achieved a PSNR of 16.82 dB and an MSE of 0.0217 on the validation set, demonstrating the model's capability to generate plausible reconstructions for missing image regions. We analyze the model architecture, training methodology, and provide quantitative and qualitative evaluation of the results.

---

## 1. Introduction

### 1.1 Background

Image completion, also known as image inpainting, is a fundamental problem in computer vision that involves filling in missing or corrupted regions of an image with plausible content. This task has numerous applications including photo restoration, object removal, and image editing.

Traditional approaches relied on texture synthesis and diffusion-based methods, but recent advances in deep learning have enabled more sophisticated solutions.

### 1.2 Assignment Objective

The primary objective of this assignment was to:
- Implement a PixelCNN autoregressive model for image completion
- Train the model on a dataset of bedroom images with artificially occluded regions
- Evaluate the model's performance using quantitative metrics (MSE, PSNR, pixel accuracy)
- Analyze the quality and coherence of generated completions

### 1.3 PixelCNN Overview

PixelCNN is an autoregressive generative model introduced by van den Oord et al. (2016) that models the conditional distribution of pixels in an image. Unlike traditional feedforward networks, PixelCNN generates pixels sequentially, conditioning each pixel on all previously generated pixels. This is achieved through masked convolutions that ensure the model respects the autoregressive property. The key innovations are:

- **Masked Convolutions:** Ensure the model only sees context from previously generated pixels
- **Autoregressive Generation:** Models P(x) = ∏P(xi | x<i), enabling coherent generation
- **Discrete Pixel Distributions:** Treats each pixel intensity as a categorical distribution over 256 values

---

## 2. Methodology

### 2.1 Dataset Description

**Dataset Composition:**
- **Domain:** Bedroom interior images
- **Training Set:** 898 image pairs (occluded + original) after validation split
- **Validation Set:** 158 image pairs (15% split)
- **Test Set:** 195 occluded images (no ground truth)
- **Image Resolution:** 64×64 pixels
- **Color Space:** RGB (3 channels)

**Occlusion Pattern:**

The dataset consists of paired images where:
- **Occluded Images:** Original images with rectangular regions masked (replaced with white pixels)
- **Original Images:** Complete, unmodified bedroom scenes
- **Occlusion Variation:** The occluded regions vary in size and position to ensure the model learns robust completion patterns

**Data Split Strategy:**

We created a validation set by randomly selecting 15% of training pairs, ensuring:
- No data leakage between train and validation sets
- Sufficient training data (898 pairs) for model convergence
- Adequate validation data (158 pairs) for reliable performance estimation

### 2.2 Preprocessing Steps

1. **Image Resizing:** All images resized to 64×64 pixels using bilinear interpolation
2. **Normalization:** Pixel values normalized to [0, 1] range by dividing by 255
3. **Tensor Conversion:** Images converted to PyTorch tensors with shape (3, 64, 64)
4. **Data Augmentation:** None applied in this implementation (future improvement)
5. **Batch Processing:** Images batched with size 32 for efficient GPU utilization

**Preprocessing Pipeline:**
```
Raw Image (Variable Size, 0-255)
         ↓
   Resize to 64×64
         ↓
Convert to Tensor (H×W×C → C×H×W)
         ↓
 Normalize to [0, 1]
         ↓
  Batch (32 images)
         ↓
  Feed to Model
```

### 2.3 Model Architecture

#### 2.3.1 Overall Architecture

Our PixelCNN implementation consists of:
- **Input Layer:** Masked Convolution (Type A) - 7×7 kernel, 3→128 channels
- **Residual Blocks:** 5 blocks with bottleneck architecture
- **Intermediate Layer:** 1×1 convolution, 128→1024 channels
- **Output Layer:** 1×1 convolution, 1024→768 channels (3 channels × 256 bins)

**Total Parameters:** Approximately 2.5M trainable parameters

#### 2.3.2 Masked Convolution

The core innovation of PixelCNN is the masked convolution layer:

**Type A Mask (First Layer):**
- Prevents the model from seeing the current pixel when predicting it
- Ensures strict autoregressive property: P(xi | x<i)
- Applied only to the first convolutional layer

**Type B Mask (Subsequent Layers):**
- Allows the model to see the current pixel in deeper layers
- Maintains causality through the network depth
- Applied to all residual blocks

#### 2.3.3 Residual Blocks

Each residual block contains:
1. **Bottleneck:** 1×1 conv, 128→64 channels (reduces computation)
2. **Masked Conv:** 3×3 masked conv, 64→64 channels (spatial processing)
3. **Expansion:** 1×1 conv, 64→128 channels (restore dimensionality)
4. **Skip Connection:** x_out = x_in + f(x_in)

### 2.4 Training Configuration

**Hyperparameters:**
- **Optimizer:** Adam
- **Learning Rate:** 1×10⁻⁴ (fixed, no scheduling)
- **Batch Size:** 32
- **Epochs:** 50
- **Loss Function:** Cross-Entropy Loss (averaged over RGB channels)
- **Weight Initialization:** Default PyTorch initialization
- **Device:** CPU (no GPU available in this setup)

### 2.5 Evaluation Metrics

#### 2.5.1 Mean Squared Error (MSE)

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

- Measures average squared difference between predicted and ground truth pixels
- Range: [0, ∞), lower is better
- **Our Result:** 0.0217 (low error indicates good reconstruction)

#### 2.5.2 Peak Signal-to-Noise Ratio (PSNR)

$$PSNR = 20 \log_{10}(MAX_I) - 10 \log_{10}(MSE)$$

- Standard metric for image quality assessment
- Measured in decibels (dB), higher is better
- **Our Result:** 16.82 dB (moderate quality, acceptable for image completion)

#### 2.5.3 Pixel Accuracy

$$Accuracy = \frac{\text{Number of Exactly Matching Pixels}}{\text{Total Number of Pixels}}$$

- Percentage of pixels that exactly match ground truth after quantization
- Very strict metric (requires exact RGB match)
- **Our Result:** 7.17% (normal for image generation tasks)

---

## 3. Results

### 3.1 Training Performance

The model successfully learned to:
- Predict plausible pixel distributions for occluded regions
- Maintain spatial coherence across the image
- Respect boundary conditions between occluded and visible regions

### 3.2 Validation Set Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Average MSE | 0.021737 | Low reconstruction error |
| Average PSNR | 16.82 dB | Moderate quality |
| Pixel Accuracy | 7.17% | Normal for generation tasks |

### 3.3 Qualitative Analysis

Based on the Streamlit viewer analysis:

1. **Texture Coherence:** The model successfully generates plausible textures for bedding, walls, and furniture.
2. **Color Consistency:** Overall color palette maintained in reconstructions.
3. **Structural Integrity:** The model respects overall room structure (walls, beds, furniture).
4. **Common Artifacts:** Blurriness in some reconstructed regions and occasional color discontinuities at boundaries.

### 3.4 Performance by Image Characteristics

**Hypothesized Performance Factors:**
1. **Occlusion Size:** Smaller occlusions → Better PSNR
2. **Texture Complexity:** Simpler textures → Higher accuracy
3. **Lighting:** Uniform lighting → More consistent colors
4. **Pattern Repetition:** Repetitive patterns (bedding) → Easier to complete

---

## 4. Discussion

### 4.1 Model Coherence Analysis

The PixelCNN model demonstrates strong spatial coherence due to its autoregressive nature. Each pixel is predicted conditioned on all previously generated pixels, ensuring local consistency. RGB channels are processed jointly, but independently predicted. Cross-channel correlations are captured through shared features in residual blocks.

### 4.2 Comparison: Pixel Accuracy vs. Visual Quality

Our model achieved only 7.17% pixel accuracy, yet visual inspection reveals plausible reconstructions. This apparent contradiction highlights that pixel accuracy is a very strict metric, and human vision is insensitive to small color differences. Perceptual quality is often better reflected by metrics like PSNR, or more advanced ones like SSIM, than by pixel accuracy alone.

### 4.3 Challenges Encountered

- **Computational Challenges:** CPU-only training took several hours for 50 epochs due to the computational expense of masked convolutions.
- **Architectural Challenges:** Inference requires slow, sequential pixel generation, which is impractical for large images. We used argmax sampling for faster, deterministic evaluation.
- **Dataset Challenges:** The dataset's focus on bedroom images limits the model's ability to generalize.
- **Evaluation Challenges:** The provided test set lacked ground truth images, necessitating the creation of a validation split from the training data for quantitative evaluation.

### 4.4 Comparison with Alternative Approaches

| Approach | Advantages | Disadvantages |
|----------|-----------|-----------------|
| PixelCNN (Our Approach) | Explicit likelihood, coherent generation, theoretically principled | Slow inference, limited receptive field |
| U-Net / CNN Encoder-Decoder | Fast inference, large receptive field | No explicit likelihood, may produce blurry outputs |
| GANs | Sharp, realistic outputs | Training instability, mode collapse |
| Diffusion Models | State-of-the-art quality | Very slow inference, complex training |
| Transformers | Global context, excellent performance | Extremely high computational cost |

---

## 5. Conclusion

### 5.1 Summary of Findings

This assignment successfully implemented and evaluated a PixelCNN-based image completion system. Key findings include:

1. **Model Performance:** Achieved 16.82 dB PSNR on the validation set (moderate quality) with an MSE of 0.0217.
2. **Qualitative Success:** Generated plausible and coherent completions for occluded regions, maintaining spatial consistency and appropriate textures.
3. **Technical Implementation:** Masked convolutions correctly enforce the autoregressive property, and residual blocks enable a deeper architecture.

### 5.2 Strengths of the Approach

- **Theoretical Foundation:** Based on sound probabilistic principles.
- **Stable Training:** Converged reliably without adversarial dynamics.
- **Interpretable Outputs:** Provides explicit probability distributions.
- **Coherent Generation:** Autoregressive property ensures consistency.

### 5.3 Limitations

- **Slow Inference:** Autoregressive sampling is extremely time-consuming.
- **Limited Receptive Field:** Masked convolutions restrict context.
- **Moderate PSNR:** Results are below state-of-the-art methods.
- **Domain-Specific:** Trained only on bedroom images.

### 5.4 Future Improvements

- **Architectural:** Implement Gated PixelCNN or add attention mechanisms.
- **Training:** Train for more epochs, use learning rate scheduling, and add data augmentation.
- **Evaluation:** Compute perceptual metrics like SSIM and LPIPS.
- **Application:** Support arbitrary mask shapes and higher resolutions.

### 5.5 Final Remarks

The PixelCNN model successfully learned to complete occluded bedroom images, achieving moderate reconstruction quality (16.82 dB PSNR) with coherent and plausible outputs. While the approach has limitations, it provides a solid foundation for understanding autoregressive generative models. The low pixel accuracy (7.17%) highlights the inherent ambiguity in image completion tasks and the limitations of strict pixel-wise metrics. Visual inspection and perceptual metrics (PSNR) provide more meaningful assessments of generation quality. The assignment successfully met its objectives by implementing a working PixelCNN, training it on real data, and evaluating its performance quantitatively and qualitatively.

---

## References

1. van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel Recurrent Neural Networks. International Conference on Machine Learning (ICML).

2. van den Oord, A., Kalchbrenner, N., Vinyals, O., Espeholt, L., Graves, A., & Kavukcuoglu, K. (2016). Conditional Image Generation with PixelCNN Decoders. Neural Information Processing Systems (NeurIPS).

3. Salimans, T., Karpathy, A., Chen, X., & Kingma, D. P. (2017). PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications. International Conference on Learning Representations (ICLR).

4. Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., & Efros, A. A. (2016). Context Encoders: Feature Learning by Inpainting. Computer Vision and Pattern Recognition (CVPR).

5. Yu, J., Lin, Z., Yang, J., Shen, X., Lu, X., & Huang, T. S. (2018). Generative Image Inpainting with Contextual Attention. Computer Vision and Pattern Recognition (CVPR).

---

## Appendices

### Appendix A: Hyperparameter Configuration

| Parameter | Value | Justification |
|-----------|-------|-----------------|
| Image Size | 64×64 | Balance between detail and computation |
| Batch Size | 32 | Fit in memory, stable gradients |
| Learning Rate | 1×10⁻⁴ | Standard for Adam optimizer |
| Epochs | 50 | Sufficient for convergence |
| Filters | 128 | Adequate capacity without overfitting |
| Residual Blocks | 5 | Balance depth and training time |
| Output Bins | 256 | Standard 8-bit color depth |

### Appendix B: File Structure

```
pixelcnn-image-inpainting/
├── README.md
├── REPORT.md
├── model.py
├── train.py
├── dataset.py
├── evaluate.py
├── viewer.py
├── app.py
├── create_validation_split.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── train/
│   │   ├── original_images/
│   │   └── occluded_images/
│   ├── val/
│   │   ├── original_images/
│   │   └── occluded_images/
│   └── occluded_test/
├── results/
│   ├── recon_*.jpg
│   └── metrics_val.txt
└── saved_models/
    └── pixelcnn_bedroom.pth
```

### Appendix C: Code Snippets

**Masked Convolution Implementation:**

```python
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        
        # Apply mask based on type A or B
        self.mask[:, :, kH//2, kW//2 + (mask_type=='B'):] = 0
        self.mask[:, :, kH//2 + 1:] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
```

**Loss Computation:**

```python
# Predict distribution for each pixel
output = model(occluded_imgs)  # (B, 256, 3, H, W)

# Ground truth as class labels
target = (original_imgs * 255).long()  # (B, 3, H, W)

# Compute loss per channel
loss_r = criterion(output[:,:,0,:,:], target[:,0,:,:])
loss_g = criterion(output[:,:,1,:,:], target[:,1,:,:])
loss_b = criterion(output[:,:,2,:,:], target[:,2,:,:])

# Average across channels
loss = (loss_r + loss_g + loss_b) / 3.
```

---

**End of Report**
