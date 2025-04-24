# 🎨 Image Colorization using Conditional GANs (Pix2Pix)

This project implements a **deep learning-based image colorization model** using a **Conditional Generative Adversarial Network (cGAN)**, inspired by the Pix2Pix architecture. It takes grayscale images as input and generates their realistic color versions.

---

## 📌 Features

- ✅ Pix2Pix-style conditional GAN
- ✅ Custom Generator and Discriminator networks
- ✅ Combined L1 Loss and GAN Loss for stable training
- ✅ PatchGAN discriminator for high-frequency details
- ✅ Input image resolution: **128×128**
- ✅ Full training and inference pipelines
- ✅ Gradient clipping and learning rate scheduling
- ✅ Model checkpoint saving and sample image generation per epoch

---

## 🧠 Model Architecture

### 🧬 Generator (U-Net Inspired)

- Encoder-decoder architecture
- Multiple convolutional → batch norm → ReLU layers
- Skip connections (optional for improved performance)
- Translates 1-channel grayscale input to 3-channel RGB color image

### 🔍 Discriminator (PatchGAN)

- Focuses on local image patches (70×70) instead of the entire image
- Concatenates the grayscale input and the corresponding color output
- Outputs a matrix of real/fake predictions for each patch

---

## 🗂️ Dataset

- Input: Grayscale images (1 channel)
- Output: Corresponding color images (3 channels)
- All images resized to **128×128**

---

## 🚀 Training Strategy

- Optimizers: Adam (lr=0.0002, betas=(0.5, 0.999))
- Losses:
  - **Adversarial Loss** (BCE)
  - **Reconstruction Loss** (L1)
- Learning Rate Scheduler: StepLR
- Gradient clipping for stable training
