<div align="center">

# 🎨 Generative AI Studio
### *Comparative Study & Interactive Web Studio for VAE, Vanilla GAN, DCGAN & Conditional GAN*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Live Features](#-studio-features--visual-results) • [Architecture Matrix](#-architecture-matrix) • [Quickstart](#-quickstart) • [Theory & Viva Q&A](#-key-theoretical-insights)

---

</div>

## 📌 Overview

This project implements and evaluates four foundational deep generative architectures built from scratch with **PyTorch** across **MNIST** and **Fashion-MNIST**:

1. **Variational Autoencoder (VAE)** — Probabilistic density modeling via ELBO optimization.
2. **Vanilla GAN** — Multi-layer perceptron adversarial minimax game.
3. **DCGAN** — Deep convolutional generative network using transposed convolutions and batch normalization.
4. **Conditional GAN (cGAN)** — Supervised generative synthesis driven by class-label embeddings.

All models are deployed inside an interactive **Streamlit Studio** (`app.py`) for live inference, latent interpolation, and head-to-head empirical comparison.

---

## 🖼️ Studio Features & Visual Results

### 1. User-Controlled Generation (Conditional GAN)
> Specify any target digit ($0 \to 9$) and synthesize targeted handwriting variations on demand.

<div align="center">
  <img src="assets/cgan_digits.png" alt="cGAN Digit Sequence" width="850"/>
</div>

---

### 2. Empirical Comparison: VAE vs. GAN
> Side-by-side analysis under identical dataset conditions: **probabilistic averaging** (blur) vs. **adversarial competition** (sharp contrast).

<div align="center">
  <img src="assets/vae_vs_gan.png" alt="VAE vs GAN Comparison" width="700"/>
</div>

---

### 3. Smooth Latent Space Morphing
> Linear interpolation across the latent manifold: $z(\alpha) = (1 - \alpha)z_A + \alpha z_B$.

<div align="center">
  <img src="assets/latent_morphing.png" alt="Latent Space Morphing" width="850"/>
</div>

---

### 4. DCGAN Synthesis on Fashion-MNIST
> Deep convolutional generation capturing complex clothing silhouettes (boots, sweaters, bags).

<div align="center">
  <img src="assets/dcgan_fashion.png" alt="DCGAN Fashion MNIST" width="850"/>
</div>

---

## 📊 Architecture Matrix

| Feature | VAE | Vanilla GAN | DCGAN | Conditional GAN (cGAN) |
| :--- | :--- | :--- | :--- | :--- |
| **Domain** | MNIST Digits | MNIST Digits | Fashion-MNIST | MNIST Digits |
| **Model Type** | Probabilistic Autoencoder | Adversarial Minimax | Deep Convolutional | Class-Conditioned GAN |
| **Latent Vector ($z$)** | $\mathbb{R}^{20}$ | $\mathbb{R}^{100}$ | $\mathbb{R}^{100}$ | $\mathbb{R}^{100} + \text{Emb}(y)_{10}$ |
| **Core Layers** | Linear (MLP) | Linear (MLP) | `ConvTranspose2d` + `Conv2d` | Linear + `nn.Embedding` |
| **Loss Function** | $\text{ELBO} = \text{BCE} + D_{\text{KL}}$ | Minimax Adversarial | Minimax Adversarial | Conditional Minimax |
| **Visual Quality** | Smooth & blurry | Sharp edges | Crisp apparel contours | User-guided crisp digits |

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/Suryanshtyagi12/Vae-Gan.git
cd Vae-Gan

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows (or source venv/bin/activate on Linux/macOS)

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Studio
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 3. (Optional) Train Vanilla GAN via CLI
```bash
python gan.py --n_epochs 100 --batch_size 64 --lr 0.0002
```

---

## 📂 Repository Structure

```plaintext
├── app.py                      # Interactive 5-tab Streamlit Studio (live UI)
├── gan.py                      # Standalone CLI training script for Vanilla GAN
├── PROJECT_REPORT.md           # Comprehensive presentation & viva notes
├── requirements.txt            # Minimal runtime dependencies
├── assets/                     # Visual demonstration outputs & comparisons
│   ├── cgan_digits.png
│   ├── vae_vs_gan.png
│   ├── latent_morphing.png
│   └── dcgan_fashion.png
├── cgan_mnist_weights.pth      # Pretrained weights: Conditional GAN
├── gan_mnist_weights.pth       # Pretrained weights: Vanilla GAN
├── gan_weights.pth             # Pretrained weights: DCGAN (Fashion-MNIST)
└── vae_weights.pth             # Pretrained weights: VAE
```

---

## 🧠 Key Theoretical Insights

* **Why are VAEs blurry while GANs are sharp?**  
  VAEs minimize pixel-wise reconstruction errors (BCE/MSE). In regions of uncertainty, the loss is minimized by predicting the *conditional mean* of pixel intensities, yielding a smooth average blur. GANs employ an adversarial discriminator that rejects blurred pixels as artificial, forcing the generator to synthesize crisp, high-frequency contours.

* **What is the Reparameterization Trick?**  
  Direct stochastic sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ cannot be backpropagated through. Expressing $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ moves the non-differentiable stochasticity to an auxiliary variable, allowing standard gradient descent on $\mu$ and $\sigma$.

* **How does cGAN steer output classes?**  
  Class label $y$ is mapped to a vector via `nn.Embedding(10, 10)` and concatenated with latent noise $z$. Both Generator and Discriminator are fed the label, constraining the generator to synthesize samples that match the conditioned category.

---

## 📜 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details. Developed by **[Suryansh Tyagi](https://github.com/Suryanshtyagi12)**.
