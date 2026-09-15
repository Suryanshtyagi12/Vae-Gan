<div align="center">

# 🎨 Generative AI Studio: VAE, Vanilla GAN, DCGAN & Conditional GAN
### *A Comparative Study, Implementation from Scratch & Interactive Web Studio for Deep Generative Models*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Tesla%20T4-76B900.svg?logo=nvidia&logoColor=white)](https://cloud.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Explore Features](#-interactive-streamlit-studio) • [Architecture Deep-Dive](#-model-architectures--mathematical-formulations) • [Quickstart](#-quickstart--installation) • [Theory & Viva Q&A](#-theoretical-insights--viva-qa)

---

</div>

## 📌 Overview

This repository is an end-to-end research and deployment suite exploring the foundational pillars of **Deep Generative Modeling**. It implements four classical and modern deep generative architectures from scratch using **PyTorch**, benchmarks them on **MNIST Handwritten Digits** and **Fashion-MNIST Apparel**, and packages them into an interactive **Streamlit Studio**.

The project investigates the theoretical and empirical differences between **probabilistic density estimators** (Variational Autoencoders) and **game-theoretic adversarial networks** (GAN, DCGAN, Conditional GAN).

```
                        [Raw Image Data (28x28 grayscale)]
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
    1. VAE Architecture       2. Vanilla GAN / DCGAN     3. Conditional GAN (cGAN)
    • Optimizes ELBO          • Minimax Zero-Sum Game    • Label Embeddings [z + y]
    • Latent Reparameterization • Adversarial Dynamics   • User-Targeted Generation
            │                         │                         │
            ▼                         ▼                         ▼
   Smooth / Blurry Digits      Sharp Contours & Shapes      Class-Directed Digits
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                        [Pre-trained Weights (.pth)]
                                      ▼
                 [Interactive Streamlit Dashboard (app.py)]
```

---

## 🌟 Key Highlights

- **Four Models Implemented From Scratch**:
  1. **Variational Autoencoder (VAE)**: Continuous latent probabilistic manifold, ELBO loss, reparameterization trick.
  2. **Vanilla GAN**: Fully connected Multi-Layer Perceptron (MLP) minimax competition.
  3. **Deep Convolutional GAN (DCGAN)**: Transposed 2D convolutions, batch normalization, and spatial feature maps for Fashion-MNIST.
  4. **Conditional GAN (cGAN)**: Supervised generative synthesis with class-label embedding injection.
- **Pre-trained Weights Included**: Ready-to-use checkpoints (`.pth`) for instant offline or cloud inference without requiring hours of GPU retraining.
- **Interactive Streamlit Web Studio**: Complete 5-tab user interface for latent space navigation, seed exploration, interpolation morphing, and head-to-head empirical comparisons.
- **Mathematical Rigor**: Full LaTeX derivations for the ELBO objective, KL-divergence, Reparameterization Trick, and Minimax adversarial loss.

---

## 🎛️ Interactive Streamlit Studio (`app.py`)

The project includes an interactive dashboard with five specialized modules:

| Tab | Feature | Description |
| :--- | :--- | :--- |
| **Tab 0** | **🎯 User-Controlled (cGAN)** | Select any digit ($0$ through $9$) from a dropdown to generate tailored handwriting variations, or render the entire $0 \to 9$ sequence in real time. |
| **Tab 1** | **🖼️ Unconditional Generator** | Synthesize random digits/clothes with full control over **Random Seed**, **Noise Temperature** ($0.2 \to 2.0$), batch size ($4 \to 16$), and color maps (`gray`, `viridis`, `inferno`, `magma`). |
| **Tab 2** | **✨ Latent Space Morphing** | Perform real-time linear interpolation ($z_{\alpha} = (1 - \alpha)z_A + \alpha z_B$) between two latent seeds with an interactive slider and transition strip. |
| **Tab 3** | **⚖️ 1-to-1 Digits Comparison** | Side-by-side empirical juxtaposition of VAE digits vs. GAN digits under identical training datasets to analyze structural blur vs. adversarial sharpness. |
| **Tab 4** | **📚 Theoretical Deep-Dive** | Interactive formulas, architectural flowcharts, and explanations of generative mechanics. |

### Running the Studio Locally:
```bash
streamlit run app.py
```

---

## 🔬 Model Architectures & Mathematical Formulations

### 1. Variational Autoencoder (VAE)
A directed probabilistic graphical model that optimizes the **Evidence Lower Bound (ELBO)**. It maps input $x$ to parameters of a distribution $q_\phi(z|x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ in latent space $\mathbb{R}^{20}$, then samples $z$ to reconstruct $p_\theta(x|z)$.

* **Objective Function:**
  $$\mathcal{L}_{\text{ELBO}}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \,||\, p(z))$$
* **Closed-form Gaussian KL-Divergence:**
  $$D_{\text{KL}}(q_\phi(z|x) \,||\, \mathcal{N}(0, I)) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
* **Reparameterization Trick:** To enable backpropagation through stochastic sampling:
  $$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

### 2. Vanilla GAN (Generative Adversarial Network)
A zero-sum game between a Generator $G_\theta$ and a Discriminator $D_\phi$:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

In practice, to prevent early-stage vanishing gradients when $D$ dominates, $G$ is trained to maximize $\log D(G(z))$ (non-saturating heuristic).

---

### 3. Deep Convolutional GAN (DCGAN)
Replaces fully connected layers with spatial convolutions following Radford et al. architectural guidelines:
- Generator utilizes **Transposed Convolutions** (`ConvTranspose2d`) for upsampling from $\mathbb{R}^{100 \times 1 \times 1}$ to $\mathbb{R}^{1 \times 28 \times 28}$.
- Batch Normalization (`BatchNorm2d`) applied across layers to stabilize gradient flow.
- Activation functions: **LeakyReLU** ($\alpha=0.2$) in intermediate layers, **Tanh** on the final output layer.
- Trained on **Fashion-MNIST** to capture complex item geometries (sneakers, coats, bags).

---

### 4. Conditional GAN (cGAN)
Extends the unconditional adversarial formulation by providing class condition $y$ to both networks:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{z, y}[\log(1 - D(G(z, y), y))]$$
- **Implementation**: The class integer $y \in \{0, \dots, 9\}$ is passed through `nn.Embedding(10, 10)` and concatenated directly with the noise vector $z \in \mathbb{R}^{100}$:
  $$z_{\text{input}} = [z \,\|\, \operatorname{Embedding}(y)] \in \mathbb{R}^{110}$$

---

## 📊 Comprehensive Comparative Matrix

| Specification | Model 1: VAE | Model 2: Vanilla GAN | Model 3: DCGAN | Model 4: Conditional GAN (cGAN) |
| :--- | :--- | :--- | :--- | :--- |
| **Domain** | MNIST Digits (0–9) | MNIST Digits (0–9) | Fashion-MNIST (Apparel) | MNIST Digits (0–9) |
| **Model Type** | Probabilistic Autoencoder | Adversarial Minimax | Convolutional Adversarial | Supervised Adversarial |
| **Latent Vector ($z$)** | 20 dimensions | 100 dimensions | 100 dimensions | 100 dims + 10 class dims |
| **Core Layers** | Fully Connected (Linear) | Fully Connected (Linear) | `ConvTranspose2d` & `Conv2d` | Linear + `nn.Embedding` |
| **Hidden Activations** | ReLU | LeakyReLU ($\alpha=0.2$) | LeakyReLU + BatchNorm2d | LeakyReLU + BatchNorm1d |
| **Output Activation** | Sigmoid ($[0, 1]$) | Tanh ($[-1, 1]$) | Tanh ($[-1, 1]$) | Tanh ($[-1, 1]$) |
| **Loss Objective** | ELBO (BCE + KL Div) | Binary Cross-Entropy | Binary Cross-Entropy | Binary Cross-Entropy + Labels |
| **Visual Character** | Smooth, continuous, blurry | Sharp, slight pixel noise | Crisp clothing silhouettes | High fidelity, user-controlled |

---

## 📈 Training Setup & Convergence Dynamics

All models were trained on Google Colab leveraging an **NVIDIA Tesla T4 GPU (16GB VRAM)**:

| Model | Batch Size | Learning Rate | Optimizer | Epochs | Final Metrics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VAE** | 128 | $1 \times 10^{-3}$ | Adam ($\beta_1=0.9, \beta_2=0.999$) | 10 | Loss: $106.06$ (Monotonic descent) |
| **Vanilla GAN** | 64 | $2 \times 10^{-4}$ | Adam ($\beta_1=0.5, \beta_2=0.999$) | 15 | $D_{\text{loss}}: 1.132$, $G_{\text{loss}}: 0.888$ |
| **DCGAN** | 64 | $2 \times 10^{-4}$ | Adam ($\beta_1=0.5, \beta_2=0.999$) | 20 | $D_{\text{loss}}: 1.137$, $G_{\text{loss}}: 1.522$ |
| **cGAN** | 128 | $2 \times 10^{-4}$ | Adam ($\beta_1=0.5, \beta_2=0.999$) | 15 | Balanced equilibrium |

> 💡 **Convergence Insight:** In GANs, loss metrics fluctuate dynamically around a Nash equilibrium rather than decaying to zero. Evaluating visual fidelity alongside discriminator-generator balance is essential.

---

## 📁 Repository Structure

```plaintext
├── app.py                      # Interactive Streamlit Web Studio (5 tabs, live inference)
├── gan.py                      # Standalone CLI training script for Vanilla GAN
├── PROJECT_REPORT.md           # Comprehensive viva guide, technical notes, and pitch
├── requirements.txt            # Python dependencies (PyTorch, Streamlit, Matplotlib, etc.)
├── .gitignore                  # Git tracking rules
│
├── cgan_mnist_weights.pth      # Pre-trained weights: Conditional GAN (MNIST)
├── gan_mnist_weights.pth       # Pre-trained weights: Vanilla GAN (MNIST)
├── gan_weights.pth             # Pre-trained weights: DCGAN (Fashion-MNIST)
├── vae_weights.pth             # Pre-trained weights: Variational Autoencoder (MNIST)
│
├── C2W2_VAE.ipynb              # Deep Dive Notebook: VAE mathematical derivation & training
├── C3W1_Assignment.ipynb       # Deep Dive Notebook: Vanilla GAN foundational assignment
└── C3W2A_Assignment.ipynb      # Deep Dive Notebook: Advanced GANs & DCGAN implementation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Suryanshtyagi12/Vae-Gan.git
cd Vae-Gan
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive Studio
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. (Optional) Train Vanilla GAN via CLI
```bash
python gan.py --n_epochs 100 --batch_size 64 --lr 0.0002 --latent_dim 100
```

---

## 🧠 Theoretical Insights & Viva Q&A

<details>
<summary><b>1. Why are images produced by VAEs often blurry, while GAN outputs are sharp?</b></summary>
<br>
VAEs optimize pixel-wise distance metrics (like Binary Cross-Entropy or Mean Squared Error). When the model is uncertain about the exact contour or edge placement, the mathematically optimal value that minimizes expected error is the <i>average</i> of all plausible configurations—leading to smooth, blurry outputs. 

In contrast, GANs do not use pixel-wise comparison. The Discriminator functions as an adaptive adversary that penalizes blurriness as an artificial signature of fake images. This forces the Generator to output crisp, high-frequency contours.
</details>

<details>
<summary><b>2. Why is the Reparameterization Trick necessary in VAEs?</b></summary>
<br>
In standard stochastic sampling, $z \sim \mathcal{N}(\mu, \sigma^2)$ is a non-differentiable operation; analytical gradients cannot backpropagate through random stochastic nodes. The reparameterization trick decouples the randomness by writing:
$$z = \mu + \sigma \odot \epsilon \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$
Here, $\epsilon$ acts as an independent stochastic noise input, allowing backpropagation to compute deterministic gradients $\nabla_\mu$ and $\nabla_\sigma$ directly.
</details>

<details>
<summary><b>3. Why is standard Early Stopping unsuitable for GAN training?</b></summary>
<br>
In supervised learning, validation loss decreases monotonically until overfitting begins. In GAN training, the loss represents a non-cooperative game between two dynamic networks. Generator loss can fluctuate or rise while visual quality improves. A generator loss near zero often indicates <i>mode collapse</i> rather than optimal convergence.
</details>

<details>
<summary><b>4. How does Conditional GAN (cGAN) steer class generation?</b></summary>
<br>
cGAN conditions both the Generator and Discriminator on label vectors $y$. For the generator, the integer label is passed through an embedding layer and concatenated directly with the noise vector $z$. The discriminator receives both the candidate image and the corresponding class label, learning to penalize samples that look realistic but do not match the requested class.
</details>

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author & Acknowledgements

Developed by **[Suryansh Tyagi](https://github.com/Suryanshtyagi12)**.

- Built using [PyTorch](https://pytorch.org/) & [Streamlit](https://streamlit.io/).
- Architectural references: Radford et al. (*Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*), Kingma & Welling (*Auto-Encoding Variational Bayes*), and Goodfellow et al. (*Generative Adversarial Nets*).
