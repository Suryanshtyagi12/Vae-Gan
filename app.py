import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Generative Studio: VAE, GAN & cGAN",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 1. Model Architectures
# -------------------------------------------------------------

# Model 1: Conditional GAN (cGAN - User specifies the digit!)
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, image_size=784):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=-1)
        return self.net(x).view(-1, 1, 28, 28)


# Model 2: DCGAN (Convolutional Fashion-MNIST)
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        if len(z.shape) == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


# Model 3: MLP / Vanilla GAN
class MLPGenerator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=256, image_size=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        if len(z.shape) == 4:
            z = z.view(z.size(0), -1)
        out = self.net(z)
        return out.view(-1, 1, 28, 28)


# Model 4: VAE for MNIST Digits
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))


# -------------------------------------------------------------
# 2. Dynamic Model Loaders
# -------------------------------------------------------------
def load_smart_gan(filepath):
    if not os.path.exists(filepath):
        return None, False
    try:
        state = torch.load(filepath, map_location='cpu')
        first_key = list(state.keys())[0]
        weight_shape = state[first_key].shape
        if len(weight_shape) == 4:
            model = DCGANGenerator()
        else:
            model = MLPGenerator()
        model.load_state_dict(state)
        model.eval()
        return model, True
    except Exception as e:
        return None, False


def load_cgan(filepath):
    if not os.path.exists(filepath):
        return None, False
    try:
        model = ConditionalGenerator()
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
        model.eval()
        return model, True
    except Exception as e:
        return None, False


def load_vae(filepath):
    if not os.path.exists(filepath):
        return None, False
    try:
        model = VAE()
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
        model.eval()
        return model, True
    except:
        return None, False


def resolve_weight_path(filename):
    """Checks weights/ directory first, then root directory."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "weights", filename),
        os.path.join("weights", filename),
        os.path.join(os.path.dirname(__file__), filename),
        filename
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filename


# Load all 4 models
dcgan_model, dcgan_ready = load_smart_gan(resolve_weight_path('gan_weights.pth'))
mnist_gan_model, mnist_gan_ready = load_smart_gan(resolve_weight_path('gan_mnist_weights.pth'))
vae_model, vae_ready = load_vae(resolve_weight_path('vae_weights.pth'))
cgan_model, cgan_ready = load_cgan(resolve_weight_path('cgan_mnist_weights.pth'))

# Fallbacks if weights not yet saved
if dcgan_model is None:
    dcgan_model = DCGANGenerator(); dcgan_model.eval()
if mnist_gan_model is None:
    mnist_gan_model = DCGANGenerator(); mnist_gan_model.eval()
if vae_model is None:
    vae_model = VAE(); vae_model.eval()
if cgan_model is None:
    cgan_model = ConditionalGenerator(); cgan_model.eval()

# -------------------------------------------------------------
# 3. Sidebar Controls
# -------------------------------------------------------------
st.sidebar.title("🎛️ Studio Controls")

sidebar_model = st.sidebar.selectbox(
    "Unconditional Model Selection:",
    [
        "1. GAN (MNIST Digits - Sharp)",
        "2. VAE (MNIST Digits - Smooth)",
        "3. DCGAN (Fashion-MNIST - Clothes)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Latent Space Settings")
user_seed = st.sidebar.slider("Random Seed (Coordinate):", min_value=0, max_value=9999, value=42, step=1)
temp = st.sidebar.slider("Noise Variance (Temperature):", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
num_samples = st.sidebar.select_slider("Number of Images:", options=[4, 8, 12, 16], value=8)
cmap = st.sidebar.selectbox("Color Palette:", ["gray", "viridis", "inferno", "magma"])

# -------------------------------------------------------------
# 4. Main Header & Tabs
# -------------------------------------------------------------
st.title("🎨 Generative AI Studio: VAE, GAN & cGAN")
st.caption("Complete interactive demonstration of generative models: Unconditional, Conditional, and Variational.")

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 User-Controlled (cGAN)",
    "🖼️ Unconditional Generator",
    "✨ Latent Space Morphing",
    "⚖️ 1-to-1 Digits Comparison",
    "📚 Theoretical Deep-Dive"
])

# -------------------------------------------------------------
# TAB 0: User-Controlled Generation (Conditional GAN)
# -------------------------------------------------------------
with tab0:
    st.subheader("🎯 Conditional Generation: You Pick the Digit!")
    st.write(
        "Unlike unconditional models that output random items, a **Conditional GAN (cGAN)** takes "
        "**User Input (Class Label)** and forces the generator to draw that exact category!"
    )

    if cgan_ready:
        st.success("✅ Active: `cgan_mnist_weights.pth` loaded successfully!")
    else:
        st.warning("⏳ Waiting for `cgan_mnist_weights.pth`. (Once it downloads from Colab, move it to `d:\\Vae_Gan` and refresh this page).")

    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        chosen_digit = st.selectbox("Choose a Digit to Generate:", list(range(10)), index=7)
        cgan_samples = st.slider("How many variations?", 1, 8, 4)
        cgan_seed = st.number_input("Variation Seed:", value=123, min_value=0, max_value=9999)

    torch.manual_seed(cgan_seed)
    z_cgan = torch.randn(cgan_samples, 100) * temp
    labels_tensor = torch.full((cgan_samples,), chosen_digit, dtype=torch.long)

    with torch.no_grad():
        cgan_imgs = (cgan_model(z_cgan, labels_tensor).squeeze(1).numpy() + 1.0) / 2.0

    with ctrl_col2:
        st.markdown(f"#### Generated Variations of Digit **{chosen_digit}**:")
        cols = st.columns(cgan_samples)
        for i in range(cgan_samples):
            with cols[i]:
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                ax.imshow(cgan_imgs[i], cmap=cmap)
                ax.axis("off")
                ax.set_title(f"Variant #{i+1}", fontsize=9)
                st.pyplot(fig)
                plt.close(fig)

    st.markdown("---")
    st.markdown("#### 🔢 Sequence Showcase: Generating Digits 0 through 9 on Demand")
    
    torch.manual_seed(user_seed)
    seq_z = torch.randn(10, 100)
    seq_labels = torch.arange(0, 10, dtype=torch.long)
    with torch.no_grad():
        seq_imgs = (cgan_model(seq_z, seq_labels).squeeze(1).numpy() + 1.0) / 2.0

    seq_cols = st.columns(10)
    for digit in range(10):
        with seq_cols[digit]:
            fig, ax = plt.subplots(figsize=(1.8, 1.8))
            ax.imshow(seq_imgs[digit], cmap=cmap)
            ax.axis("off")
            ax.set_title(f"Digit {digit}", fontsize=10, fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

    st.info("💡 **Key Insight:** The Generator takes `Input = [Latent Noise z, Class Label y]`. By conditioning on $y$, we guide the generation process precisely to the desired manifold.")

# -------------------------------------------------------------
# TAB 1: Live Unconditional Generation
# -------------------------------------------------------------
with tab1:
    st.subheader(f"Active Unconditional Model: **{sidebar_model}**")

    torch.manual_seed(user_seed)
    if "Fashion" in sidebar_model:
        z = torch.randn(num_samples, 100, 1, 1) * temp
        with torch.no_grad():
            gen_imgs = (dcgan_model(z).squeeze(1).numpy() + 1.0) / 2.0
    elif "GAN (MNIST" in sidebar_model:
        z = torch.randn(num_samples, 100, 1, 1) * temp
        with torch.no_grad():
            gen_imgs = (mnist_gan_model(z).squeeze(1).numpy() + 1.0) / 2.0
    else:
        z = torch.randn(num_samples, 20) * temp
        with torch.no_grad():
            gen_imgs = vae_model.decode(z).view(-1, 28, 28).numpy()

    cols_per_row = 4
    rows = (num_samples + cols_per_row - 1) // cols_per_row
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx < num_samples:
                with cols[c]:
                    fig, ax = plt.subplots(figsize=(2.5, 2.5))
                    ax.imshow(gen_imgs[idx], cmap=cmap)
                    ax.axis("off")
                    ax.set_title(f"Sample #{idx+1}", fontsize=9)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

# -------------------------------------------------------------
# TAB 2: Latent Space Morphing
# -------------------------------------------------------------
with tab2:
    st.subheader("Smooth Latent Walk (Interpolation)")
    st.write("Linearly traversing coordinates in latent space: $z = (1 - \\alpha) z_A + \\alpha z_B$.")

    col_a, col_b, col_alpha = st.columns([1, 1, 2])
    with col_a:
        seed_a = st.number_input("Start Seed (A):", value=10, min_value=0, max_value=9999)
    with col_b:
        seed_b = st.number_input("End Seed (B):", value=88, min_value=0, max_value=9999)
    with col_alpha:
        alpha = st.slider("Morph Position (α):", min_value=0.0, max_value=1.0, value=0.5, step=0.02)

    active_net = dcgan_model if "Fashion" in sidebar_model else mnist_gan_model
    if "Fashion" in sidebar_model or "GAN (MNIST" in sidebar_model:
        torch.manual_seed(seed_a); zA = torch.randn(1, 100, 1, 1)
        torch.manual_seed(seed_b); zB = torch.randn(1, 100, 1, 1)
        zM = (1.0 - alpha) * zA + alpha * zB
        with torch.no_grad():
            imgA = (active_net(zA).squeeze().numpy() + 1.0) / 2.0
            imgB = (active_net(zB).squeeze().numpy() + 1.0) / 2.0
            imgM = (active_net(zM).squeeze().numpy() + 1.0) / 2.0
    else:
        torch.manual_seed(seed_a); zA = torch.randn(1, 20)
        torch.manual_seed(seed_b); zB = torch.randn(1, 20)
        zM = (1.0 - alpha) * zA + alpha * zB
        with torch.no_grad():
            imgA = vae_model.decode(zA).view(28, 28).numpy()
            imgB = vae_model.decode(zB).view(28, 28).numpy()
            imgM = vae_model.decode(zM).view(28, 28).numpy()

    col_view_a, col_view_morph, col_view_b = st.columns([1, 2, 1])
    with col_view_a:
        figA, axA = plt.subplots(figsize=(2.5, 2.5))
        axA.imshow(imgA, cmap=cmap); axA.axis("off"); axA.set_title("Point A (α=0.0)")
        st.pyplot(figA); plt.close(figA)
    with col_view_morph:
        figM, axM = plt.subplots(figsize=(4, 4))
        axM.imshow(imgM, cmap=cmap); axM.axis("off")
        axM.set_title(f"Morphed Output at α = {alpha:.2f}", fontsize=12, fontweight='bold')
        st.pyplot(figM); plt.close(figM)
    with col_view_b:
        figB, axB = plt.subplots(figsize=(2.5, 2.5))
        axB.imshow(imgB, cmap=cmap); axB.axis("off"); axB.set_title("Point B (α=1.0)")
        st.pyplot(figB); plt.close(figB)

    st.markdown("#### Full Trajectory (8-Step Strip)")
    strip_steps = 8
    alphas = np.linspace(0.0, 1.0, strip_steps)
    strip_cols = st.columns(strip_steps)
    for i, a in enumerate(alphas):
        zi = (1.0 - a) * zA + a * zB
        with torch.no_grad():
            if "Fashion" in sidebar_model or "GAN (MNIST" in sidebar_model:
                im = (active_net(zi).squeeze().numpy() + 1.0) / 2.0
            else:
                im = vae_model.decode(zi).view(28, 28).numpy()
        with strip_cols[i]:
            figS, axS = plt.subplots(figsize=(1.8, 1.8))
            axS.imshow(im, cmap=cmap); axS.axis("off"); axS.set_title(f"α={a:.2f}", fontsize=8)
            st.pyplot(figS); plt.close(figS)

# -------------------------------------------------------------
# TAB 3: Direct 1-to-1 Digits Comparison
# -------------------------------------------------------------
with tab3:
    st.subheader("Direct 1-to-1 Comparison on MNIST Digits")
    comp_col1, comp_col2 = st.columns(2)

    torch.manual_seed(user_seed)
    with comp_col1:
        st.markdown("### 🔹 VAE Digits (Smooth & Blurry)")
        z_v = torch.randn(4, 20)
        with torch.no_grad():
            imgs_v = vae_model.decode(z_v).view(-1, 28, 28).numpy()

        fig_v, axes_v = plt.subplots(2, 2, figsize=(3.8, 3.8))
        for i, ax in enumerate(axes_v.flatten()):
            ax.imshow(imgs_v[i], cmap='gray'); ax.axis('off')
        st.pyplot(fig_v); plt.close(fig_v)
        st.markdown("* **Cause:** Pixel-level BCE reconstruction loss averaging pixel intensities.")

    with comp_col2:
        st.markdown("### 🔸 GAN Digits (Sharp & Contrast)")
        z_g = torch.randn(4, 100, 1, 1)
        with torch.no_grad():
            imgs_g = (mnist_gan_model(z_g).squeeze(1).numpy() + 1.0) / 2.0

        fig_g, axes_g = plt.subplots(2, 2, figsize=(3.8, 3.8))
        for i, ax in enumerate(axes_g.flatten()):
            ax.imshow(imgs_g[i], cmap='gray'); ax.axis('off')
        st.pyplot(fig_g); plt.close(fig_g)
        st.markdown("* **Cause:** Minimax adversarial game rejecting blurry intermediate pixels.")

# -------------------------------------------------------------
# TAB 4: Theory
# -------------------------------------------------------------
with tab4:
    st.subheader("Key Mathematical Formulations")
    st.markdown("### 1. Conditional GAN (cGAN) Minimax Objective")
    st.latex(r"\min_G \max_D V(D, G) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{z, y}[\log (1 - D(G(z, y), y))]")
    
    st.markdown("### 2. Standard GAN Minimax Game")
    st.latex(r"\min_G \max_D V(D, G) = \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log (1 - D(G(z)))]")
    
    st.markdown("### 3. VAE ELBO Objective")
    st.latex(r"\log p(x) \ge \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))")
