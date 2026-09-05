import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

# -------------------------------------------------------------
# 1. Setup & Model Architecture
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmallCNN(nn.Module):
    def __init__(self, latent_dim=64):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(latent_dim, 10)

    def forward(self, x):
        feat = self.features(x)
        z = self.latent(feat)          # Penultimate representation
        out = self.classifier(z)
        return out, z

# -------------------------------------------------------------
# 2. Topological Metric: Persistent Entropy E(D)
# -------------------------------------------------------------
def compute_persistent_entropy(point_cloud, maxdim=1):
    """
    Computes Persistent Entropy E(D) for H0 and H1 barcode lifespans.
    point_cloud: (N, d) numpy array of latent activations.
    """
    diagrams = ripser(point_cloud, maxdim=maxdim)['dgms']
    entropies = []
    
    for dgm in diagrams:
        if len(dgm) == 0:
            entropies.append(0.0)
            continue
            
        finite_dgm = dgm[np.isfinite(dgm[:, 1])]
        if len(finite_dgm) == 0:
            entropies.append(0.0)
            continue
            
        lifespans = finite_dgm[:, 1] - finite_dgm[:, 0]
        total_life = np.sum(lifespans)
        
        if total_life == 0:
            entropies.append(0.0)
        else:
            p = lifespans / total_life
            entropy = -np.sum(p * np.log2(p + 1e-12))
            entropies.append(entropy)
            
    return entropies[0], entropies[1]  # E(H0), E(H1)

# -------------------------------------------------------------
# 3. Data Preparation (Subsampled to accelerate overfitting)
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Subsample 2000 train images to induce clean overfitting
indices = torch.arange(2000)
train_subset = torch.utils.data.Subset(trainset, indices)

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False)

# Fixed evaluation subset for consistent topological tracking (N=300)
fixed_val_subset, _ = next(iter(val_loader))
fixed_val_subset = fixed_val_subset[:300].to(device)

# -------------------------------------------------------------
# 4. Training Loop & Topological Telemetry
# -------------------------------------------------------------
model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 40
history = {
    'train_loss': [],
    'val_loss': [],
    'h0_entropy': [],
    'h1_entropy': []
}

print("Starting training with topological telemetry...")

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * images.size(0)
        
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    
    # Evaluate Validation Loss & Latent Topology
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * images.size(0)
            
        # Topological feature extraction on the fixed batch
        _, latent_z = model(fixed_val_subset)
        z_np = latent_z.cpu().numpy()
        
        # Normalize activations for stable geometric scale
        z_np = (z_np - z_np.mean(axis=0)) / (z_np.std(axis=0) + 1e-8)
        
        h0_ent, h1_ent = compute_persistent_entropy(z_np)
        
    avg_val_loss = total_val_loss / len(val_loader.dataset)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['h0_entropy'].append(h0_ent)
    history['h1_entropy'].append(h1_ent)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | H0 Ent: {h0_ent:.4f} | H1 Ent: {h1_ent:.4f}")

# -------------------------------------------------------------
# 5. Optimized Dual-Axis Visualization for Presentation Slides
# -------------------------------------------------------------
epochs = np.arange(1, num_epochs + 1)

# Compact, slide-friendly dimensions and high resolution
fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=300)

# Colors
color_val = '#e74c3c'   # Red for Lagging Metric
color_train = '#95a5a6' # Light Slate for Train Loss
color_topo = '#00acc1'  # Cyan for Lead Topological Metric

# Left Axis: Loss curves
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (Lagging Metric)', fontsize=12, fontweight='bold')
ax1.plot(epochs, history['train_loss'], color=color_train, linestyle=':', label='Train Loss', linewidth=1.8)
ax1.plot(epochs, history['val_loss'], color=color_val, label='Validation Loss', linewidth=2.5)
ax1.tick_params(axis='y', labelsize=10)
ax1.set_xlim(1, num_epochs)

# Right Axis: Topological Entropy (H0)
ax2 = ax1.twinx()
ax2.set_ylabel('Latent $H_0$ Persistent Entropy (Lead Metric)', color=color_topo, fontsize=12, fontweight='bold')
ax2.plot(epochs, history['h0_entropy'], color=color_topo, linestyle='-', linewidth=2.5, label='Persistent Entropy ($H_0$)')
ax2.tick_params(axis='y', labelcolor=color_topo, labelsize=10)

# Clean single-line header and legend
plt.title('Topological Signal Preceding Validation Loss Degradation', fontsize=13, pad=12, fontweight='bold')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, fontsize=9)

# Tight layout prevents text truncation
plt.tight_layout()

# Save both high-res artifacts
plt.savefig('topological_lead_indicator.png', bbox_inches='tight', dpi=300)
plt.savefig('topological_lead_indicator_compact.png', bbox_inches='tight', dpi=300)
print("\n[Done] Figures saved as 'topological_lead_indicator.png' and 'topological_lead_indicator_compact.png'")
plt.show()