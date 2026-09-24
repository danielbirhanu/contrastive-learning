import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNEncoder(nn.Module):
    """Simple CNN encoder f(x) -> h for MNIST."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten() # (B, 64 * 7 * 7) = (B, 3136)
        )
        self.fc = nn.Linear(64 * 7 * 7, hidden_dim)
        
    def forward(self, x):
        features = self.conv_blocks(x)
        h = self.fc(features)
        return h

class ProjectionHead(nn.Module):
    """MLP projection head g(h) -> z."""
    def __init__(self, hidden_dim=128, out_dim=64):
        super().__init__()
        # 2-layer MLP with ReLU
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, h):
        return self.net(h)

class ContrastiveModel(nn.Module):
    """Combines encoder, projection head, and L2 normalization."""
    def __init__(self, hidden_dim=128, out_dim=64, use_projection_head=True):
        super().__init__()
        self.encoder = SimpleCNNEncoder(hidden_dim=hidden_dim)
        self.use_projection_head = use_projection_head
        
        if self.use_projection_head:
            self.head = ProjectionHead(hidden_dim=hidden_dim, out_dim=out_dim)
        else:
            self.head = nn.Identity()
            
    def forward(self, x):
        # 1. Get representations from the encoder
        h = self.encoder(x)
        
        # 2. Pass through projection head
        z = self.head(h)
        
        # 3. L2 normalize output
        z_normalized = F.normalize(z, p=2, dim=1)
        
        return h, z_normalized
