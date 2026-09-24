import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) Loss with in-batch negatives."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        """
        Args:
            z1, z2: L2-normalized embeddings of shape (B, D).
        """
        batch_size = z1.size(0)
        
        # 1. Concatenate all embeddings of shape (2N, D)
        embeddings = torch.cat((z1, z2), dim=0)
        
        # 2. Compute similarity matrix (2N, 2N)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # 3. Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # 4. Mask self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # 5. Construct labels for positive pairs
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=similarity_matrix.device),
            torch.arange(0, batch_size, device=similarity_matrix.device)
        ])
        
        # 6. Compute CrossEntropyLoss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss
