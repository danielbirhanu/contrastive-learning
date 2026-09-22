import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) Loss for contrastive learning.
    Computes loss using in-batch negatives.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        """
        Args:
            z1, z2: L2-normalized embeddings from the two augmented views.
                    Shape: (BatchSize, EmbedDim)
        """
        batch_size = z1.size(0)
        
        # 1. Concatenate all embeddings in the batch to form a single tensor of size (2N, D)
        # Layout: [z1_0, z1_1, ..., z1_N, z2_0, z2_1, ..., z2_N]
        embeddings = torch.cat((z1, z2), dim=0)
        
        # 2. Compute the cosine similarity matrix between all pairs
        # Since embeddings are L2 normalized, cosine similarity is just the dot product
        # Shape: (2N, 2N)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # 3. Apply Temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # 4. Mask out self-similarity (the diagonal)
        # We don't want the model to compare an image view against itself.
        # We set the diagonal to a very large negative number so e^(val) -> 0.
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # 5. Construct labels for the positive pairs
        # For z1[i], the positive pair is z2[i], which is at index i + batch_size
        # For z2[i] (which is at index i + batch_size), the positive pair is z1[i], which is at index i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=similarity_matrix.device),
            torch.arange(0, batch_size, device=similarity_matrix.device)
        ])
        
        # 6. Compute CrossEntropyLoss
        # CrossEntropy computes -log( e^(sim(pos)) / sum(e^(sim(all))) )
        loss = self.criterion(similarity_matrix, labels)
        
        return loss
