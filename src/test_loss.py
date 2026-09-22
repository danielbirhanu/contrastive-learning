import torch
import torch.nn.functional as F
from src.loss import InfoNCELoss

def test_infonce_loss():
    batch_size = 4
    embed_dim = 64
    
    # 1. Create dummy embeddings (representing z1 and z2)
    # We require gradients to test backward pass
    z1_raw = torch.randn(batch_size, embed_dim, requires_grad=True)
    z2_raw = torch.randn(batch_size, embed_dim, requires_grad=True)
    
    # 2. L2 normalize (simulating what the model does)
    z1 = F.normalize(z1_raw, p=2, dim=1)
    z2 = F.normalize(z2_raw, p=2, dim=1)
    
    # 3. Instantiate Loss
    temperature = 0.5
    loss_fn = InfoNCELoss(temperature=temperature)
    
    # 4. Compute Loss
    loss = loss_fn(z1, z2)
    
    print(f"InfoNCE Loss value: {loss.item():.4f}")
    
    # 5. Tests
    # Is loss finite?
    assert torch.isfinite(loss), "Loss is not finite!"
    print("✓ Loss is finite.")
    
    # Check backward pass
    loss.backward()
    
    # Check if gradients flow back to raw inputs
    assert z1_raw.grad is not None and torch.any(z1_raw.grad != 0), "Gradients did not flow back to z1!"
    assert z2_raw.grad is not None and torch.any(z2_raw.grad != 0), "Gradients did not flow back to z2!"
    print("✓ Gradients flow correctly.")
    
    print("\nLoss sanity checks passed!")

if __name__ == "__main__":
    test_infonce_loss()
