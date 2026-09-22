import torch
from src.model import ContrastiveModel

def test_forward_pass():
    # 1. Instantiate the model
    # For this project, we'll use a hidden representation size of 128
    # and a projection output size of 64.
    model = ContrastiveModel(hidden_dim=128, out_dim=64, use_projection_head=True)
    
    # 2. Create some dummy data (simulating a batch of 4 MNIST images)
    # Shape: (Batch Size, Channels, Height, Width)
    batch_size = 4
    dummy_x = torch.randn(batch_size, 1, 28, 28)
    
    print(f"Input shape (x): {dummy_x.shape}")
    
    # 3. Run forward pass
    h, z = model(dummy_x)
    
    # 4. Verify shapes
    print(f"Encoder representation shape (h): {h.shape}")
    print(f"Projected & normalized output shape (z): {z.shape}")
    
    # 5. Verify L2 normalization
    # The L2 norm of each vector in the batch should be exactly 1.0 (or very close due to float precision)
    z_norms = torch.norm(z, p=2, dim=1)
    print(f"L2 norms of the projected outputs (z): {z_norms.detach().numpy()}")
    
    # Assertions to ensure it works properly
    assert h.shape == (batch_size, 128), "Encoder output shape is incorrect!"
    assert z.shape == (batch_size, 64), "Projection output shape is incorrect!"
    assert torch.allclose(z_norms, torch.ones_like(z_norms)), "L2 Normalization failed!"
    
    print("\nForward-pass sanity check passed successfully!")

if __name__ == "__main__":
    test_forward_pass()
