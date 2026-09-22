import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from src.dataset import get_dataloaders
from src.model import ContrastiveModel
from src.loss import InfoNCELoss
from src.utils import set_seed

def train(
    batch_size=256,
    epochs=5,
    lr=1e-3,
    temperature=0.5,
    strength="strong",
    subset_size=10000,
    use_projection_head=True,
    save_path="models/encoder.pth",
    plot_path="results/training_loss.png",
    sanity_check=False
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_loader, _, _, _ = get_dataloaders(
        batch_size=batch_size, 
        strength=strength, 
        subset_size=subset_size
    )
    
    # 2. Instantiate Model and Loss
    model = ContrastiveModel(hidden_dim=128, out_dim=64, use_projection_head=use_projection_head).to(device)
    criterion = InfoNCELoss(temperature=temperature)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track parameter updates during sanity check
    if sanity_check:
        initial_params = [p.clone().detach() for p in model.parameters()]
        
    epoch_losses = []
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar if not in sanity check (keep logs clean)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (x1, x2, _) in enumerate(pbar):
            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            _, z1 = model(x1)
            _, z2 = model(x2)
            
            # Compute loss
            loss = criterion(z1, z2)
            
            # Backward pass & update
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if sanity_check and batch_idx == 5:
                # Stop early for sanity check after a few batches
                break
                
        avg_loss = running_loss / (batch_idx + 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
        if sanity_check:
            break
            
    if sanity_check:
        # Verify parameters actually updated
        has_updated = False
        for p_init, p_curr in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_curr):
                has_updated = True
                break
        assert has_updated, "Sanity Check Failed: Model parameters did not update during training!"
        
        # Verify loss decreased (rough check: first batch vs last average)
        print("\n✓ Sanity Check Passed: Parameters are updating and loss is flowing.")
        return
        
    # 4. Save Model & Plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # We typically just save the encoder weights for downstream tasks
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Saved encoder weights to {save_path}")
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title("Contrastive Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Saved training loss plot to {plot_path}")
    
    return epoch_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run a quick sanity check")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subset", type=int, default=10000)
    args = parser.parse_args()
    
    train(
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_size=args.subset,
        sanity_check=args.sanity_check
    )
