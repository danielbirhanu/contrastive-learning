import os
import matplotlib.pyplot as plt
import torchvision
import numpy as np

# A small helper to un-normalize and display an MNIST image
def imshow(img, title=None, ax=None):
    # Un-normalize for MNIST
    mean = np.array([0.1307])
    std = np.array([0.3081])
    
    img = img.numpy().transpose((1, 2, 0)) # H, W, C
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    if ax is None:
        # cmap='gray' is important for 1-channel images
        plt.imshow(img.squeeze(), cmap='gray')
        if title:
            plt.title(title)
    else:
        ax.imshow(img.squeeze(), cmap='gray')
        if title:
            ax.set_title(title)
        ax.axis('off')

if __name__ == "__main__":
    from src.dataset import get_dataloaders
    from src.utils import set_seed
    
    set_seed(42)
    
    # Let's get weak and strong loaders to compare
    train_loader_weak, _, train_size, _ = get_dataloaders(batch_size=4, strength="weak", subset_size=10000)
    train_loader_strong, _, _, _ = get_dataloaders(batch_size=4, strength="strong", subset_size=10000)
    
    print(f"Dataset subset size loaded: {train_size}")
    
    # Fetch one batch
    x1_weak, x2_weak, _ = next(iter(train_loader_weak))
    x1_strong, x2_strong, _ = next(iter(train_loader_strong))
    
    # Plotting
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i in range(4):
        # Weak
        imshow(x1_weak[i], title="Weak View 1" if i == 0 else None, ax=axes[i, 0])
        imshow(x2_weak[i], title="Weak View 2" if i == 0 else None, ax=axes[i, 1])
        
        # Strong
        imshow(x1_strong[i], title="Strong View 1" if i == 0 else None, ax=axes[i, 2])
        imshow(x2_strong[i], title="Strong View 2" if i == 0 else None, ax=axes[i, 3])
        
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/aug_sanity_check.png')
    print("Saved augmentation sanity check plot to results/aug_sanity_check.png")
