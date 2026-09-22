import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class ContrastiveDataset(torch.utils.data.Dataset):
    """
    Wrapper around a torchvision dataset to return two augmented views of each image.
    These two views form our 'positive pair' for contrastive learning.
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        # We ignore the target for self-supervised training,
        # but keep it around for evaluation later.
        img, target = self.base_dataset[idx]
        
        # Generate two different augmented views of the same image
        x1 = self.transform(img)
        x2 = self.transform(img)
        
        return x1, x2, target

def get_transforms(strength="weak"):
    """
    Returns the data augmentation pipeline for MNIST based on the specified strength.
    MNIST is grayscale 28x28.
    """
    if strength == "weak":
        # Weak augmentation: Just a slight translation/crop
        return transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif strength == "strong":
        # Strong augmentation: Resized crop and rotation (makes the digit harder but preserves shape mostly)
        return transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")

def get_dataloaders(batch_size=128, strength="weak", subset_size=10000):
    """
    Loads MNIST dataset, wraps it in the contrastive dataset, and returns dataloaders.
    Sub-sampled to ensure fast CPU execution.
    """
    transform = get_transforms(strength)
    
    # Download and load the standard MNIST training dataset
    full_train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=None
    )
    
    # Download and load the test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    
    # Sub-sample the training dataset to fit within CPU budget
    if subset_size is not None and subset_size < len(full_train_dataset):
        indices = list(range(subset_size))
        base_dataset = Subset(full_train_dataset, indices)
    else:
        base_dataset = full_train_dataset
        
    contrastive_train_dataset = ContrastiveDataset(base_dataset, transform)
    
    train_loader = DataLoader(
        contrastive_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader, len(base_dataset), len(test_dataset)
