import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import argparse

from src.dataset import get_dataloaders
from src.model import ContrastiveModel
from src.utils import set_seed

def extract_embeddings(model, dataloader, device):
    """Extracts representations 'h' and labels."""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, _, target = batch
            else:
                x, target = batch
            
            x = x.to(device)
            # Get encoder output 'h', ignore projected 'z'
            h, _ = model(x)
            embeddings.append(h.cpu())
            labels.append(target)
            
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return embeddings, labels

def compute_knn_accuracy(train_embs, train_labels, test_embs, test_labels, k=5):
    """Trains k-NN classifier and returns test accuracy."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_embs, train_labels)
    acc = knn.score(test_embs, test_labels)
    return acc

def compute_similarity_metrics(embs, labels):
    """Computes average cosine similarity for same and different classes, and their gap."""
    
    idx = np.random.choice(len(embs), min(2000, len(embs)), replace=False)
    embs_sub = embs[idx]
    labels_sub = labels[idx]
    
    # L2 normalize
    embs_sub = embs_sub / np.linalg.norm(embs_sub, axis=1, keepdims=True)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embs_sub)
    
    # Create masks
    labels_matrix = labels_sub.reshape(-1, 1) == labels_sub.reshape(1, -1)
    
    # Remove self-similarity
    np.fill_diagonal(labels_matrix, False)
    
    same_class_sims = sim_matrix[labels_matrix]
    diff_class_sims = sim_matrix[~labels_matrix]
    
    # Remove diagonal from different class mask
    diag_mask = np.eye(len(labels_sub), dtype=bool)
    diff_class_mask = (~labels_matrix) & (~diag_mask)
    diff_class_sims = sim_matrix[diff_class_mask]
    
    s_same = np.mean(same_class_sims)
    s_diff = np.mean(diff_class_sims)
    gap = s_same - s_diff
    
    return s_same, s_diff, gap

def plot_pca(embs_untrained, embs_trained, labels, save_path="results/pca_comparison.png"):
    """
    Plots PCA of untrained vs trained embeddings.
    """
    # Plot 1000 samples
    idx = np.random.choice(len(embs_trained), 1000, replace=False)
    
    pca_untrained = PCA(n_components=2).fit_transform(embs_untrained[idx])
    pca_trained = PCA(n_components=2).fit_transform(embs_trained[idx])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(pca_untrained[:, 0], pca_untrained[:, 1], c=labels[idx], cmap='tab10', alpha=0.7)
    axes[0].set_title("Untrained Encoder (Random Weights)")
    axes[0].axis('off')
    
    scatter2 = axes[1].scatter(pca_trained[:, 0], pca_trained[:, 1], c=labels[idx], cmap='tab10', alpha=0.7)
    axes[1].set_title("Contrastively Trained Encoder")
    axes[1].axis('off')
    
    plt.colorbar(scatter2, ax=axes.ravel().tolist())
    plt.savefig(save_path)
    print(f"Saved PCA visualization to {save_path}")

def evaluate():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    train_loader, test_loader, _, _ = get_dataloaders(batch_size=256, subset_size=5000)
    
    # 2. Setup Models
    # Model A: Untrained
    model_untrained = ContrastiveModel(use_projection_head=False).to(device)
    
    # Model B: Trained
    model_trained = ContrastiveModel(use_projection_head=False).to(device)
    model_trained.encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device, weights_only=True))
    
    print("Extracting embeddings for train set...")
    train_embs_untrained, train_labels_u = extract_embeddings(model_untrained, train_loader, device)
    train_embs_trained, train_labels_t = extract_embeddings(model_trained, train_loader, device)
    
    print("Extracting embeddings for test set...")
    test_embs_untrained, test_labels_u = extract_embeddings(model_untrained, test_loader, device)
    test_embs_trained, test_labels_t = extract_embeddings(model_trained, test_loader, device)
    
    # 3. Metric 1: k-NN Classification Accuracy
    print("\n--- Metric 1: k-NN Accuracy (k=5) ---")
    knn_untrained = compute_knn_accuracy(train_embs_untrained, train_labels_u, test_embs_untrained, test_labels_u)
    knn_trained = compute_knn_accuracy(train_embs_trained, train_labels_t, test_embs_trained, test_labels_t)
    print(f"Untrained Encoder: {knn_untrained * 100:.2f}%")
    print(f"Trained Encoder:   {knn_trained * 100:.2f}%")
    
    # 4. Metric 2: Similarity Metrics
    print("\n--- Metric 2: Cosine Similarity Metrics (Test Set) ---")
    s_same_u, s_diff_u, gap_u = compute_similarity_metrics(test_embs_untrained, test_labels_u)
    s_same_t, s_diff_t, gap_t = compute_similarity_metrics(test_embs_trained, test_labels_t)
    
    print("Untrained Encoder:")
    print(f"  Same-class Sim: {s_same_u:.4f}")
    print(f"  Diff-class Sim: {s_diff_u:.4f}")
    print(f"  Similarity Gap: {gap_u:.4f}")
    
    print("Trained Encoder:")
    print(f"  Same-class Sim: {s_same_t:.4f}")
    print(f"  Diff-class Sim: {s_diff_t:.4f}")
    print(f"  Similarity Gap: {gap_t:.4f}")
    
    # 5. Metric 3: PCA Visualization
    print("\n--- Metric 3: PCA Visualization ---")
    plot_pca(test_embs_untrained, test_embs_trained, test_labels_t)
    
if __name__ == "__main__":
    evaluate()
