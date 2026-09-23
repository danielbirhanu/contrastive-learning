import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from src.dataset import get_dataloaders
from src.model import ContrastiveModel
from src.evaluate import extract_embeddings
from src.utils import set_seed

def analyze_geometry():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load 2000 test samples
    _, test_loader, _, _ = get_dataloaders(batch_size=256, subset_size=2000, strength="weak")
    
    # Untrained Model
    model_untrained = ContrastiveModel(use_projection_head=False).to(device)
    
    # Trained Model (from Phase 4, saved as models/encoder.pth)
    model_trained = ContrastiveModel(use_projection_head=False).to(device)
    model_trained.encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device, weights_only=True))
    
    print("Extracting representations...")
    embs_u, labels_u = extract_embeddings(model_untrained, test_loader, device)
    embs_t, labels_t = extract_embeddings(model_trained, test_loader, device)
    
    # L2 normalize embeddings for cosine distance computations
    embs_u = embs_u / np.linalg.norm(embs_u, axis=1, keepdims=True)
    embs_t = embs_t / np.linalg.norm(embs_t, axis=1, keepdims=True)
    
    # 1. Embedding Variance (Is there collapse?)
    var_u = np.var(embs_u, axis=0).mean()
    var_t = np.var(embs_t, axis=0).mean()
    print(f"\nMean Embedding Variance (per dimension):")
    print(f"Untrained: {var_u:.6f}")
    print(f"Trained:   {var_t:.6f}")
    
    # 2. Average Pairwise Similarity
    sim_matrix_t = cosine_similarity(embs_t)
    np.fill_diagonal(sim_matrix_t, 0) # exclude self
    avg_pairwise_t = sim_matrix_t.sum() / (len(embs_t) * (len(embs_t) - 1))
    print(f"\nAverage Pairwise Similarity (Trained): {avg_pairwise_t:.4f}")
    
    # 3. Class-Level Geometry Analysis
    classes = np.unique(labels_t)
    class_centroids_t = np.array([embs_t[labels_t == c].mean(axis=0) for c in classes])
    # Normalize centroids
    class_centroids_t = class_centroids_t / np.linalg.norm(class_centroids_t, axis=1, keepdims=True)
    
    # Compute similarity between class centroids
    centroid_sim_matrix = cosine_similarity(class_centroids_t)
    
    print("\nIdentifying Failure Cases (Class Confusion)...")
    np.fill_diagonal(centroid_sim_matrix, -1) # Ignore self
    max_confusions = []
    for i, c in enumerate(classes):
        most_similar_idx = np.argmax(centroid_sim_matrix[i])
        max_confusions.append((c, classes[most_similar_idx], centroid_sim_matrix[i][most_similar_idx]))
        
    # Sort to find the worst separated classes
    max_confusions.sort(key=lambda x: x[2], reverse=True)
    
    worst_c1, worst_c2, worst_sim = max_confusions[0]
    print(f"WORST SEPARATION: Class {worst_c1} and Class {worst_c2} (Similarity: {worst_sim:.4f})")
    
    # Plotting confusion matrix of centroids
    plt.figure(figsize=(8, 6))
    plt.imshow(cosine_similarity(class_centroids_t), cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.title("Class Centroid Similarity Matrix (Trained)")
    plt.xlabel("Digit Class")
    plt.ylabel("Digit Class")
    plt.xticks(classes)
    plt.yticks(classes)
    plt.savefig("results/geometry_analysis.png")
    print("Saved geometry analysis plot to results/geometry_analysis.png")

if __name__ == "__main__":
    analyze_geometry()
