import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

from src.dataset import get_dataloaders
from src.model import ContrastiveModel
from src.evaluate import extract_embeddings, compute_similarity_metrics
from src.utils import set_seed

def compute_recall_at_k(embs, labels, k=5):
    """Computes Recall@k, excluding the query image itself."""
    # L2 normalize
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    
    # Compute full N x N similarity matrix
    sim_matrix = cosine_similarity(embs)
    
    # Mask self-similarity
    np.fill_diagonal(sim_matrix, -np.inf)
    
    # Get indices of top-k similar images
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:][:, ::-1]
    
    # Check retrieved labels against query labels
    query_labels = labels.reshape(-1, 1)
    retrieved_labels = labels[top_k_indices]
    
    # Class match matrix (N, k)
    matches = (retrieved_labels == query_labels)
    
    # Check if any match in top k
    hit_at_k = np.any(matches, axis=1)
    
    # Compute recall
    recall = np.mean(hit_at_k)
    return recall

def plot_retrieval_examples(images, embs_u, embs_t, labels, num_examples=3, save_path="results/retrieval_examples.png"):
    """Plots qualitative retrieval examples."""
    # Normalize for cosine similarity
    embs_u = embs_u / np.linalg.norm(embs_u, axis=1, keepdims=True)
    embs_t = embs_t / np.linalg.norm(embs_t, axis=1, keepdims=True)
    
    sim_u = cosine_similarity(embs_u)
    sim_t = cosine_similarity(embs_t)
    
    np.fill_diagonal(sim_u, -np.inf)
    np.fill_diagonal(sim_t, -np.inf)
    
    # Pick random queries
    query_indices = np.random.choice(len(images), num_examples, replace=False)
    
    fig, axes = plt.subplots(num_examples, 13, figsize=(18, 2.5 * num_examples))
    
    # Helper to unnormalize image
    mean = np.array([0.1307])
    std = np.array([0.3081])
    
    for row, q_idx in enumerate(query_indices):
        # Top 5 indices
        top5_u = np.argsort(sim_u[q_idx])[-5:][::-1]
        top5_t = np.argsort(sim_t[q_idx])[-5:][::-1]
        
        # Plot Query
        img = images[q_idx].numpy().transpose((1, 2, 0)) * std + mean
        axes[row, 0].imshow(np.clip(img.squeeze(), 0, 1), cmap='gray')
        axes[row, 0].set_title(f"Query: {labels[q_idx]}")
        axes[row, 0].axis('off')
        
        # Spacer
        axes[row, 1].axis('off')
        
        # Plot Untrained Top-5
        for col in range(5):
            idx = top5_u[col]
            img = images[idx].numpy().transpose((1, 2, 0)) * std + mean
            ax = axes[row, col + 2]
            ax.imshow(np.clip(img.squeeze(), 0, 1), cmap='gray')
            # Green border if match, Red if mismatch
            color = "green" if labels[idx] == labels[q_idx] else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and col == 2:
                ax.set_title("Untrained (Top 5)")
                
        # Spacer
        axes[row, 7].axis('off')
        
        # Plot Trained Top-5
        for col in range(5):
            idx = top5_t[col]
            img = images[idx].numpy().transpose((1, 2, 0)) * std + mean
            ax = axes[row, col + 8]
            ax.imshow(np.clip(img.squeeze(), 0, 1), cmap='gray')
            # Green border if match, Red if mismatch
            color = "green" if labels[idx] == labels[q_idx] else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and col == 2:
                ax.set_title("Trained (Top 5)")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved retrieval examples to {save_path}")


def run_retrieval_system():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use 2000 test images for queries and gallery
    _, test_loader, _, _ = get_dataloaders(batch_size=256, subset_size=2000, strength="weak")
    
    # Load raw images for visualization
    all_images = []
    for x, target in test_loader:
        all_images.append(x)
    all_images = torch.cat(all_images, dim=0)
    
    # Models
    model_untrained = ContrastiveModel(use_projection_head=False).to(device)
    model_trained = ContrastiveModel(use_projection_head=False).to(device)
    model_trained.encoder.load_state_dict(torch.load("models/encoder.pth", map_location=device, weights_only=True))
    
    print("Extracting embeddings for retrieval gallery...")
    embs_u, labels_u = extract_embeddings(model_untrained, test_loader, device)
    embs_t, labels_t = extract_embeddings(model_trained, test_loader, device)
    
    # 1. Recall@5
    print("\n--- Metric 1: Recall@5 ---")
    recall_u = compute_recall_at_k(embs_u, labels_u, k=5)
    recall_t = compute_recall_at_k(embs_t, labels_t, k=5)
    print(f"Untrained Recall@5: {recall_u * 100:.2f}%")
    print(f"Trained Recall@5:   {recall_t * 100:.2f}%")
    
    # 2. k-NN Accuracy (Gallery Self-Eval)
    print("\n--- Metric 2: k-NN Accuracy (Gallery Self-Eval) ---")
    knn_u = KNeighborsClassifier(n_neighbors=5).fit(embs_u, labels_u).score(embs_u, labels_u)
    knn_t = KNeighborsClassifier(n_neighbors=5).fit(embs_t, labels_t).score(embs_t, labels_t)
    print(f"Untrained k-NN: {knn_u * 100:.2f}%")
    print(f"Trained k-NN:   {knn_t * 100:.2f}%")
    
    # 3. Similarity Gap
    print("\n--- Metric 3: Similarity Gap ---")
    _, _, gap_u = compute_similarity_metrics(embs_u, labels_u)
    _, _, gap_t = compute_similarity_metrics(embs_t, labels_t)
    print(f"Untrained Gap: {gap_u:.4f}")
    print(f"Trained Gap:   {gap_t:.4f}")
    
    # 4. Qualitative Examples
    print("\n--- Generating Qualitative Examples ---")
    plot_retrieval_examples(all_images, embs_u, embs_t, labels_t, num_examples=4)

if __name__ == "__main__":
    run_retrieval_system()
