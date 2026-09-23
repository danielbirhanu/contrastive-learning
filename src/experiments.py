import os
import torch
import numpy as np

from src.train import train
from src.dataset import get_dataloaders
from src.model import ContrastiveModel
from src.evaluate import extract_embeddings, compute_knn_accuracy, compute_similarity_metrics
from src.utils import set_seed

def run_evaluation(model_path, use_projection_head):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We evaluate on 5000 samples for the evaluation dataloaders
    train_loader, test_loader, _, _ = get_dataloaders(batch_size=256, subset_size=5000, strength="weak")
    
    model = ContrastiveModel(use_projection_head=use_projection_head).to(device)
    model.encoder.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    train_embs, train_labels = extract_embeddings(model, train_loader, device)
    test_embs, test_labels = extract_embeddings(model, test_loader, device)
    
    knn_acc = compute_knn_accuracy(train_embs, train_labels, test_embs, test_labels, k=5)
    s_same, s_diff, gap = compute_similarity_metrics(test_embs, test_labels)
    
    return knn_acc, s_same, s_diff, gap

def run_all_experiments():
    set_seed(42)
    
    # Due to CPU constraints and needing to run 5 experiments sequentially,
    # we aggressively reduce the training subset to 2,500 and epochs to 3.
    subset_size = 2500
    epochs = 3
    
    configs = [
        {"name": "Baseline (T=0.5, Strong Aug, Head)", "temp": 0.5, "aug": "strong", "head": True},
        {"name": "Exp 1: Low Temp (T=0.1)", "temp": 0.1, "aug": "strong", "head": True},
        {"name": "Exp 1: High Temp (T=1.0)", "temp": 1.0, "aug": "strong", "head": True},
        {"name": "Exp 2: Weak Augmentation", "temp": 0.5, "aug": "weak", "head": True},
        {"name": "Exp 3: No Projection Head", "temp": 0.5, "aug": "strong", "head": False},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*50}")
        print(f"Running Configuration: {config['name']}")
        print(f"{'='*50}")
        
        save_path = f"models/exp_{i}.pth"
        
        # 1. Train
        epoch_losses = train(
            batch_size=256,
            epochs=epochs,
            lr=1e-3,
            temperature=config["temp"],
            strength=config["aug"],
            subset_size=subset_size,
            use_projection_head=config["head"],
            save_path=save_path,
            plot_path=f"results/exp_{i}_loss.png",
            sanity_check=False
        )
        
        final_loss = epoch_losses[-1]
        
        # 2. Evaluate
        knn_acc, s_same, s_diff, gap = run_evaluation(save_path, use_projection_head=config["head"])
        
        results.append({
            "name": config["name"],
            "loss": final_loss,
            "knn": knn_acc * 100,
            "gap": gap,
            "s_same": s_same,
            "s_diff": s_diff
        })
        
    print("\n\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<35} | {'Loss':<8} | {'k-NN %':<8} | {'Gap':<8} | {'S_same':<8} | {'S_diff':<8}")
    print("-" * 80)
    for res in results:
        print(f"{res['name']:<35} | {res['loss']:<8.4f} | {res['knn']:<8.2f} | {res['gap']:<8.4f} | {res['s_same']:<8.4f} | {res['s_diff']:<8.4f}")

if __name__ == "__main__":
    run_all_experiments()
