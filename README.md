# Contrastive Learning and Representation Geometry

This repository contains a complete implementation of a self-supervised contrastive learning pipeline (based on InfoNCE / SimCLR concepts) evaluated on the MNIST dataset.
## Project Structure

```
├── src/
│   ├── dataset.py            # Data loading & weak/strong augmentations
│   ├── model.py              # CNN Encoder and MLP Projection Head
│   ├── loss.py               # InfoNCE loss with in-batch negatives
│   ├── train.py              # Main training loop
│   ├── evaluate.py           # Evaluation (k-NN, Similarity Gap, PCA)
│   ├── experiments.py        # Controlled hyperparameter experiments
│   ├── geometry_analysis.py  # Analysis of embedding variance & failure cases
│   ├── retrieval.py          # Nearest neighbor retrieval and visualization
│   └── utils.py              # Reproducibility utilities (seeds)
├── Final_Report.md           # Comprehensive analysis and results
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup & Installation

This project uses modern Python packaging via `uv` (or standard `pip`). The code is designed to run efficiently on CPU.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/danielbirhanu/contrastive-learning
    cd contrastive-learning
    ```

2.  **Create a virtual environment and install dependencies:**
    If using `uv` (Recommended):
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```
    If using standard `pip`:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Running the Pipeline

The project is broken down into modular scripts. To reproduce the results, run them from the root directory and set the `PYTHONPATH`:

**1. Train the Baseline Model**
Trains the encoder and projection head for 5 epochs on a 10,000 image subset.
```bash
PYTHONPATH=. python src/train.py
```

**2. Evaluate Representation Quality**
Evaluates the trained encoder against an untrained baseline using k-NN, Similarity Gap, and PCA.
```bash
PYTHONPATH=. python src/evaluate.py
```

**3. Run Controlled Experiments**
Tests different temperatures ($T=0.1, 0.5, 1.0$), augmentation strengths, and the necessity of the projection head.
```bash
PYTHONPATH=. python src/experiments.py
```

**4. Analyze Geometry & Failure Cases**
Generates embedding variance statistics and class-centroid similarity matrices to hunt for representation collapse and class confusion.
```bash
PYTHONPATH=. python src/geometry_analysis.py
```

**5. Evaluate Retrieval System**
Computes Recall@5 and generates qualitative nearest-neighbor retrieval comparisons.
```bash
PYTHONPATH=. python src/retrieval.py
```
