"""
src/classify.py
Auxiliary MLP classifier trained on top of frozen ResNet50 embeddings.
Predicts clothing category from a 2048-d embedding vector.
"""

import os 
import numpy as np
import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    """
    Loads precomputed embeddings + category labels from disk.
    No images needed - training is purely on the 2048-d vectors.
    """

    def __init__(self, embeddings_path, metadata_path):
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.metadata = pd.read_csv(metadata_path)

        # Validate alignment
        assert len(self.embeddings) == len(self.metadata), \
            "Mismatch: embeddings and metadata row counts differ."

        # FIX: Dynamically create the 'category' column by extracting the parent folder name from img_path
        # Example path: data/deepfashion/tshirt/0.jpg -> category: tshirt
        self.metadata['category'] = self.metadata['img_path'].apply(lambda x: os.path.basename(os.path.dirname(x)))

        # Build label encoder: category string -> integer index
        self.classes = sorted(self.metadata['category'].unique().tolist())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.labels = self.metadata['category'].map(self.class_to_idx).values

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx]) # (2048, )
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
    
# ─────────────────────────────────────────────
# 2. MLP MODEL
# ─────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    Lightweight 3-layer MLP on top of frozen ResNet embeddings.

    Architecture: 2048 -> 512 -> 128 -> num_classes
    Each hidden layer: linear -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(128, num_classes)
            # No Softmax here - CrossEntropyLoss includes it internally
        )

    def forward(self, x):
        return self.network(x)

# ─────────────────────────────────────────────
# 3. TRAINER
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, model, device, lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(y)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)

            return total_loss / total, correct / total 

    def fit(self, train_loader, val_loader, epochs=20):
        print(f"\n{'Epoch':<8} {'Train Loss': <14} {'Val Loss': <14} {'Val Acc':<10}")
        print("-" * 50)

        for epoch in range(1, epochs + 1):
            # FIX: changed run_epoch to _run_epoch
            train_loss, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            # FIX: val loss -> val_loss
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"{epoch:<8} {train_loss:<14.4f} {val_loss:<14.4f} {val_acc:<10.4f}")

        print("\n✅ Training complete.")
        # FIX: histoy -> history
        return self.history

# ─────────────────────────────────────────────
# 4. INFERENCE HELPER (used in app.py)
# ─────────────────────────────────────────────

def predict_category(embedding: np.ndarray, model: MLPClassifier,
                     classes: list, device: torch.device) -> tuple[str, float]:
    
    """
    Predicts the clothing category for a single 2048-d embedding.

    Returns: (predicted_class_name, confidence_score)
    """
    model.eval()
    x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
    return classes[idx.item()], conf.item()


# ─────────────────────────────────────────────
# 5. MAIN - TRAIN AND SAVE
# ─────────────────────────────────────────────

def main():
    # ── Paths ────────────────────────────────
    EMBEDDINGS_PATH = "embeddings/embeddings.npy"
    METADATA_PATH = "embeddings/metadata.csv"
    MODEL_SAVE_PATH = "models/mlp_classifier.pth"
    os.makedirs("models", exist_ok=True)

    # ── Hyperparameters ───────────────────────
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3
    VAL_SPLIT = 0.2 # 80/20 train-val split
    DROPOUT = 0.4

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading embeddings dataset...")
    dataset = EmbeddingDataset(EMBEDDINGS_PATH, METADATA_PATH)
    num_classes = len(dataset.classes)
    print(f"Classes ({num_classes}): {dataset.classes}")

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Build Model ───────────────────────────
    model = MLPClassifier(input_dim=2048, num_classes=num_classes, dropout=DROPOUT)
    trainer = Trainer(model, DEVICE, lr=LR)

    # ── Train ───────────────────────────
    # FIX: trainer_fit -> trainer.fit
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)

    # ── Save ───────────────────────────
    # Save model weights + class list together so we can reload cleanly
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "input_dim": 2048,
        "num_classes": num_classes,
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# FIX: Un-indented the name == main block so it actually runs
if __name__ == "__main__":
    main()