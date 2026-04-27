# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
from model import ChestXRayClassifier
from dataset import NIHChestXrayDataset
from config import *

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(LABEL_FILE)
    # Optional: keep only frontal PA views (uncomment if desired)
    # df = df[df['View Position'] == 'PA']
    df = df.reset_index(drop=True)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = NIHChestXrayDataset(train_df, DATA_ROOT, is_train=True)
    val_dataset = NIHChestXrayDataset(val_df, DATA_ROOT, is_train=False)
    test_dataset = NIHChestXrayDataset(test_df, DATA_ROOT, is_train=False)

    # Windows fix: num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = ChestXRayClassifier(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler()
    best_auroc = 0.0
    patience_counter = 0

    mlflow.set_experiment("chest_xray_5diseases")
    with mlflow.start_run():
        mlflow.log_params({
            "backbone": BACKBONE,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "use_focal": USE_FOCAL_LOSS,
            "img_size": IMG_SIZE,
            "num_classes": NUM_CLASSES
        })

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            model.eval()
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels.cpu().numpy())
            probs = np.vstack(all_probs)
            labels = np.vstack(all_labels)
            auroc = roc_auc_score(labels, probs, average='macro')
            print(f"Epoch {epoch+1}: loss {train_loss/len(train_loader):.4f}, val AUROC {auroc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss/len(train_loader), "val_auroc": auroc}, step=epoch)

            scheduler.step(auroc)

            if auroc > best_auroc:
                best_auroc = auroc
                torch.save(model.state_dict(), "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping")
                    break

        # Test evaluation
        model.load_state_dict(torch.load("best_model.pth"))
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
        probs = np.vstack(all_probs)
        labels = np.vstack(all_labels)
        test_auroc = roc_auc_score(labels, probs, average='macro')
        print(f"Test AUROC: {test_auroc:.4f}")
        mlflow.log_metric("test_auroc", test_auroc)

if __name__ == "__main__":
    main()