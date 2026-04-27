# eval.py
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from model import ChestXRayClassifier
from dataset import NIHChestXrayDataset, DESIRED_LABELS
from config import *
from torch.utils.data import DataLoader

def evaluate():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = ChestXRayClassifier(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    df = pd.read_csv(LABEL_FILE)
    # Recreate same test split as training
    _, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    test_dataset = NIHChestXrayDataset(test_df, DATA_ROOT, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    probs = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    preds = (probs > 0.5).astype(int)

    print("=" * 50)
    print("AUROC per disease (5 diseases):")
    for i, name in enumerate(DESIRED_LABELS):
        if len(np.unique(labels[:, i])) > 1:
            auroc = roc_auc_score(labels[:, i], probs[:, i])
        else:
            auroc = 0.0
        print(f"{name:20} : {auroc:.4f}")

    print("\n" + "=" * 50)
    print("Classification Report (threshold 0.5):")
    for i, name in enumerate(DESIRED_LABELS):
        print(f"\n--- {name} ---")
        print(classification_report(labels[:, i], preds[:, i], zero_division=0))

    macro_auroc = roc_auc_score(labels, probs, average='macro')
    print(f"\nOverall macro AUROC: {macro_auroc:.4f}")

if __name__ == "__main__":
    evaluate()