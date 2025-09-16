import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

from .data import BrainMRIDataset, collect_image_paths
from .model import ResNet50Classifier
from .utils import compute_metrics


def evaluate_checkpoint(ckpt_path: str, data_dir: str, batch_size: int = 32, img_size: int = 224, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    num_classes = cfg.get("num_classes", 4)

    model = ResNet50Classifier(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    X, y = collect_image_paths(data_dir)
    ds = BrainMRIDataset(X, y, img_size=img_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    return metrics


def plot_confusion_matrix(metrics: dict, class_names=None, out_path: str = None):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names, rotation=45, ha="right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    return fig, ax


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args()

    m = evaluate_checkpoint(args.ckpt, args.data_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Saved metrics to {args.out}")
