import os
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CLASS_NAMES = ["glioma", "meningioma", "pituitary", "normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def ensure_dirs(path_dict):
    for k, p in path_dict.items():
        os.makedirs(p, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob=None, num_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {str(i): report[str(i)] for i in range(num_classes)},
        "confusion_matrix": cm.tolist(),
    }
    if y_prob is not None:
        try:
            auc = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
            metrics["auc_macro"] = float(auc)
        except ValueError:
            metrics["auc_macro"] = None
    return metrics
