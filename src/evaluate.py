import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_recall_fscore_support,
)

from src import CFG


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(CFG["device"])
        labels = labels.float().unsqueeze(1).to(CFG["device"])
        preds  = model(imgs)
        loss   = criterion(preds, labels)
        total_loss += loss.item() * imgs.size(0)
        probs       = torch.sigmoid(preds)
        correct    += ((probs > 0.5).float() == labels).sum().item()
        total      += imgs.size(0)
        all_probs.extend(probs.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc, all_probs, all_labels


def detailed_report(name, labels, probs, threshold=0.5):
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = (probs > threshold).astype(int)

    print(f"\nModel: {name}")
    print(classification_report(labels, preds, target_names=["Normal", "Cancer"]))

    cm = confusion_matrix(labels, preds)
    return cm


def plot_confusion_matrix(cm, name):
    out = CFG["paths"]["output_dir"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Cancer"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Cancer"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {name}")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results, best_name):
    colors = CFG["colors"]
    out    = CFG["paths"]["output_dir"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

    names = list(results.keys())

    # ── [0,0] Precision / Recall / F1 per model ──────────
    ax = axes[0, 0]
    x  = np.arange(len(names))
    w  = 0.25
    prec_list, rec_list, f1_list = [], [], []
    for n, r in results.items():
        preds = (np.array(r["probs"]) > 0.5).astype(int)
        p, rec, f1, _ = precision_recall_fscore_support(
            r["labels"], preds, average="macro", zero_division=0
        )
        prec_list.append(p); rec_list.append(rec); f1_list.append(f1)
    ax.bar(x - w, prec_list, w, label="Precision", color="steelblue",  edgecolor="black")
    ax.bar(x,     rec_list,  w, label="Recall",    color="seagreen",   edgecolor="black")
    ax.bar(x + w, f1_list,   w, label="F1",        color="darkorange", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.9, 1.01); ax.set_title("Precision / Recall / F1")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, (p, r_, f) in enumerate(zip(prec_list, rec_list, f1_list)):
        ax.text(i - w, p + 0.001, f"{p:.3f}", ha="center", fontsize=7)
        ax.text(i,     r_ + 0.001, f"{r_:.3f}", ha="center", fontsize=7)
        ax.text(i + w, f + 0.001, f"{f:.3f}", ha="center", fontsize=7)

    # ── [0,1] Accuracy per model ──────────────────────────
    ax = axes[0, 1]
    accs = [results[n]["history"]["val_acc"][0] for n in names]
    bars = ax.bar(names, accs, color=[colors[n] for n in names], edgecolor="black")
    ax.set_ylim(min(accs) - 0.02, 1.0)
    ax.set_title("Accuracy per Model"); ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=15)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{acc:.4f}", ha="center", fontweight="bold", fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)

    # ── [0,2] Confusion matrices per model ───────────────
    ax = axes[0, 2]
    ax.axis("off")
    n_models = len(names)
    cell_w = 1.0 / n_models
    for idx, n in enumerate(names):
        preds = (np.array(results[n]["probs"]) > 0.5).astype(int)
        cm    = confusion_matrix(results[n]["labels"], preds)
        cm_norm = cm.astype(float) / cm.sum()
        left = idx * cell_w
        sub  = fig.add_axes([
            axes[0, 2].get_position().x0 + left * axes[0, 2].get_position().width,
            axes[0, 2].get_position().y0,
            cell_w * axes[0, 2].get_position().width * 0.88,
            axes[0, 2].get_position().height,
        ])
        sub.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        sub.set_xticks([0, 1]); sub.set_xticklabels(["N", "C"], fontsize=7)
        sub.set_yticks([0, 1]); sub.set_yticklabels(["N", "C"], fontsize=7)
        sub.set_title(n, fontsize=8)
        for i in range(2):
            for j in range(2):
                sub.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         fontsize=7, fontweight="bold",
                         color="white" if cm_norm[i, j] > 0.5 else "black")

    # ── [1,0] ROC curves ──────────────────────────────────
    ax = axes[1, 0]
    for n, r in results.items():
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        ax.plot(fpr, tpr, label=f"{n} ({r['auc']:.3f})", color=colors[n], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curves"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)

    # ── [1,1] AUC bar chart ───────────────────────────────
    ax = axes[1, 1]
    aucs = [results[n]["auc"] for n in names]
    bars = ax.bar(names, aucs, color=[colors[n] for n in names], edgecolor="black")
    ax.set_ylim(min(aucs) - 0.05, 1.0)
    ax.set_title("Final AUC"); ax.set_ylabel("AUC")
    ax.tick_params(axis="x", rotation=15)
    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{auc:.4f}", ha="center", fontweight="bold", fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)

    # ── [1,2] Train vs Val loss (best model) ─────────────
    ax = axes[1, 2]
    h = results[best_name]["history"]
    ax.plot(h["train_loss"], label="Train", color="steelblue", linewidth=2)
    ax.plot(h["val_loss"],   label="Val",   color="tomato",    linewidth=2)
    ax.set_title(f"Train vs Val Loss - {best_name}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()