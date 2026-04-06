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
    out.mkdir(parents=True, exist_ok=True)
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
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold", y=0.98)

    names = list(results.keys())

    # ── [0,0] Precision / Recall / F1 ────────────────────
    ax = axes[0, 0]
    x, w = np.arange(len(names)), 0.25
    prec_list, rec_list, f1_list = [], [], []
    for n, r in results.items():
        preds = (np.array(r["probs"]) > 0.5).astype(int)
        p, rec, f1, _ = precision_recall_fscore_support(
            r["labels"], preds, average="macro", zero_division=0
        )
        prec_list.append(p); rec_list.append(rec); f1_list.append(f1)
    b1 = ax.bar(x - w, prec_list, w, label="Precision", color="#4472C4", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x,     rec_list,  w, label="Recall",    color="#70AD47", edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, f1_list,   w, label="F1",        color="#ED7D31", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0.85, 1.02); ax.set_title("Precision / Recall / F1", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    for i, (p, r_, f) in enumerate(zip(prec_list, rec_list, f1_list)):
        ax.text(i - w, p + 0.002, f"{p:.3f}", ha="center", fontsize=7, fontweight="bold")
        ax.text(i,     r_ + 0.002, f"{r_:.3f}", ha="center", fontsize=7, fontweight="bold")
        ax.text(i + w, f + 0.002, f"{f:.3f}", ha="center", fontsize=7, fontweight="bold")

    # ── [0,1] Accuracy ────────────────────────────────────
    ax = axes[0, 1]
    accs = [results[n]["history"]["val_acc"][0] for n in names]
    bars = ax.bar(names, accs, color=[colors[n] for n in names],
                  edgecolor="white", linewidth=0.5, width=0.5)
    ax.set_ylim(min(accs) - 0.03, 1.01)
    ax.set_title("Accuracy per Model", fontweight="bold"); ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=12, labelsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.4f}", ha="center", fontweight="bold", fontsize=9)

    # ── [0,2] Confusion matrices (красивые) ───────────────
    ax = axes[0, 2]
    ax.set_title("Confusion Matrices", fontweight="bold")
    ax.axis("off")
    n_models = len(names)
    fig_pos = axes[0, 2].get_position()
    pad = 0.01
    cell_w = (fig_pos.width - pad * (n_models - 1)) / n_models

    for idx, n in enumerate(names):
        preds   = (np.array(results[n]["probs"]) > 0.5).astype(int)
        cm      = confusion_matrix(results[n]["labels"], preds)
        cm_norm = cm.astype(float) / cm.sum()
        
        row = idx // 2
        col = idx % 2

        left = fig_pos.x0 + col * (cell_w + pad)
        bottom = fig_pos.y0 + (1 - row) * (fig_pos.height * 0.5)        


        sub = fig.add_axes([
            left,
            bottom,
            cell_w,
            fig_pos.height * 0.44
        ])        
        sub.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        sub.set_xticks([0, 1]); sub.set_xticklabels(["N", "C"], fontsize=8)
        sub.set_yticks([0, 1]); sub.set_yticklabels(["N", "C"], fontsize=8, rotation=90, va="center")
        short = n.replace("EfficientNet-B0", "Eff-B0").replace("DenseNet-121", "Dense-121")
        sub.set_title(short, fontsize=8, fontweight="bold", pad=4)
        for i in range(2):
            for j in range(2):
                val  = cm[i, j]
                disp = f"{val/1000:.1f}k" if val >= 1000 else str(val)
                sub.text(j, i, disp, ha="center", va="center", fontsize=8, fontweight="bold",
                         color="white" if cm_norm[i, j] > 0.5 else "#333333")


    # ── [1,0] ROC curves ──────────────────────────────────
    ax = axes[1, 0]
    for n, r in results.items():
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        ax.plot(fpr, tpr, label=f"{n} (AUC={r['auc']:.3f})", color=colors[n], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_title("ROC Curves", fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # ── [1,1] AUC bar chart ───────────────────────────────
    ax = axes[1, 1]
    aucs = [results[n]["auc"] for n in names]
    bars = ax.bar(names, aucs, color=[colors[n] for n in names],
                  edgecolor="white", linewidth=0.5, width=0.5)
    ax.set_ylim(min(aucs) - 0.05, 1.01)
    ax.set_title("Final AUC", fontweight="bold"); ax.set_ylabel("AUC-ROC")
    ax.tick_params(axis="x", rotation=12, labelsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{auc:.4f}", ha="center", fontweight="bold", fontsize=9)

    # ── [1,2] Train vs Val Loss (best model) ─────────────
    ax = axes[1, 2]
    h = results[best_name]["history"]
    epochs = range(1, len(h["train_loss"]) + 1)
    ax.plot(epochs, h["train_loss"], label="Train", color="#378ADD", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, h["val_loss"],   label="Val",   color="#E24B4A", linewidth=2, marker="o", markersize=4)
    ax.set_title(f"Train vs Val Loss — {best_name}", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(list(epochs))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()