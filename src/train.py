import sys
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from src import CFG
from src.evaluate import evaluate
from src.model import build_model, BUILDERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            CFG["paths"]["log_dir"]
            / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
        ),
    ],
)
log = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(CFG["device"])
        labels = labels.float().unsqueeze(1).to(CFG["device"])
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += ((torch.sigmoid(preds) > 0.5).float() == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def train_model(model, name, train_loader, val_loader, num_epochs=None):
    if num_epochs is None:
        num_epochs = CFG["training"]["num_epochs"]
    patience = CFG["training"]["patience"]

    model = model.to(CFG["device"])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["training"]["lr"],
        weight_decay=CFG["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    best_auc     = 0
    best_weights = None
    no_improve   = 0

    log.info(f"{'='*55}")
    log.info(f"  {name}")
    log.info(f"{'='*55}")
    log.info(f"{'Epoch':>5} | {'T-Loss':>7} | {'V-Loss':>7} | {'V-Acc':>6} | {'V-AUC':>6}")
    log.info("-" * 42)

    for epoch in range(1, num_epochs + 1):
        t_loss, t_acc              = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc, v_auc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["val_auc"].append(v_auc)

        star = " *" if v_auc > best_auc else ""
        log.info(f"{epoch:>5} | {t_loss:>7.4f} | {v_loss:>7.4f} | {v_acc:>6.4f} | {v_auc:>6.4f}{star}")

        if v_auc > best_auc:
            best_auc     = v_auc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stopping (best AUC: {best_auc:.4f})")
                break

    model.load_state_dict(best_weights)

    safe_name = name.replace(" ", "_").replace("-", "_")
    save_path = CFG["paths"]["checkpoint_dir"] / f"{safe_name}.pt"
    torch.save(model.state_dict(), save_path)
    log.info(f"  Saved checkpoint -> {save_path}")

    return model, history, best_auc


def run_training():
    from src.dataset import get_train_val_loaders
    from src.evaluate import evaluate as eval_fn, plot_comparison, plot_confusion_matrix, detailed_report

    log.info(f"Device: {CFG['device']}")
    log.info("Loading data...")
    train_loader, val_loader, df = get_train_val_loaders()
    log.info(f"Total images: {len(df):,}")

    criterion = nn.BCEWithLogitsLoss()
    results = {}

    for name in BUILDERS:
        model = build_model(name)
        trained, history, best_auc = train_model(model, name, train_loader, val_loader)
        _, _, final_auc, probs, labels = eval_fn(trained, val_loader, criterion)
        results[name] = {"history": history, "auc": best_auc, "probs": probs, "labels": labels}
        trained.cpu()
        torch.cuda.empty_cache()

    log.info("\n=== RESULTS ===")
    for name, res in results.items():
        log.info(f"  {name:<20} AUC: {res['auc']:.4f}")

    best_name = max(results, key=lambda n: results[n]["auc"])
    log.info(f"Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")

    cm = detailed_report(best_name, results[best_name]["labels"], results[best_name]["probs"])
    plot_confusion_matrix(cm, best_name)
    plot_comparison(results, best_name)


if __name__ == "__main__":
    run_training()
