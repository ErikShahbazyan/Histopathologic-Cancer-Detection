#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "
import torch.nn as nn
from src.model import load_all_trained_models
from src.dataset import get_train_val_loaders
from src.evaluate import evaluate, detailed_report, plot_confusion_matrix, plot_comparison
from src import CFG

print(f'Device: {CFG[\"device\"]}')
print('Loading data...')
_, val_loader, df = get_train_val_loaders()
print(f'Total images: {len(df):,}')

print('\nLoading trained models...')
trained = load_all_trained_models()

criterion = nn.BCEWithLogitsLoss()
results = {}
for name, model in trained.items():
    v_loss, v_acc, v_auc, probs, labels = evaluate(model, val_loader, criterion)
    results[name] = {
        'history': {'val_auc': [v_auc], 'val_loss': [v_loss], 'val_acc': [v_acc],
                    'train_loss': [0], 'train_acc': [0]},
        'auc': v_auc, 'probs': probs, 'labels': labels,
    }
    print(f'  {name:<20} AUC: {v_auc:.4f}  Acc: {v_acc:.4f}')

best = max(results, key=lambda n: results[n]['auc'])
print(f'\nBest: {best} (AUC={results[best][\"auc\"]:.4f})')
cm = detailed_report(best, results[best]['labels'], results[best]['probs'])
plot_confusion_matrix(cm, best)
if len(results) > 1:
    plot_comparison(results, best)
"
