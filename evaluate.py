"""
evaluate.py
-----------
Evaluate trained model on test set.
Saves confusion matrix, ROC curve, and metrics JSON.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pt
"""

import os, json, argparse, yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from model   import load_checkpoint
from dataset import DrowsinessDataset, get_val_transform


@torch.no_grad()
def run_eval(model, loader, device, threshold=0.5):
    model.eval()
    labels_all, preds_all, probs_all = [], [], []
    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs   = imgs.to(device)
        probs  = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
        preds  = (probs >= threshold).astype(int)
        probs_all.extend(probs.tolist())
        preds_all.extend(preds.tolist())
        labels_all.extend(labels.numpy().tolist())
    return np.array(labels_all), np.array(preds_all), np.array(probs_all)


def metrics(y, yp, probs):
    cm      = confusion_matrix(y, yp)
    tn,fp,fn,tp = cm.ravel()
    return {
        'accuracy':      round(accuracy_score(y, yp), 4),
        'precision':     round(precision_score(y, yp, zero_division=0), 4),
        'recall_drowsy': round(recall_score(y, yp, zero_division=0), 4),
        'f1_drowsy':     round(f1_score(y, yp, zero_division=0), 4),
        'roc_auc':       round(roc_auc_score(y, probs), 4),
        'FNR':           round(fn/(fn+tp+1e-8), 4),
        'FPR':           round(fp/(fp+tn+1e-8), 4),
        'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),
        'confusion_matrix': cm.tolist()
    }


def print_report(m):
    targets = [
        ('accuracy',      '≥ 0.90', lambda v: v >= 0.90),
        ('recall_drowsy', '≥ 0.95', lambda v: v >= 0.95),
        ('f1_drowsy',     '≥ 0.90', lambda v: v >= 0.90),
        ('roc_auc',       '≥ 0.95', lambda v: v >= 0.95),
        ('FNR',           '≤ 0.05', lambda v: v <= 0.05),
        ('FPR',           '≤ 0.15', lambda v: v <= 0.15),
    ]
    print("\n" + "═"*52)
    print("  EVALUATION RESULTS")
    print("═"*52)
    for k, tgt, fn in targets:
        v = m[k]
        s = "✓" if fn(v) else "✗"
        print(f"  {k:<22} {v:.4f}   target {tgt}  {s}")
    print(f"\n  TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")
    print("═"*52)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='config.yaml')
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--threshold',  type=float, default=0.5)
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt    = args.checkpoint or cfg['paths']['best_model']
    out_dir = cfg['paths']['reports']
    os.makedirs(out_dir, exist_ok=True)

    model = load_checkpoint(ckpt, device, cfg)
    ds    = DrowsinessDataset(os.path.join(cfg['dataset']['root'], 'test'),
                              get_val_transform(cfg['dataset']['image_size']))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg['training']['batch_size'],
        shuffle=False, num_workers=cfg['training']['num_workers'])

    y, yp, probs = run_eval(model, loader, device, args.threshold)
    m = metrics(y, yp, probs)
    print_report(m)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(np.array(m['confusion_matrix']), annot=True, fmt='d',
                cmap='Blues', ax=ax,
                xticklabels=['ALERT','DROWSY'],
                yticklabels=['ALERT','DROWSY'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=120)
    plt.close()

    # ROC curve
    fpr_arr, tpr_arr, _ = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr_arr, tpr_arr, lw=2, label=f"AUC={m['roc_auc']:.4f}")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=120)
    plt.close()

    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"\n[Evaluate] Reports saved → {out_dir}/")
