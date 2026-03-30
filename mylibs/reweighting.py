"""
Confidence-based sample reweighting for training with noisy labels.

After a warm-up phase with standard CE, samples with high loss are identified
as likely mislabeled and down-weighted (or discarded) in subsequent training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np


def compute_sample_weights(model, features, labels, prune_percentile=5, device='cpu'):
    """Compute per-sample weights based on loss magnitude.

    Samples in the top `prune_percentile`% of CE loss are assigned weight 0
    (treated as likely noisy). All other samples get weight 1.

    Args:
        model: trained classifier
        features: (N, D) feature tensor
        labels: (N,) label tensor
        prune_percentile: percentage of highest-loss samples to discard
        device: torch device

    Returns:
        weights: (N,) tensor of 0s and 1s
    """
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        per_sample_loss = nn.functional.cross_entropy(
            logits, labels.to(device), reduction='none'
        ).cpu()

    threshold = np.percentile(per_sample_loss.numpy(), 100 - prune_percentile)
    weights = (per_sample_loss <= threshold).float()
    n_pruned = (weights == 0).sum().item()
    print(f'    Pruned {n_pruned}/{len(labels)} samples ({n_pruned/len(labels):.1%})')
    return weights


def train_model_reweighted(features, labels, test_features, test_labels,
                           feat_dim, num_classes=10, warmup_epochs=10,
                           total_epochs=50, prune_percentile=5,
                           lr=0.01, batch_size=256, device='cpu'):
    """Two-phase training with confidence-based sample reweighting.

    Phase 1 (warm-up): Train with standard CE for `warmup_epochs` to let
    the model learn enough to distinguish clean from noisy samples.

    Phase 2 (reweighted): Compute per-sample weights, discard high-loss
    samples, and continue training on the clean subset.

    Args:
        features: (N, D) training features
        labels: (N,) noisy training labels
        test_features: (M, D) test features
        test_labels: (M,) clean test labels
        feat_dim: feature dimensionality
        num_classes: number of classes
        warmup_epochs: epochs for warm-up phase
        total_epochs: total training epochs
        prune_percentile: % of high-loss samples to discard after warm-up
        lr: learning rate
        batch_size: batch size
        device: torch device

    Returns:
        model: trained classifier
        history: dict with 'train_loss' and 'test_acc' lists
    """
    model = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}

    # Phase 1: Warm-up with standard CE on all data
    train_ds = TensorDataset(features, labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(warmup_epochs):
        model.train()
        epoch_loss = 0.0
        for feats, labs in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            logits = model(feats)
            loss = loss_fn(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * feats.size(0)

        epoch_loss /= len(train_ds)
        history['train_loss'].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            logits = model(test_features.to(device))
            preds = logits.argmax(dim=1).cpu()
            acc = accuracy_score(test_labels.numpy(), preds.numpy())
        history['test_acc'].append(acc)

        if (epoch + 1) % 10 == 0:
            print(f'  [Warmup] Epoch {epoch+1}/{warmup_epochs} — '
                  f'loss: {epoch_loss:.4f}, test acc: {acc:.4f}')

    # Compute sample weights after warm-up
    print(f'  Computing sample weights (pruning top {prune_percentile}% loss)...')
    weights = compute_sample_weights(model, features, labels, prune_percentile, device)

    # Phase 2: Train on clean subset only
    clean_mask = weights > 0
    clean_features = features[clean_mask]
    clean_labels = labels[clean_mask]
    clean_ds = TensorDataset(clean_features, clean_labels)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(warmup_epochs, total_epochs):
        model.train()
        epoch_loss = 0.0
        for feats, labs in clean_loader:
            feats, labs = feats.to(device), labs.to(device)
            logits = model(feats)
            loss = loss_fn(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * feats.size(0)

        epoch_loss /= len(clean_ds)
        history['train_loss'].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            logits = model(test_features.to(device))
            preds = logits.argmax(dim=1).cpu()
            acc = accuracy_score(test_labels.numpy(), preds.numpy())
        history['test_acc'].append(acc)

        if (epoch + 1) % 10 == 0:
            print(f'  [Reweighted] Epoch {epoch+1}/{total_epochs} — '
                  f'loss: {epoch_loss:.4f}, test acc: {acc:.4f}')

    return model, history
