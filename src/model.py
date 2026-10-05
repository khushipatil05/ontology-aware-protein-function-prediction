"""3-layer MLP head on frozen ESM-2 embeddings + GO-hierarchy consistency loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GOClassifier(nn.Module):
    def __init__(self, in_dim, n_terms, hidden=(1024, 512), dropout=0.3):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, n_terms),
        )

    def forward(self, x):
        return self.net(x)          # logits


def hierarchy_loss(logits, child_idx, parent_idx):
    """Penalise p(child) > p(parent): a child GO term must never be more likely than its ancestor."""
    p = torch.sigmoid(logits)
    return F.relu(p[:, child_idx] - p[:, parent_idx]).mean()


def total_loss(logits, y, child_idx, parent_idx, lam):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    hier = hierarchy_loss(logits, child_idx, parent_idx) if lam > 0 and len(child_idx) else logits.new_zeros(())
    return bce + lam * hier, bce.detach(), hier.detach()


@torch.no_grad()
def enforce_hierarchy(probs, child_idx, parent_idx):
    """Post-hoc fix: p(parent) = max(p(parent), p(child)) over all (transitive) edges -> always consistent."""
    out = probs.clone()
    if len(child_idx):
        out.index_reduce_(1, parent_idx, probs[:, child_idx], "amax", include_self=True)
    return out
