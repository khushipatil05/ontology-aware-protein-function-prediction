"""Step 3: train the MLP on ESM-2 embeddings.   python src/train.py [--lam 0.5] [--epochs 40]"""
import argparse
import json
import numpy as np
import scipy.sparse as sp
import torch

import config as C
from model import GOClassifier, total_loss


def load_data():
    emb = np.load(C.PROCESSED / "embeddings.npy")
    n = len(emb)                                       # supports --limit runs of embed.py
    Y = sp.load_npz(C.PROCESSED / "labels.npz")[:n]
    split = np.array([l.split("\t")[1] for l in open(C.PROCESSED / "proteins.tsv").read().splitlines()[1:]])[:n]
    edges = np.load(C.PROCESSED / "hierarchy.npy")
    return emb, Y, split, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=C.HIER_LAMBDA, help="hierarchy-loss weight (0 = baseline)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"mlp_lam{a.lam}"

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb, Y, split, edges = load_data()
    tr, va = np.where(split == "train")[0], np.where(split == "val")[0]

    X = torch.tensor(emb, dtype=torch.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = ((X - mu) / sd).to(dev)
    Yd = torch.tensor(Y.toarray(), dtype=torch.float32).to(dev) if Y.shape[0] * Y.shape[1] < 4e8 else None
    ch = torch.tensor(edges[:, 0], device=dev); pa = torch.tensor(edges[:, 1], device=dev)

    model = GOClassifier(X.shape[1], Y.shape[1], C.HIDDEN, C.DROPOUT).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    def batch_y(idx):
        return Yd[idx] if Yd is not None else torch.tensor(Y[idx.cpu().numpy()].toarray(), dtype=torch.float32, device=dev)

    best, bad, hist = 1e9, 0, []
    C.MODELS.mkdir(exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.tensor(np.random.permutation(tr), device=dev)
        tl = 0.0
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            if len(idx) < 2:
                continue
            loss, _, _ = total_loss(model(X[idx]), batch_y(idx), ch, pa, a.lam)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vi = torch.tensor(va, device=dev)
            vl, vb, vh = total_loss(model(X[vi]), batch_y(vi), ch, pa, a.lam)
        sched.step(vl.item())
        hist.append({"epoch": ep, "train_loss": tl / len(tr), "val_loss": vl.item(),
                     "val_bce": vb.item(), "val_hier": vh.item()})
        print(f"epoch {ep:02d}  train {tl/len(tr):.4f}  val {vl.item():.4f}  (bce {vb:.4f} hier {vh:.4f})")
        if vl.item() < best - 1e-5:
            best, bad = vl.item(), 0
            torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "in_dim": X.shape[1],
                        "n_terms": Y.shape[1], "lam": a.lam}, C.MODELS / f"{tag}.pt")
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print("early stopping"); break
    C.RESULTS.mkdir(exist_ok=True)
    json.dump(hist, open(C.RESULTS / f"{tag}_history.json", "w"), indent=1)
    print("best val loss", best, "-> models/%s.pt" % tag)


if __name__ == "__main__":
    main()
