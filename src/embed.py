"""Step 2: extract frozen ESM-2 (650M) mean-pooled embeddings.

Run this on a GPU (Google Colab T4 is enough with fp16). Writes shards so that you can resume
after a disconnect: simply run the same command again.

  python src/embed.py                 # all proteins
  python src/embed.py --limit 20000   # quick first experiment on a subset

Outputs: data/processed/embeddings/shard_00000.npy (float16) ... and then embeddings.npy (merged).
"""
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

import config as C


def load_proteins(limit=None):
    accs, seqs = [], []
    with open(C.PROCESSED / "proteins.tsv") as f:
        next(f)
        for line in f:
            a, _, _, s = line.rstrip("\n").split("\t")
            accs.append(a); seqs.append(s)
    if limit:
        accs, seqs = accs[:limit], seqs[:limit]
    return accs, seqs


def make_batches(idx, seqs, budget):
    """Length-sorted batches under a token budget -> little padding, no OOM."""
    idx = sorted(idx, key=lambda i: len(seqs[i]))
    batch, longest = [], 0
    for i in idx:
        L = len(seqs[i]) + 2
        if batch and max(longest, L) * (len(batch) + 1) > budget:
            yield batch
            batch, longest = [], 0
        batch.append(i); longest = max(longest, L)
    if batch:
        yield batch


@torch.no_grad()
def embed_shard(model, tok, seqs, device, budget):
    out = np.zeros((len(seqs), C.EMB_DIM), dtype=np.float16)
    for b in make_batches(range(len(seqs)), seqs, budget):
        enc = tok([seqs[i] for i in b], return_tensors="pt", padding=True, truncation=True,
                  max_length=C.MAX_SEQ_LEN + 2).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0                                           # drop <cls>
        last = enc["attention_mask"].sum(1) - 1
        mask[torch.arange(mask.size(0)), last] = 0               # drop <eos>
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out[b] = pooled.float().cpu().numpy().astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--token_budget", type=int, default=C.TOKEN_BUDGET)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found. The 650M model on CPU is extremely slow; use Colab (Runtime > GPU) "
              "or set ESM_MODEL to facebook/esm2_t12_35M_UR50D (EMB_DIM=480) in config.py.")
    tok = AutoTokenizer.from_pretrained(C.ESM_MODEL)
    model = EsmModel.from_pretrained(C.ESM_MODEL, add_pooling_layer=False).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    accs, seqs = load_proteins(args.limit)
    d = C.PROCESSED / "embeddings"; d.mkdir(exist_ok=True)
    n_shards = (len(seqs) + C.SHARD_SIZE - 1) // C.SHARD_SIZE
    for s in tqdm(range(n_shards), desc="shards"):
        path = d / f"shard_{s:05d}.npy"
        if path.exists():
            continue                                             # resume
        chunk = seqs[s * C.SHARD_SIZE:(s + 1) * C.SHARD_SIZE]
        np.save(path, embed_shard(model, tok, chunk, device, args.token_budget))
    emb = np.concatenate([np.load(d / f"shard_{s:05d}.npy") for s in range(n_shards)])
    np.save(C.PROCESSED / "embeddings.npy", emb)
    np.save(C.PROCESSED / "embedded_n.npy", np.array([len(emb)]))
    print("Saved", emb.shape)


if __name__ == "__main__":
    main()
