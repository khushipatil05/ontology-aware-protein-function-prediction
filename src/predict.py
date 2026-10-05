"""Predict GO terms for new sequences (used later by the Streamlit app).
   python src/predict.py my_proteins.fasta --tag mlp_lam0.5 --top 20
"""
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

import config as C
from data_prep import read_fasta
from embed import embed_shard
from model import GOClassifier, enforce_hierarchy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta"); ap.add_argument("--tag", required=True); ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(C.MODELS / f"{a.tag}.pt", map_location="cpu")
    terms = [l.split("\t") for l in open(C.PROCESSED / "terms.tsv").read().splitlines()[1:]]
    edges = np.load(C.PROCESSED / "hierarchy.npy")
    seqs = read_fasta(a.fasta); ids = list(seqs)
    tok = AutoTokenizer.from_pretrained(C.ESM_MODEL)
    esm = EsmModel.from_pretrained(C.ESM_MODEL, add_pooling_layer=False).to(dev).eval()
    if dev.type == "cuda":
        esm = esm.half()
    E = embed_shard(esm, tok, [seqs[i].upper()[:C.MAX_SEQ_LEN] for i in ids], dev, C.TOKEN_BUDGET)
    clf = GOClassifier(ck["in_dim"], ck["n_terms"], C.HIDDEN, C.DROPOUT); clf.load_state_dict(ck["state"]); clf.eval()
    with torch.no_grad():
        P = torch.sigmoid(clf((torch.tensor(E, dtype=torch.float32) - ck["mu"]) / ck["sd"]))
        P = enforce_hierarchy(P, torch.tensor(edges[:, 0]), torch.tensor(edges[:, 1]))
    for i, acc in enumerate(ids):
        print(f"\n>{acc}")
        for j in torch.argsort(P[i], descending=True)[:a.top]:
            print(f"  {terms[j][1]}  [{terms[j][2]}]  {P[i, j]:.3f}  {terms[j][3]}")


if __name__ == "__main__":
    main()
