"""Step 4: evaluate on the test split.   python src/evaluate.py --tag mlp_lam0.5"""
import argparse
import json
import numpy as np
import torch

import config as C
from model import GOClassifier, enforce_hierarchy
from metrics import fmax, aupr, prf_at, violation_rate
from train import load_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    tag = ap.parse_args().tag
    ck = torch.load(C.MODELS / f"{tag}.pt", map_location="cpu")
    emb, Y, split, edges = load_data()
    te = np.where(split == "test")[0]
    terms = [l.split("\t") for l in open(C.PROCESSED / "terms.tsv").read().splitlines()[1:]]
    aspect = np.array([t[2] for t in terms])

    model = GOClassifier(ck["in_dim"], ck["n_terms"], C.HIDDEN, C.DROPOUT); model.load_state_dict(ck["state"]); model.eval()
    X = (torch.tensor(emb[te], dtype=torch.float32) - ck["mu"]) / ck["sd"]
    with torch.no_grad():
        P = torch.sigmoid(torch.cat([model(X[i:i + 2048]) for i in range(0, len(X), 2048)]))
    ch, pa = torch.tensor(edges[:, 0]), torch.tensor(edges[:, 1])
    P_fix = enforce_hierarchy(P, ch, pa).numpy(); P = P.numpy()
    Yt = Y[te].toarray()

    res = {}
    names = {"F": "Molecular Function", "P": "Biological Process", "C": "Cellular Component"}
    for asp in "FPC":
        cols = np.where(aspect == asp)[0]
        if len(cols) == 0:
            continue
        r = {}
        for label, pr in (("raw", P), ("hierarchy_enforced", P_fix)):
            f, t = fmax(Yt[:, cols], pr[:, cols])
            mi, ma = aupr(Yt[:, cols], pr[:, cols])
            p, rc, f1 = prf_at(Yt[:, cols], pr[:, cols], t)
            r[label] = {"Fmax": round(f, 4), "threshold": round(float(t), 2), "precision_micro": round(p, 4),
                        "recall_micro": round(rc, 4), "F1_micro": round(f1, 4), "AUPR_micro": round(mi, 4), "AUPR_macro": round(ma, 4)}
        r["violation_rate_raw@0.5"] = round(violation_rate(P, edges[:, 0], edges[:, 1], 0.5), 4)
        r["violation_rate_enforced@0.5"] = round(violation_rate(P_fix, edges[:, 0], edges[:, 1], 0.5), 4)
        res[names[asp]] = r
        print(names[asp], json.dumps(r, indent=1))
    C.RESULTS.mkdir(exist_ok=True)
    json.dump(res, open(C.RESULTS / f"{tag}_test_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
