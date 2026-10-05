"""Step 1: build the labelled dataset from the three raw files.

  uniprot_sprot.fasta  -> sequences
  goa_uniprot_all.gaf  -> protein-GO annotations (streamed; the file is huge)
  go-basic.obo         -> hierarchy (propagation + parent/child edges)

Outputs (data/processed/):
  proteins.tsv      accession, split, length, sequence
  terms.tsv         index, go_id, aspect, name, n_proteins
  labels.npz        sparse multi-hot matrix  [n_proteins x n_terms]  (propagated)
  hierarchy.npy     [n_edges x 2] (child_idx, parent_idx) over kept terms

Usage:  python src/data_prep.py
"""
import gzip
import json
import random
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

import config as C
from ontology import GeneOntology, ROOTS


def _open(path):
    path = str(path)
    return gzip.open(path, "rt", errors="ignore") if path.endswith(".gz") else open(path, errors="ignore")


def read_fasta(path):
    """Swiss-Prot headers look like  >sp|P12345|NAME_HUMAN ...  -> accession = P12345"""
    seqs, acc, buf = {}, None, []
    with _open(path) as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) > 2 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if acc:
            seqs[acc] = "".join(buf)
    return seqs


def clean_sequences(seqs):
    """Length filter, replace rare amino acids by X, drop exact duplicate sequences, truncate for ESM-2."""
    out, seen, dropped_dup = {}, set(), 0
    for acc, s in seqs.items():
        s = s.upper()
        if len(s) < C.MIN_SEQ_LEN:
            continue
        s = "".join(ch if ch in "ACDEFGHIKLMNPQRSTVWY" else "X" for ch in s)
        if s in seen:
            dropped_dup += 1
            continue
        seen.add(s)
        out[acc] = s[: C.MAX_SEQ_LEN]
    print(f"  sequences kept: {len(out)}  (exact duplicates removed: {dropped_dup})")
    return out


def read_gaf(path, wanted, go):
    """Stream the GAF (can be >20 GB uncompressed) and keep only Swiss-Prot accessions."""
    ann = defaultdict(set)
    n_lines = n_kept = 0
    with _open(path) as f:
        for line in f:
            if line[0] == "!":
                continue
            n_lines += 1
            if n_lines % 20_000_000 == 0:
                print(f"  ...scanned {n_lines/1e6:.0f}M lines, kept {n_kept}")
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            acc = c[1]
            if acc not in wanted:
                continue
            if "NOT" in c[3]:              # negative annotations must never be used as positives
                continue
            if C.KEEP_EVIDENCE is not None and c[6] not in C.KEEP_EVIDENCE:
                continue
            t = go.canonical(c[4])
            if t is None:                  # obsolete / unknown term
                continue
            ann[acc].add(t)
            n_kept += 1
    print(f"  scanned {n_lines:,} GAF lines; kept {n_kept:,} annotations for {len(ann):,} proteins")
    return ann


def main():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    print("[1/5] Parsing ontology ...")
    go = GeneOntology(C.OBO_PATH)
    print(f"  {len(go.name):,} terms")

    print("[2/5] Reading FASTA ...")
    seqs = clean_sequences(read_fasta(C.FASTA_PATH))

    print("[3/5] Streaming GAF ...")
    ann = read_gaf(C.GAF_PATH, set(seqs), go)

    print("[4/5] Propagating annotations up the GO graph & selecting terms ...")
    prop = {}
    for acc, terms in ann.items():
        full = {t for t in go.propagate(terms) if t in go.aspect and go.aspect[t] and t not in ROOTS}
        if full:
            prop[acc] = full
    accs = sorted(prop)
    cnt = Counter(t for a in accs for t in prop[a])
    kept = []
    for asp in "FPC":
        cand = [(t, n) for t, n in cnt.items() if go.aspect[t] == asp and n >= C.MIN_TERM_COUNT]
        cand.sort(key=lambda x: -x[1])
        if C.MAX_TERMS_PER_ASPECT:
            cand = cand[: C.MAX_TERMS_PER_ASPECT]
        kept += [t for t, _ in cand]
        print(f"  aspect {asp}: {len(cand)} terms kept")
    t_idx = {t: i for i, t in enumerate(kept)}

    # drop proteins left with no kept term
    accs = [a for a in accs if any(t in t_idx for t in prop[a])]
    rows, cols = [], []
    for r, a in enumerate(accs):
        for t in prop[a]:
            if t in t_idx:
                rows.append(r); cols.append(t_idx[t])
    Y = sp.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(accs), len(kept)))

    # hierarchy edges among kept terms (child -> every kept ancestor)
    edges = [(t_idx[t], t_idx[p]) for t in kept for p in go.ancestors(t) if p in t_idx]
    edges = np.array(edges, dtype=np.int64).reshape(-1, 2)

    print("[5/5] Splitting (80/10/10, random, seed fixed) & saving ...")
    rnd = random.Random(C.SEED)
    order = list(range(len(accs))); rnd.shuffle(order)
    n_tr, n_va = int(0.8 * len(accs)), int(0.1 * len(accs))
    split = ["train"] * len(accs)
    for k, i in enumerate(order):
        split[i] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")

    with open(C.PROCESSED / "proteins.tsv", "w") as f:
        f.write("accession\tsplit\tlength\tsequence\n")
        for a, s in zip(accs, split):
            f.write(f"{a}\t{s}\t{len(seqs[a])}\t{seqs[a]}\n")
    with open(C.PROCESSED / "terms.tsv", "w") as f:
        f.write("index\tgo_id\taspect\tname\tn_proteins\n")
        for i, t in enumerate(kept):
            f.write(f"{i}\t{t}\t{go.aspect[t]}\t{go.name[t]}\t{cnt[t]}\n")
    sp.save_npz(C.PROCESSED / "labels.npz", Y)
    np.save(C.PROCESSED / "hierarchy.npy", edges)
    json.dump({"n_proteins": len(accs), "n_terms": len(kept), "n_edges": int(len(edges)),
               "train": split.count("train"), "val": split.count("val"), "test": split.count("test")},
              open(C.PROCESSED / "stats.json", "w"), indent=2)
    print(f"Done: {len(accs):,} proteins x {len(kept):,} GO terms, {len(edges):,} hierarchy edges")


if __name__ == "__main__":
    main()
