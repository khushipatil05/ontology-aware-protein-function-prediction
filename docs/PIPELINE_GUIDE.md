# Pipeline Guide — what to run, where, and with which data

This is the reference for running the whole project. Follow it top to bottom.

## 0. Big picture

```
 LAPTOP (no GPU needed)                              GOOGLE COLAB (T4 GPU)
 ──────────────────────                              ─────────────────────
 uniprot_sprot.fasta ┐
 goa_uniprot_all.gaf ├─► data_prep.py ─► data/processed/ ─► processed.zip ─► Drive ─► notebook:
 go-basic.obo        ┘                                                         embed.py (ESM-2 650M)
                                                                               train.py  (baseline + ontology-aware)
                                                                               evaluate.py
                                                                               ─► models/, results/ (saved on Drive)
```

| Step | Script | Runs on | Reads | Writes |
|------|--------|---------|-------|--------|
| 0 | `src/download_go.py` | Laptop | internet | `data/raw/go-basic.obo` |
| 1 | `src/data_prep.py` | Laptop | the 3 raw files in `data/raw/` | `data/processed/*` |
| 1b | `src/make_processed_zip.py` | Laptop | `data/processed/*` | `data/processed.zip` |
| 2 | `src/embed.py` | **Colab GPU** | `proteins.tsv` | `embeddings.npy` (+ shards) |
| 3 | `src/train.py` | Colab | embeddings + labels | `models/*.pt`, `results/*_history.json` |
| 4 | `src/evaluate.py` | Colab | model + test split | `results/*_test_metrics.json` |
| 5 | `src/predict.py` | Colab (GPU preferred) | a FASTA file | printed GO predictions |

**Rule of thumb:** raw files stay on the laptop. Only `processed.zip` goes to Colab/Drive.

## 1. Laptop setup (one time)

```bash
git clone https://github.com/khushipatil05/ontology-aware-protein-function-prediction.git
cd ontology-aware-protein-function-prediction
pip install numpy scipy scikit-learn tqdm        # enough for steps 0-1b
```

Put your downloaded files in `data/raw/` (exact names matter — or edit the paths in `src/config.py`):

```
data/raw/uniprot_sprot.fasta       (dataset 1)
data/raw/goa_uniprot_all.gaf       (dataset 2)   .gz also works
data/raw/go-basic.obo              (dataset 3 — step 2 below)
```

## 2. Get dataset 3 — Gene Ontology (`go-basic.obo`, ~30 MB)

Try in this order:
1. `python src/download_go.py` (tries several mirrors).
2. In a browser open `https://purl.obolibrary.org/obo/go/go-basic.obo`, press Ctrl+S, save as `go-basic.obo` into `data/raw/`. If your college network blocks it, use a phone hotspot.
3. If everything is blocked locally, download it inside Colab (`!wget https://purl.obolibrary.org/obo/go/go-basic.obo`), then download it from Colab to your laptop.

Check: the first line of the file should be `format-version: 1.2`.

## 3. Build the labelled dataset (laptop)

```bash
python src/data_prep.py
```

- Scanning the huge GAF file takes a while (it prints progress). Leave it running.
- It writes to `data/processed/`: `proteins.tsv`, `terms.tsv`, `labels.npz`, `hierarchy.npy`, `stats.json`.
- Look at `stats.json`: number of proteins, GO terms and train/val/test sizes.

Then make the upload file:

```bash
python src/make_processed_zip.py       # creates data/processed.zip
```

## 4. Put things on Google Drive

Create this folder structure in your Drive (My Drive):

```
MyDrive/ontology-aware-protein-function-prediction/
    data/
        processed.zip        <-- upload this one file
```

The code (`src/`) is fetched from GitHub by the notebook, so **commit and push your repo first**:

```bash
git add src docs notebooks requirements.txt README.md .gitignore
git commit -m "Add data prep, embedding, training and evaluation pipeline"
git push
```

(Raw data and `processed/` are in `.gitignore` on purpose — they are too big for GitHub.)

## 5. Run the Colab notebook

1. Open `notebooks/protein_function_colab.ipynb` (from GitHub: colab.research.google.com → GitHub tab, or upload it).
2. Runtime → Change runtime type → **T4 GPU**.
3. Edit the settings cell (cell 2) if needed. Leave `EMBED_LIMIT = 50000` for the first run.
4. Runtime → Run all. Approve the Drive access popup.

**Recommended order:**
1. First run with `EMBED_LIMIT = 50000`: checks that everything works end to end and gives first results.
2. Then set `EMBED_LIMIT = None` and run again for all proteins. Shards already computed are reused, so nothing is repeated. The full run takes several hours.

**If Colab disconnects:** reopen the notebook, re-run cells 1–5, then re-run the embedding cell. It resumes from the last finished shard.

## 6. What you get

On Drive, inside the project folder:
- `models/baseline.pt`, `models/mlp_lam0.5.pt` — trained classifiers
- `results/*_test_metrics.json` — Fmax, AUPR, precision/recall/F1 (micro), hierarchy-violation rate, per ontology (MF / BP / CC)
- `results/*_history.json` — loss per epoch (for plots)
- `data/processed/embeddings.npy` — keep it, you never need to recompute it

Download `models/` and `results/` to your laptop and commit them (they are small). Do **not** commit `embeddings.npy`.

## 7. Settings you may want to change (`src/config.py`)

| Setting | Default | Meaning |
|---------|---------|---------|
| `KEEP_EVIDENCE` | `None` (all annotations) | set to `{"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"}` for experimental-only labels (matches the proposal) — then re-run `data_prep.py` |
| `MIN_TERM_COUNT` | 50 | drop GO terms with fewer proteins than this |
| `MAX_TERMS_PER_ASPECT` | 1500 | cap on terms per ontology |
| `HIER_LAMBDA` | 0.5 | weight of the hierarchy loss (0 = baseline) |
| `TOKEN_BUDGET` | 16000 | lower to 8000 if the GPU runs out of memory |
| `ESM_MODEL` / `EMB_DIM` | 650M / 1280 | smaller options: `facebook/esm2_t12_35M_UR50D` / 480 |

If you change anything that affects `data_prep.py` or the model, delete `data/processed/embeddings*` and `models/` so old files are not mixed with new ones.

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Missing ... in data/processed` in the notebook | `data_prep.py` didn't finish, or `processed.zip` isn't in `Drive/.../data/` |
| `src/config.py not found` | You haven't pushed the code to GitHub yet (step 4) |
| CUDA out of memory | `--token_budget 8000` in the embedding cell |
| "NO GPU" printed | Runtime → Change runtime type → T4 GPU, then restart |
| Colab disconnected mid-embedding | Re-run cells 1–5 and the embedding cell; it resumes |
| `go-basic.obo` won't download | See section 2, options 2 and 3 |
| Metrics look too good | The split is random, so similar proteins can appear in train and test. A similarity-based split would be stricter |
| `data_prep.py` is slow | Normal for the GAF file; it scans every line once |

## 9. Evaluation notes (for the report)

- **Fmax**: best protein-centric F1 over thresholds (CAFA standard), per ontology.
- **AUPR**: micro (all protein–term pairs) and macro (average over terms).
- **Hierarchy violations**: fraction of predicted terms whose ancestor was *not* predicted. Compare `baseline` vs `mlp_lam0.5`, and raw vs `hierarchy_enforced` predictions.
- Labels are propagated to ancestors (true-path rule); the three root terms are excluded.
- `NOT` annotations are never used as positive labels.
