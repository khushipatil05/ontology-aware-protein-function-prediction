# Ontology-Aware Protein Function Prediction from Primary Sequence Using Frozen Protein Language Models

Predict a protein's biological function directly from its amino acid sequence, addressing the huge gap between
known sequences and experimentally verified functions. A pretrained **ESM-2 (650M)** protein language model is used
as a **frozen feature extractor** (no expensive fine-tuning); its embeddings feed a multi-label **MLP** that predicts
**Gene Ontology (GO)** terms across Molecular Function (MF), Biological Process (BP) and Cellular Component (CC).
The model is **ontology-aware**: a hierarchy-consistency loss discourages predicting a child GO term with a higher
probability than its parent, a common biological-consistency failure of simpler models.

## Team

Khushi Patil
Diya Kalghatgi
Nidhi Nayak. 

Guide: Prof. AshaRani Patil.

## Where we stand (updated 2026-10-06, Phase 1)

**Status: core pipeline built and trained end to end. Validation of results and deployment are next.**

- [x] Literature survey, research gap, problem statement, objectives
- [x] Design (ER, sequence and data-flow diagrams), requirement and feasibility analysis
- [x] Data collection: UniProt Swiss-Prot (FASTA), GOA (GAF), Gene Ontology (OBO)
- [x] Data preprocessing and label building (GO propagation, term selection, split)
- [x] ESM-2 650M embedding extraction (Google Colab T4 GPU)
- [x] Model development: baseline MLP and ontology-aware MLP
- [x] First evaluation (Fmax, AUPR, hierarchy violations)
- [ ] **Validate results** (stricter split, experimental-only labels, per-ontology violation metric)
- [ ] Hyper-parameter tuning, rare-term analysis, ablations
- [ ] Streamlit app (`streamlit_app/`) and prediction history
- [ ] Final report

## Pipeline at a glance

```
uniprot_sprot.fasta ┐
goa_uniprot_all.gaf ├─► data_prep.py ─► labelled dataset ─► embed.py (ESM-2 650M, frozen) ─► train.py (MLP)
go-basic.obo        ┘     (propagate annotations up the GO graph)                          ─► evaluate.py ─► predict.py / app
```

## First results (test split)

| Model | Ontology | Fmax | AUPR micro | AUPR macro | Hierarchy violations @0.5 (raw) |
|---|---|---|---|---|---|
| baseline (no hierarchy loss) | MF | 0.9542 | 0.9864 | 0.8980 | 0.0071 |
| baseline | BP | 0.9430 | 0.9671 | 0.7298 | 0.0071 |
| baseline | CC | 0.9525 | 0.9642 | 0.6919 | 0.0071 |
| ontology-aware (lambda = 0.5) | MF | 0.9542 | 0.9870 | 0.8958 | 0.0048 |
| ontology-aware | BP | 0.9436 | 0.9677 | 0.7228 | 0.0048 |
| ontology-aware | CC | 0.9523 | 0.9630 | 0.6824 | 0.0048 |

**How to read these honestly**
- The hierarchy loss reduced inconsistent predictions (violations 0.0071 to 0.0048, about one third fewer) while
  keeping accuracy essentially unchanged (Fmax differences are 0.001 or less, i.e. within noise).
- The accuracy numbers are **much higher than published CAFA-style results**. Likely causes: random train/test split
  (similar proteins on both sides), inclusion of computationally inferred (IEA) annotations, and many very common
  ancestor terms that are easy to predict. They should be validated before being presented as real performance.
- The violation rate is currently computed over all terms together, so it is identical for the three ontologies.
  It should be computed per ontology (known issue, see Next steps).

## Next steps

1. Re-evaluate with a **similarity-based split** (e.g. cluster sequences at 30-50% identity, keep clusters whole).
2. Re-run with **experimental evidence only** (`KEEP_EVIDENCE` in `src/config.py`) and compare.
3. Report Fmax on **rarer, more specific terms** as well as overall.
4. Fix the **per-ontology violation metric** in `src/evaluate.py`.
5. Try several lambda values (0, 0.1, 0.5, 1.0) and plot loss curves from `results/*_history.json`.
6. Build the **Streamlit app** around `src/predict.py`.

Full dated history: **[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md)** | Phase 1 talk prep: **[docs/PHASE1_PRESENTATION_GUIDE.md](docs/PHASE1_PRESENTATION_GUIDE.md)**

## How to run (quick reference)

Full details, troubleshooting and settings: **[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md)**

**Rule of thumb:** raw datasets stay on the laptop; only `processed.zip` goes to Google Colab.

**On the laptop**

```bash
# put uniprot_sprot.fasta and goa_uniprot_all.gaf in data/raw/
python src/download_go.py          # dataset 3: go-basic.obo -> data/raw/  (or download manually, see guide)
python src/data_prep.py            # builds data/processed/ (slow GAF scan, no GPU needed)
python src/make_processed_zip.py   # creates data/processed.zip
git push                           # so Colab can fetch the code
```

**On Google Drive:** upload `data/processed.zip` to `MyDrive/ontology-aware-protein-function-prediction/data/`

**On Colab (T4 GPU):** open `notebooks/01_data_preprocessing.ipynb` and run all cells. It does:
ESM-2 650M embeddings (`embed.py`) → MLP baseline + ontology-aware model (`train.py`) → metrics (`evaluate.py`).
Start with `EMBED_LIMIT = 50000`, then set it to `None` for the full run (resumes if Colab disconnects).

| Script | Purpose | Runs on |
|--------|---------|---------|
| `src/download_go.py` | download Gene Ontology file | laptop |
| `src/data_prep.py` | FASTA + GAF + OBO → labelled dataset | laptop |
| `src/make_processed_zip.py` | package processed data for Drive | laptop |
| `src/embed.py` | frozen ESM-2 650M embeddings | Colab GPU |
| `src/train.py` | 3-layer MLP + GO-hierarchy loss | Colab |
| `src/evaluate.py` | Fmax, AUPR, hierarchy violations | Colab |
| `src/predict.py` | predict GO terms for a new FASTA | Colab / GPU |
