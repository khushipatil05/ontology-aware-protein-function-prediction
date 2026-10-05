# Running the pipeline

```
data/raw/uniprot_sprot.fasta      <- dataset 1 (you have it)
data/raw/goa_uniprot_all.gaf      <- dataset 2 (you have it)
data/raw/go-basic.obo             <- dataset 3 (Gene Ontology; python src/download_go.py)
```

| Step | Command | Where |
|------|---------|-------|
| 0. get GO file | `python src/download_go.py` | laptop |
| 1. build labelled dataset | `python src/data_prep.py` | laptop (GAF scan takes a while) |
| 2. ESM-2 650M embeddings | `python src/embed.py [--limit 50000]` | **Colab GPU** |
| 3. train MLP | `python src/train.py --lam 0.5` and `--lam 0 --tag baseline` | Colab or laptop |
| 4. evaluate | `python src/evaluate.py --tag mlp_lam0.5` | Colab or laptop |
| 5. predict new FASTA | `python src/predict.py file.fasta --tag mlp_lam0.5` | GPU preferred |

Only `data/processed/` (not the multi-GB raw files) needs to go to Colab after step 1.
