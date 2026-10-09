# Project log

Newest entries at the bottom. Add a new entry whenever something meaningful is done.

## Milestone 1 — Design and planning (done before Phase 1)
- Wrote problem statement, objectives, literature survey (DeepGOPlus, DeepGraphGO, ESM, ProtTrans, ontology-aware losses, AI survey) and research gap.
- Designed the system: ER diagram, sequence diagram, data-flow diagram; requirement and feasibility analysis.
- Chose the approach: frozen ESM-2 embeddings + multi-label MLP + GO-hierarchy-aware loss, Streamlit front end.
- Collected datasets: UniProt Swiss-Prot FASTA, GOA GAF, Gene Ontology OBO.

## Milestone 2 — Code pipeline written
Repo structure (`data/`, `src/`, `models/`, `notebooks/`, `results/`, `docs/`) and the following scripts:
- `src/download_go.py`: downloads `go-basic.obo` from several mirrors.
- `src/ontology.py`: own OBO parser; maps alternate IDs, drops obsolete terms, follows `is_a` and `part_of` links to find ancestors.
- `src/data_prep.py`: reads Swiss-Prot, removes exact duplicate sequences, streams the large GAF file keeping only Swiss-Prot proteins, ignores `NOT` annotations, propagates labels to ancestors (true-path rule), excludes the 3 root terms, keeps terms with at least 50 proteins (max 1500 per ontology), makes an 80/10/10 random split.
- `src/embed.py`: frozen ESM-2 650M, mean-pooled embeddings (1280-d), fp16, length-sorted batching, shard saving so Colab can resume.
- `src/model.py`, `src/train.py`: 3-layer MLP (1280-1024-512-terms) with BCE loss plus hierarchy loss (penalises child probability above parent).
- `src/metrics.py`, `src/evaluate.py`: protein-centric Fmax, micro/macro AUPR, hierarchy violation rate, with and without post-hoc hierarchy enforcement.
- `src/predict.py`: GO term prediction for new FASTA files.
- `src/make_processed_zip.py`: packs processed data for upload to Drive.
- `notebooks/protein_function_colab.ipynb`: runs embedding, training and evaluation on a Colab GPU.
- `docs/PIPELINE_GUIDE.md`: step-by-step guide on what to run where.
- Tested data prep, training and evaluation code on small synthetic data (embedding step could not be tested before the Colab run).

## Milestone 3 — Environment decisions
- Laptop has integrated graphics only, so no CUDA. Decision: laptop for data prep (and optionally training), Colab T4 for ESM-2 650M embeddings.

## Milestone 4 — First full run on Colab (done)
- Embeddings extracted, baseline (lambda=0) and ontology-aware (lambda=0.5) models trained and evaluated.
- Results (test split): see the table in the main README. Summary: Fmax about 0.94 to 0.95 for all three ontologies; hierarchy loss lowers violations from 0.0071 to 0.0048 with no real change in Fmax or AUPR.

### Observations and known issues
1. Fmax around 0.95 is far above typical published values, so leakage or easy labels are suspected (random split, IEA annotations, frequent ancestor terms). Needs validation before claiming performance.
2. The hierarchy-violation metric is computed over all terms at once, so the same value appears for MF, BP and CC. To be fixed.
3. Macro AUPR (0.68 to 0.90) is much lower than micro AUPR (0.96 to 0.99): rare terms are predicted much worse than common ones, which matches the class-imbalance problem named in the research gap.

## Next entries (to be filled as work progresses)
- [ ] Similarity-based split results
- [ ] Experimental-evidence-only results
- [ ] Lambda sweep and loss-curve plots
- [ ] Per-ontology violation metric fix
- [ ] Streamlit app
- [ ] Final report
