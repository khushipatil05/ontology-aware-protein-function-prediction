"""Central paths and hyper-parameters. Edit the three raw-file paths to match your laptop."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# ---- raw inputs (put / symlink the files in data/raw, or change these paths) ----
FASTA_PATH = RAW / "uniprot_sprot.fasta"        # dataset 1 (also works with .fasta.gz)
GAF_PATH = RAW / "goa_uniprot_all.gaf"          # dataset 2 (also works with .gaf.gz)
OBO_PATH = RAW / "go-basic.obo"                 # dataset 3 (see src/download_go.py)

# ---- preprocessing ----
MAX_SEQ_LEN = 1022          # ESM-2 limit is 1024 tokens incl. <cls>/<eos>
MIN_SEQ_LEN = 10
MIN_TERM_COUNT = 50         # keep GO terms with >= this many proteins (after propagation)
MAX_TERMS_PER_ASPECT = 1500 # cap per ontology (most frequent kept); None = no cap
KEEP_EVIDENCE = None        # None = ALL annotations. Or e.g. {"EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC"}
SEED = 42

# ---- embeddings ----
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"   # 1280-dim embeddings
EMB_DIM = 1280
TOKEN_BUDGET = 16000        # residues per batch; lower if you hit GPU OOM (try 8000)
SHARD_SIZE = 5000           # sequences per saved shard (allows resume after Colab disconnect)

# ---- classifier ----
HIDDEN = (1024, 512)        # 3-layer MLP: 1280 -> 1024 -> 512 -> n_terms
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
EPOCHS = 40
PATIENCE = 6
HIER_LAMBDA = 0.5           # weight of the GO-hierarchy consistency loss
