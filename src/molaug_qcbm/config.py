from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CLUSTERED_DIR = DATA_DIR / "clustered"
BENCHMARK_SPLITS_DIR = DATA_DIR / "benchmark_splits"

RESULTS_DIR = PROJECT_ROOT / "results"

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

MODELS_DIR = PROJECT_ROOT / "models"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DOCS_DIR = PROJECT_ROOT / "docs"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

PRIMARY_DATASET = (
    RAW_DATA_DIR /
    "1Mstoned_vsc_initial_dataset_insilico_chemistry42_filtered.csv"
)

KNOWN_INHIBITORS_DATASET = (
    RAW_DATA_DIR /
    "650known_inhibitors_smiles_and_selfies.csv"
)

BENCHMARK_DATASET = (
    RAW_DATA_DIR /
    "KRAS_G12D_inhibitors_update202209_updated.csv"
)

SMILES_COLUMN = "SMILES"

SELFIES_COLUMN = "SELFIES"

RANDOM_SEED = 42


# ============================================================
# DATASET SIZE CONFIGURATION
# ============================================================

USE_FULL_DATASET = True

MAX_ROWS = None

TRAIN_FRACTION = 0.80

VALIDATION_FRACTION = 0.10

TEST_FRACTION = 0.10


# ============================================================
# ENCODING CONFIGURATION
# ============================================================

MAX_SELFIES_TOKENS = 32

N_CLUSTERS = 32

KMEANS_RANDOM_STATE = 42

NORMALIZE_LATENT_VECTORS = True


# ============================================================
# QCBM CONFIGURATION
# ============================================================

N_QUBITS = 5

N_STATES = 2 ** N_QUBITS

N_LAYERS = 3

ENTANGLEMENT = "linear"

LOSS_METRIC = "mmd"

OPTIMIZER = "COBYLA"

MAXITER = 200

INITIALIZATION = "random"


# ============================================================
# IBM QUANTUM CONFIGURATION
# ============================================================

IBM_BACKEND = "ibm_fez"

SHOTS = 4096

OPTIMIZATION_LEVEL = 1

RESILIENCE_LEVEL = 1

USE_HARDWARE = True

USE_STATEVECTOR = True


# ============================================================
# BIG DATA CONFIGURATION
# ============================================================

CHUNK_SIZE = 50_000

MINIBATCH_SIZE = 5_000

PARQUET_BATCH_SIZE = 100_000

USE_PARQUET_CACHE = True


# ============================================================
# MOLECULAR GENERATION CONFIGURATION
# ============================================================

N_GENERATED_MOLECULES = 1000

RETRIEVAL_RANDOM_STATE = 42

ENABLE_RETRIEVAL = True


# ============================================================
# STONED BASELINE CONFIGURATION
# ============================================================

STONED_MUTATION_DEPTH = 3

STONED_NUM_SAMPLES = 1000


# ============================================================
# CLASSICAL BORN CONFIGURATION
# ============================================================

CLASSICAL_BORN_NUM_SAMPLES = 5000

MARKOV_BORN_NUM_SAMPLES = 5000


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

TOP_K_STATES = 10

SYBA_ENABLED = True

QED_ENABLED = True

SA_SCORE_ENABLED = True

VALIDITY_ENABLED = True

UNIQUENESS_ENABLED = True

NOVELTY_ENABLED = True


# ============================================================
# CHOW-LIU CONFIGURATION
# ============================================================

CHOW_LIU_ENABLED = False

MUTUAL_INFORMATION_THRESHOLD = 0.01


# ============================================================
# GMM ENCODING CONFIGURATION
# ============================================================

GMM_ENABLED = False

N_GMM_COMPONENTS = 32


# ============================================================
# SCAFFOLD ENCODING CONFIGURATION
# ============================================================

SCAFFOLD_ENCODING_ENABLED = False


# ============================================================
# FRAGMENT ENCODING CONFIGURATION
# ============================================================

FRAGMENT_ENCODING_ENABLED = False


# ============================================================
# GNN CONFIGURATION
# ============================================================

GNN_ENABLED = False

NODE_FEATURE_DIM = 128

EDGE_FEATURE_DIM = 32

GNN_EMBEDDING_DIM = 64


# ============================================================
# AUTOENCODER CONFIGURATION
# ============================================================

AUTOENCODER_ENABLED = False

LATENT_DIM = 32

AE_EPOCHS = 50

AE_BATCH_SIZE = 256


# ============================================================
# RL-QCBM CONFIGURATION
# ============================================================

RL_ENABLED = False

REWARD_WEIGHT_SYBA = 1.0

REWARD_WEIGHT_QED = 1.0

REWARD_WEIGHT_NOVELTY = 1.0


# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================

RUN_CLASSICAL_BORN = True

RUN_MARKOV_BORN = True

RUN_STONED = True

RUN_QCBM = True

RUN_HARDWARE_QCBM = True

RUN_CHOWLIU_QCBM = False

RUN_GNN_QCBM = False

RUN_LATENT_QCBM = False

RUN_RL_QCBM = False


# ============================================================
# OUTPUT FILES
# ============================================================

LATENT_VECTORS_FILE = (
    EMBEDDINGS_DIR /
    "latent_vectors.npy"
)

VOCAB_FILE = (
    EMBEDDINGS_DIR /
    "token_vocabulary.pkl"
)

CLUSTER_ASSIGNMENTS_FILE = (
    CLUSTERED_DIR /
    "cluster_assignments.csv"
)

TARGET_DISTRIBUTION_FILE = (
    CLUSTERED_DIR /
    "target_distribution.npy"
)

QCBM_RESULTS_FILE = (
    TABLES_DIR /
    "qcbm_results.csv"
)

BENCHMARK_RESULTS_FILE = (
    TABLES_DIR /
    "benchmark_results.csv"
)

GENERATED_MOLECULES_FILE = (
    PROCESSED_DATA_DIR /
    "generated_molecules.csv"
)


# ============================================================
# DIRECTORY CREATION
# ============================================================

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DIR,
    CLUSTERED_DIR,
    BENCHMARK_SPLITS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    DOCS_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONFIG SUMMARY
# ============================================================

def print_config_summary():

    print("\n========== QCBM Molecular Augmentation ==========\n")

    print(f"Primary Dataset      : {PRIMARY_DATASET.name}")
    print(f"Benchmark Dataset    : {BENCHMARK_DATASET.name}")
    print(f"Known Inhibitors     : {KNOWN_INHIBITORS_DATASET.name}")

    print(f"\nClusters             : {N_CLUSTERS}")
    print(f"Qubits               : {N_QUBITS}")
    print(f"States               : {N_STATES}")
    print(f"Layers               : {N_LAYERS}")

    print(f"\nBackend              : {IBM_BACKEND}")
    print(f"Shots                : {SHOTS}")

    print(f"\nLoss Metric          : {LOSS_METRIC}")
    print(f"Optimizer            : {OPTIMIZER}")

    print(f"\nGenerated Molecules  : {N_GENERATED_MOLECULES}")

    print("\n================================================\n")


if __name__ == "__main__":
    print_config_summary()