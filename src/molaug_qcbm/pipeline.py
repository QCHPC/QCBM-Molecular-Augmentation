# Placeholder pipeline module
import pandas as pd

from molaug_qcbm.config import (
    PRIMARY_DATASET,
    SMILES_COLUMN,
    N_CLUSTERS,
    N_QUBITS,
    N_LAYERS,
    MAXITER,
    LOSS_METRIC,
    RANDOM_SEED
)

from molaug_qcbm.preprocessing.selfies import (
    safe_selfies
)

from molaug_qcbm.encoding.kmeans import (
    selfies_to_latent,
    KMeansEncoder
)

from molaug_qcbm.qcbm.qcbm import (
    QCBM
)

from molaug_qcbm.augmentation.generator import (
    MolecularGenerator
)

from molaug_qcbm.evaluation.benchmark_metrics import (
    distribution_report
)


class MolecularAugmentationPipeline:
    """
    End-to-end KMeans + QCBM pipeline.

    Dataset
        ↓
    SELFIES
        ↓
    Latent Vectors
        ↓
    KMeans
        ↓
    Target Distribution
        ↓
    QCBM
        ↓
    Generated Molecules
    """

    def __init__(
        self,
        dataset_path=PRIMARY_DATASET
    ):

        self.dataset_path = dataset_path

        self.df = None

        self.smiles_list = None

        self.selfies_list = None

        self.latent_vectors = None

        self.cluster_ids = None

        self.target_distribution = None

        self.qcbm = None

        self.generated_smiles = None

    def load_dataset(
        self
    ):

        self.df = pd.read_csv(
            self.dataset_path
        )

        self.smiles_list = (
            self.df[
                SMILES_COLUMN
            ]
            .dropna()
            .tolist()
        )

        print(
            f"Loaded {len(self.smiles_list)} molecules"
        )

        return self

    def smiles_to_selfies(
        self
    ):

        selfies_list = []

        for smi in self.smiles_list:

            selfie = safe_selfies(
                smi
            )

            if selfie is not None:

                selfies_list.append(
                    selfie
                )

        self.selfies_list = selfies_list

        print(
            f"Converted {len(self.selfies_list)} SELFIES"
        )

        return self

    def build_latent_space(
        self
    ):

        self.latent_vectors, self.vocab = (
            selfies_to_latent(
                self.selfies_list
            )
        )

        print(
            f"Latent Shape: {self.latent_vectors.shape}"
        )

        return self

    def cluster_molecules(
        self
    ):

        encoder = KMeansEncoder(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_SEED
        )

        self.cluster_ids = (
            encoder.fit_predict(
                self.latent_vectors
            )
        )

        self.target_distribution = (
            encoder.target_distribution(
                self.cluster_ids,
                N_CLUSTERS
            )
        )

        print(
            f"Created {N_CLUSTERS} clusters"
        )

        return self

    def train_qcbm(
        self
    ):

        self.qcbm = QCBM(
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            loss_metric=LOSS_METRIC,
            maxiter=MAXITER
        )

        self.qcbm.fit(
            self.target_distribution
        )

        print(
            "QCBM Training Complete"
        )

        return self

    def generate(
        self,
        n_samples=1000
    ):

        learned_distribution = (
            self.qcbm.probabilities()
        )

        generator = (
            MolecularGenerator(
                smiles_list=self.smiles_list,
                cluster_labels=self.cluster_ids
            )
        )

        self.generated_smiles = (
            generator.generate_from_distribution(
                learned_distribution,
                n_samples=n_samples
            )
        )

        print(
            f"Generated {len(self.generated_smiles)} molecules"
        )

        return self.generated_smiles

    def evaluate(
        self
    ):

        learned_distribution = (
            self.qcbm.probabilities()
        )

        report = distribution_report(
            "QCBM",
            learned_distribution,
            self.target_distribution
        )

        return report

    def run(
        self,
        n_samples=1000
    ):

        (
            self.load_dataset()
            .smiles_to_selfies()
            .build_latent_space()
            .cluster_molecules()
            .train_qcbm()
        )

        generated = self.generate(
            n_samples=n_samples
        )

        report = self.evaluate()

        return {
            "generated_smiles":
                generated,
            "metrics":
                report
        }