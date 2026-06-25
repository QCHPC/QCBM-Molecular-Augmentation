# chowliu qcbm placeholder
import numpy as np

from molaug_qcbm.encoding.chow_liu import (
    ChowLiuEncoder
)

from molaug_qcbm.encoding.kmeans import (
    KMeansEncoder
)

from molaug_qcbm.qcbm.qcbm import (
    QCBM
)

from molaug_qcbm.evaluation.benchmark_metrics import (
    distribution_report
)

from molaug_qcbm.config import (
    N_CLUSTERS,
    N_QUBITS,
    N_LAYERS,
    LOSS_METRIC,
    MAXITER,
    RANDOM_SEED
)


class ChowLiuQCBM:
    """
    Proposed Method:

    Chow-Liu Tree
        +
    KMeans
        +
    QCBM

    Captures dependencies between
    SELFIES token dimensions before
    quantum distribution learning.
    """

    def __init__(
        self,
        n_clusters=N_CLUSTERS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        loss_metric=LOSS_METRIC,
        maxiter=MAXITER
    ):

        self.n_clusters = n_clusters

        self.chowliu = ChowLiuEncoder()

        self.kmeans = KMeansEncoder(
            n_clusters=n_clusters,
            random_state=RANDOM_SEED
        )

        self.qcbm = QCBM(
            n_qubits=n_qubits,
            n_layers=n_layers,
            loss_metric=loss_metric,
            maxiter=maxiter
        )

        self.cluster_ids = None

        self.target_distribution = None

        self.learned_distribution = None

    def build_features(
        self,
        latent_vectors
    ):
        """
        Apply Chow-Liu encoding.
        """

        return (
            self.chowliu.fit_transform(
                latent_vectors
            )
        )

    def cluster(
        self,
        chowliu_features
    ):
        """
        Cluster enhanced features.
        """

        self.cluster_ids = (
            self.kmeans.fit_predict(
                chowliu_features
            )
        )

        self.target_distribution = (
            self.kmeans.target_distribution(
                self.cluster_ids,
                self.n_clusters
            )
        )

        return (
            self.cluster_ids,
            self.target_distribution
        )

    def train_qcbm(
        self
    ):
        """
        Train QCBM.
        """

        self.qcbm.fit(
            self.target_distribution
        )

        self.learned_distribution = (
            self.qcbm.probabilities()
        )

        return self.learned_distribution

    def fit(
        self,
        latent_vectors
    ):
        """
        Full training pipeline.
        """

        chowliu_features = (
            self.build_features(
                latent_vectors
            )
        )

        self.cluster(
            chowliu_features
        )

        self.train_qcbm()

        return self

    def sample_states(
        self,
        n_samples=1000
    ):
        """
        Sample from trained QCBM.
        """

        return (
            self.qcbm.sample_states(
                n_samples=n_samples
            )
        )

    def sample_bitstrings(
        self,
        n_samples=1000
    ):
        """
        Sample bitstrings.
        """

        return (
            self.qcbm.sample_bitstrings(
                n_samples=n_samples
            )
        )

    def benchmark(
        self
    ):
        """
        Compare learned distribution
        against target distribution.
        """

        return distribution_report(
            "ChowLiu+QCBM",
            self.learned_distribution,
            self.target_distribution
        )

    def chowliu_edges(
        self
    ):
        """
        Return learned dependency edges.
        """

        return (
            self.chowliu.get_edges()
        )

    def summary(
        self
    ):

        return {

            "method":
                "ChowLiu+QCBM",

            "clusters":
                self.n_clusters,

            "tree_edges":
                len(
                    self.chowliu.get_edges()
                ),

            "distribution_states":
                len(
                    self.target_distribution
                )
                if self.target_distribution is not None
                else 0
        }