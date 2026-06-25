# latent qcbm placeholder
import numpy as np

from molaug_qcbm.encoding.autoencoder_latent import (
    AutoencoderLatentEncoder
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
    RANDOM_SEED,
    LATENT_DIM
)


class LatentQCBM:
    """
    Proposed Method

    Token Counts
            ↓
    Autoencoder
            ↓
    Latent Space
            ↓
    KMeans
            ↓
    QCBM

    Goal:
    Learn a more compact and meaningful
    molecular latent representation before
    quantum distribution learning.
    """

    def __init__(
        self,
        latent_dim=LATENT_DIM,
        n_clusters=N_CLUSTERS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        loss_metric=LOSS_METRIC,
        maxiter=MAXITER,
        random_state=RANDOM_SEED
    ):

        self.latent_dim = latent_dim

        self.n_clusters = n_clusters

        self.random_state = random_state

        self.autoencoder = (
            AutoencoderLatentEncoder(
                latent_dim=latent_dim
            )
        )

        self.kmeans = KMeansEncoder(
            n_clusters=n_clusters,
            random_state=random_state
        )

        self.qcbm = QCBM(
            n_qubits=n_qubits,
            n_layers=n_layers,
            loss_metric=loss_metric,
            maxiter=maxiter
        )

        self.latent_vectors = None

        self.cluster_ids = None

        self.target_distribution = None

        self.learned_distribution = None

        self.is_fitted = False

    def build_latent_space(
        self,
        token_count_vectors
    ):
        """
        Train autoencoder and create
        latent representations.
        """

        self.latent_vectors = (
            self.autoencoder.fit_transform(
                token_count_vectors
            )
        )

        return self.latent_vectors

    def cluster_latent_space(
        self
    ):
        """
        KMeans clustering on latent vectors.
        """

        if self.latent_vectors is None:

            raise RuntimeError(
                "Run build_latent_space() first."
            )

        self.cluster_ids = (
            self.kmeans.fit_predict(
                self.latent_vectors
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
        Train QCBM on latent distribution.
        """

        if self.target_distribution is None:

            raise RuntimeError(
                "Run cluster_latent_space() first."
            )

        self.qcbm.fit(
            self.target_distribution
        )

        self.learned_distribution = (
            self.qcbm.probabilities()
        )

        return self.learned_distribution

    def fit(
        self,
        token_count_vectors
    ):
        """
        Complete Latent-QCBM pipeline.
        """

        self.build_latent_space(
            token_count_vectors
        )

        self.cluster_latent_space()

        self.train_qcbm()

        self.is_fitted = True

        return self

    def probabilities(
        self
    ):
        """
        Return learned distribution.
        """

        self._check_fitted()

        return self.learned_distribution

    def sample_states(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample latent cluster states.
        """

        self._check_fitted()

        return self.qcbm.sample_states(
            n_samples=n_samples,
            random_state=random_state
        )

    def sample_bitstrings(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample bitstrings.
        """

        self._check_fitted()

        return self.qcbm.sample_bitstrings(
            n_samples=n_samples,
            random_state=random_state
        )

    def benchmark(
        self
    ):
        """
        Compare learned distribution
        against target distribution.
        """

        self._check_fitted()

        return distribution_report(
            "Latent+QCBM",
            self.learned_distribution,
            self.target_distribution
        )

    def reconstruction_error(
        self,
        token_count_vectors
    ):
        """
        Autoencoder reconstruction quality.
        """

        self._check_fitted()

        return (
            self.autoencoder
            .reconstruction_error(
                token_count_vectors
            )
        )

    def latent_statistics(
        self
    ):
        """
        Latent space diagnostics.
        """

        self._check_fitted()

        return {

            "n_samples":
                len(
                    self.latent_vectors
                ),

            "latent_dim":
                self.latent_vectors.shape[1],

            "mean":
                float(
                    np.mean(
                        self.latent_vectors
                    )
                ),

            "std":
                float(
                    np.std(
                        self.latent_vectors
                    )
                ),

            "min":
                float(
                    np.min(
                        self.latent_vectors
                    )
                ),

            "max":
                float(
                    np.max(
                        self.latent_vectors
                    )
                )
        }

    def cluster_statistics(
        self
    ):
        """
        Cluster population analysis.
        """

        self._check_fitted()

        unique, counts = np.unique(
            self.cluster_ids,
            return_counts=True
        )

        return {

            int(k): int(v)

            for k, v

            in zip(
                unique,
                counts
            )
        }

    def save_latent_vectors(
        self,
        filepath
    ):
        """
        Save latent vectors.
        """

        self._check_fitted()

        np.save(
            filepath,
            self.latent_vectors
        )

    def load_latent_vectors(
        self,
        filepath
    ):
        """
        Load precomputed latent vectors.
        """

        self.latent_vectors = np.load(
            filepath
        )

        return self.latent_vectors

    def summary(
        self
    ):
        """
        Model summary.
        """

        return {

            "method":
                "Latent+QCBM",

            "fitted":
                self.is_fitted,

            "latent_dim":
                self.latent_dim,

            "n_clusters":
                self.n_clusters,

            "distribution_states":
                (
                    len(
                        self.target_distribution
                    )
                    if self.target_distribution
                    is not None
                    else 0
                )
        }

    def _check_fitted(
        self
    ):
        if not self.is_fitted:

            raise RuntimeError(
                "Model has not been fitted."
            )