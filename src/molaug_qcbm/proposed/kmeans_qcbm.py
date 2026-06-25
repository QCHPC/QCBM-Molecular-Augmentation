# kmeans qcbm placeholder
import numpy as np

from molaug_qcbm.encoding.kmeans import (
    KMeansEncoder
)

from molaug_qcbm.qcbm.qcbm import (
    QCBM
)

from molaug_qcbm.qcbm.sampler import (
    QCBMSampler
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


class KMeansQCBM:
    """
    Proposed baseline method:

    KMeans + QCBM

    Pipeline:
        latent vectors
            ↓
        KMeans clustering
            ↓
        target probability distribution
            ↓
        QCBM training
            ↓
        sampled molecular states
    """

    def __init__(
        self,
        n_clusters=N_CLUSTERS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        loss_metric=LOSS_METRIC,
        maxiter=MAXITER,
        random_state=RANDOM_SEED
    ):
        self.n_clusters = n_clusters
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.loss_metric = loss_metric
        self.maxiter = maxiter
        self.random_state = random_state

        if 2 ** self.n_qubits != self.n_clusters:
            raise ValueError(
                "n_clusters must equal 2 ** n_qubits for QCBM state mapping."
            )

        self.kmeans = KMeansEncoder(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )

        self.qcbm = QCBM(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            loss_metric=self.loss_metric,
            maxiter=self.maxiter
        )

        self.cluster_ids = None
        self.target_distribution = None
        self.learned_distribution = None
        self.is_fitted = False

    def fit_clusters(
        self,
        latent_vectors
    ):
        """
        Fit KMeans and build target distribution.
        """

        self.cluster_ids = self.kmeans.fit_predict(
            latent_vectors
        )

        self.target_distribution = self.kmeans.target_distribution(
            self.cluster_ids,
            self.n_clusters
        )

        return self.cluster_ids, self.target_distribution

    def fit_qcbm(
        self
    ):
        """
        Train QCBM on KMeans target distribution.
        """

        if self.target_distribution is None:
            raise RuntimeError(
                "Run fit_clusters() before fit_qcbm()."
            )

        self.qcbm.fit(
            self.target_distribution
        )

        self.learned_distribution = self.qcbm.probabilities()

        return self.learned_distribution

    def fit(
        self,
        latent_vectors
    ):
        """
        Full KMeans + QCBM pipeline.
        """

        self.fit_clusters(
            latent_vectors
        )

        self.fit_qcbm()

        self.is_fitted = True

        return self

    def predict_clusters(
        self,
        latent_vectors
    ):
        """
        Assign new molecules to existing KMeans clusters.
        """

        return self.kmeans.predict(
            latent_vectors
        )

    def probabilities(
        self
    ):
        """
        Return learned QCBM distribution.
        """

        self._check_fitted()

        return self.learned_distribution

    def sample_states(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample integer molecular states from QCBM.
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
        Sample bitstrings from QCBM.
        """

        self._check_fitted()

        return self.qcbm.sample_bitstrings(
            n_samples=n_samples,
            random_state=random_state
        )

    def empirical_distribution(
        self,
        n_samples=5000,
        random_state=None
    ):
        """
        Empirical distribution from sampled QCBM states.
        """

        self._check_fitted()

        sampler = QCBMSampler(
            self.learned_distribution
        )

        return sampler.empirical_distribution(
            n_samples=n_samples,
            random_state=random_state
        )

    def benchmark(
        self,
        use_empirical=False,
        n_samples=5000,
        random_state=None
    ):
        """
        Compare QCBM learned distribution against target distribution.
        """

        self._check_fitted()

        if use_empirical:
            model_distribution = self.empirical_distribution(
                n_samples=n_samples,
                random_state=random_state
            )
            name = "KMeans+QCBM_Sampled"
        else:
            model_distribution = self.learned_distribution
            name = "KMeans+QCBM_Statevector"

        return distribution_report(
            name,
            model_distribution,
            self.target_distribution
        )

    def top_states(
        self,
        top_k=10
    ):
        """
        Return most probable QCBM states.
        """

        self._check_fitted()

        probs = np.asarray(
            self.learned_distribution,
            dtype=float
        )

        order = np.argsort(
            probs
        )[::-1][:top_k]

        return [
            {
                "state": int(state),
                "bitstring": format(
                    int(state),
                    f"0{self.n_qubits}b"
                ),
                "probability": float(
                    probs[state]
                )
            }
            for state in order
        ]

    def summary(
        self
    ):
        """
        Compact model summary.
        """

        return {
            "method": "KMeans+QCBM",
            "fitted": self.is_fitted,
            "n_clusters": self.n_clusters,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "loss_metric": self.loss_metric,
            "target_states": (
                len(self.target_distribution)
                if self.target_distribution is not None
                else None
            ),
            "nonzero_target_states": (
                int(np.sum(self.target_distribution > 0))
                if self.target_distribution is not None
                else None
            )
        }

    def _check_fitted(
        self
    ):
        if not self.is_fitted:
            raise RuntimeError(
                "KMeansQCBM must be fitted first."
            )