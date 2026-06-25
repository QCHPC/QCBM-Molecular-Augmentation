# gnn qcbm placeholder
import numpy as np

from molaug_qcbm.encoding.gnn_embedding import (
    GNNEmbeddingEncoder
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


class GNNQCBM:
    """
    Proposed Method:

    Molecular Graph
            ↓
    GNN Embeddings
            ↓
    KMeans
            ↓
    QCBM
            ↓
    Molecular State Generation

    Research Goal:

    Compare:

    SELFIES + KMeans + QCBM

    versus

    Graph + GNN + KMeans + QCBM
    """

    def __init__(
        self,
        n_clusters=N_CLUSTERS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        loss_metric=LOSS_METRIC,
        maxiter=MAXITER,
        embedding_dim=32,
        hidden_dim=64
    ):

        self.n_clusters = n_clusters

        self.gnn = GNNEmbeddingEncoder(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim
        )

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

        self.embeddings = None

        self.cluster_ids = None

        self.target_distribution = None

        self.learned_distribution = None

        self.smiles_list = None

        self.is_fitted = False

    def build_embeddings(
        self,
        smiles_list
    ):
        """
        Generate GNN embeddings.
        """

        self.smiles_list = smiles_list

        self.embeddings = (
            self.gnn.fit_transform(
                smiles_list
            )
        )

        return self.embeddings

    def cluster_embeddings(
        self
    ):
        """
        KMeans clustering on graph embeddings.
        """

        if self.embeddings is None:

            raise RuntimeError(
                "Run build_embeddings() first."
            )

        self.cluster_ids = (
            self.kmeans.fit_predict(
                self.embeddings
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

        if self.target_distribution is None:

            raise RuntimeError(
                "Run cluster_embeddings() first."
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
        smiles_list
    ):
        """
        Complete GNN-QCBM pipeline.
        """

        self.build_embeddings(
            smiles_list
        )

        self.cluster_embeddings()

        self.train_qcbm()

        self.is_fitted = True

        return self

    def probabilities(
        self
    ):
        """
        Learned distribution.
        """

        self._check_fitted()

        return self.learned_distribution

    def sample_states(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample cluster states.
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

    def top_states(
        self,
        top_k=10
    ):
        """
        Most probable quantum states.
        """

        self._check_fitted()

        probs = np.asarray(
            self.learned_distribution
        )

        order = np.argsort(
            probs
        )[::-1][:top_k]

        results = []

        for idx in order:

            results.append(
                {
                    "state":
                        int(idx),

                    "bitstring":
                        format(
                            int(idx),
                            f"0{self.qcbm.n_qubits}b"
                        ),

                    "probability":
                        float(
                            probs[idx]
                        )
                }
            )

        return results

    def benchmark(
        self
    ):
        """
        Distribution quality benchmark.
        """

        self._check_fitted()

        return distribution_report(
            "GNN+QCBM",
            self.learned_distribution,
            self.target_distribution
        )

    def embedding_statistics(
        self
    ):
        """
        Embedding diagnostics.
        """

        self._check_fitted()

        return {

            "n_molecules":
                len(
                    self.embeddings
                ),

            "embedding_dim":
                self.embeddings.shape[1],

            "mean":
                float(
                    np.mean(
                        self.embeddings
                    )
                ),

            "std":
                float(
                    np.std(
                        self.embeddings
                    )
                )
        }

    def cluster_statistics(
        self
    ):
        """
        Cluster diagnostics.
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

    def summary(
        self
    ):
        """
        Model summary.
        """

        return {

            "method":
                "GNN+QCBM",

            "fitted":
                self.is_fitted,

            "n_clusters":
                self.n_clusters,

            "embedding_dim":
                self.gnn.embedding_dim,

            "hidden_dim":
                self.gnn.hidden_dim,

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

    def save_embeddings(
        self,
        filepath
    ):
        """
        Save embeddings.
        """

        self._check_fitted()

        np.save(
            filepath,
            self.embeddings
        )

    def load_embeddings(
        self,
        filepath
    ):
        """
        Load precomputed embeddings.
        """

        self.embeddings = np.load(
            filepath
        )

        return self.embeddings

    def _check_fitted(
        self
    ):
        if not self.is_fitted:

            raise RuntimeError(
                "Model has not been fitted."
            )