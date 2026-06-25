# augmentation retrieval placeholder
import numpy as np
import pandas as pd

from collections import defaultdict


class EmpiricalClusterRetriever:
    """
    Retrieve molecules from existing clusters.

    Used by:

    KMeans + QCBM
    ChowLiu + QCBM
    RL + QCBM

    Pipeline:

    cluster_id
        ↓
    retrieve molecule
    """

    def __init__(
        self,
        smiles_list,
        cluster_labels
    ):

        self.smiles_list = list(smiles_list)

        self.cluster_labels = np.asarray(
            cluster_labels
        )

        self.cluster_to_molecules = (
            defaultdict(list)
        )

        self._build_index()

    def _build_index(self):

        for smiles, cluster in zip(
            self.smiles_list,
            self.cluster_labels
        ):

            self.cluster_to_molecules[
                int(cluster)
            ].append(smiles)

    def cluster_size(
        self,
        cluster_id
    ):

        return len(
            self.cluster_to_molecules.get(
                int(cluster_id),
                []
            )
        )

    def available_clusters(self):

        return sorted(
            self.cluster_to_molecules.keys()
        )

    def retrieve_single(
        self,
        cluster_id,
        random_state=None
    ):

        molecules = (
            self.cluster_to_molecules.get(
                int(cluster_id),
                []
            )
        )

        if len(molecules) == 0:
            return None

        rng = np.random.default_rng(
            random_state
        )

        return rng.choice(
            molecules
        )

    def retrieve_batch(
        self,
        cluster_ids,
        random_state=None
    ):

        rng = np.random.default_rng(
            random_state
        )

        generated = []

        for cluster in cluster_ids:

            molecules = (
                self.cluster_to_molecules.get(
                    int(cluster),
                    []
                )
            )

            if len(molecules) == 0:
                continue

            generated.append(
                rng.choice(molecules)
            )

        return generated

    def retrieve_from_distribution(
        self,
        distribution,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample clusters according
        to learned distribution.

        distribution length =
        number of clusters
        """

        rng = np.random.default_rng(
            random_state
        )

        distribution = np.asarray(
            distribution,
            dtype=float
        )

        distribution = (
            distribution /
            distribution.sum()
        )

        cluster_ids = np.arange(
            len(distribution)
        )

        sampled_clusters = (
            rng.choice(
                cluster_ids,
                size=n_samples,
                p=distribution
            )
        )

        molecules = (
            self.retrieve_batch(
                sampled_clusters,
                random_state=random_state
            )
        )

        return molecules

    def retrieve_with_counts(
        self,
        cluster_counts,
        random_state=None
    ):
        """
        cluster_counts:

        {
            0:100,
            1:50,
            2:25
        }
        """

        rng = np.random.default_rng(
            random_state
        )

        generated = []

        for cluster_id, count in (
            cluster_counts.items()
        ):

            molecules = (
                self.cluster_to_molecules.get(
                    int(cluster_id),
                    []
                )
            )

            if len(molecules) == 0:
                continue

            for _ in range(count):

                generated.append(
                    rng.choice(molecules)
                )

        return generated

    def retrieval_report(self):

        rows = []

        for cluster in (
            self.available_clusters()
        ):

            rows.append(
                {
                    "cluster": cluster,
                    "count": self.cluster_size(
                        cluster
                    )
                }
            )

        return pd.DataFrame(rows)