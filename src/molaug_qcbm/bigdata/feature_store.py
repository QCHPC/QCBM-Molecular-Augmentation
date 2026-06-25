# feature store placeholder
from pathlib import Path

import pandas as pd
import numpy as np

from molaug_qcbm.config import (
    EMBEDDINGS_DIR,
    CLUSTERED_DIR,
    PROCESSED_DATA_DIR
)


class FeatureStore:
    """
    Central storage layer for:

    - Processed molecules
    - SELFIES
    - Latent vectors
    - GNN embeddings
    - Cluster assignments
    - Target distributions

    Designed for 1M+ molecule datasets.
    """

    def __init__(
        self,
        root_dir=None
    ):

        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else PROCESSED_DATA_DIR
        )

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # DATAFRAMES
    # =====================================================

    def save_dataframe(
        self,
        df,
        filename
    ):
        """
        Save dataframe as parquet.
        """

        path = self.root_dir / filename

        df.to_parquet(
            path,
            index=False
        )

        return path

    def load_dataframe(
        self,
        filename
    ):
        """
        Load parquet dataframe.
        """

        path = self.root_dir / filename

        return pd.read_parquet(
            path
        )

    # =====================================================
    # NUMPY FEATURES
    # =====================================================

    def save_features(
        self,
        features,
        filename
    ):
        """
        Save feature matrix.
        """

        path = self.root_dir / filename

        np.save(
            path,
            features
        )

        return path

    def load_features(
        self,
        filename
    ):
        """
        Load feature matrix.
        """

        path = self.root_dir / filename

        return np.load(
            path,
            allow_pickle=True
        )

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def save_gnn_embeddings(
        self,
        embeddings,
        filename="gnn_embeddings.npy"
    ):

        path = (
            EMBEDDINGS_DIR /
            filename
        )

        np.save(
            path,
            embeddings
        )

        return path

    def load_gnn_embeddings(
        self,
        filename="gnn_embeddings.npy"
    ):

        path = (
            EMBEDDINGS_DIR /
            filename
        )

        return np.load(
            path,
            allow_pickle=True
        )

    def save_latent_vectors(
        self,
        vectors,
        filename="latent_vectors.npy"
    ):

        path = (
            EMBEDDINGS_DIR /
            filename
        )

        np.save(
            path,
            vectors
        )

        return path

    def load_latent_vectors(
        self,
        filename="latent_vectors.npy"
    ):

        path = (
            EMBEDDINGS_DIR /
            filename
        )

        return np.load(
            path,
            allow_pickle=True
        )

    # =====================================================
    # CLUSTERS
    # =====================================================

    def save_cluster_ids(
        self,
        cluster_ids,
        filename="cluster_ids.npy"
    ):

        path = (
            CLUSTERED_DIR /
            filename
        )

        np.save(
            path,
            cluster_ids
        )

        return path

    def load_cluster_ids(
        self,
        filename="cluster_ids.npy"
    ):

        path = (
            CLUSTERED_DIR /
            filename
        )

        return np.load(
            path,
            allow_pickle=True
        )

    def save_cluster_dataframe(
        self,
        smiles,
        cluster_ids,
        filename="cluster_mapping.parquet"
    ):
        """
        Save molecule-cluster mapping.
        """

        df = pd.DataFrame(
            {
                "SMILES":
                    smiles,

                "Cluster":
                    cluster_ids
            }
        )

        path = (
            CLUSTERED_DIR /
            filename
        )

        df.to_parquet(
            path,
            index=False
        )

        return path

    # =====================================================
    # DISTRIBUTIONS
    # =====================================================

    def save_distribution(
        self,
        distribution,
        filename="target_distribution.npy"
    ):

        path = (
            CLUSTERED_DIR /
            filename
        )

        np.save(
            path,
            distribution
        )

        return path

    def load_distribution(
        self,
        filename="target_distribution.npy"
    ):

        path = (
            CLUSTERED_DIR /
            filename
        )

        return np.load(
            path,
            allow_pickle=True
        )

    # =====================================================
    # CACHE CHECKS
    # =====================================================

    def exists(
        self,
        directory,
        filename
    ):
        """
        Check if file exists.
        """

        path = (
            Path(directory)
            /
            filename
        )

        return path.exists()

    def embedding_exists(
        self,
        filename="gnn_embeddings.npy"
    ):

        return (
            EMBEDDINGS_DIR /
            filename
        ).exists()

    def latent_exists(
        self,
        filename="latent_vectors.npy"
    ):

        return (
            EMBEDDINGS_DIR /
            filename
        ).exists()

    def clusters_exist(
        self,
        filename="cluster_ids.npy"
    ):

        return (
            CLUSTERED_DIR /
            filename
        ).exists()

    # =====================================================
    # METADATA
    # =====================================================

    def summary(
        self
    ):

        return {

            "processed_dir":
                str(
                    PROCESSED_DATA_DIR
                ),

            "embeddings_dir":
                str(
                    EMBEDDINGS_DIR
                ),

            "clustered_dir":
                str(
                    CLUSTERED_DIR
                )
        }