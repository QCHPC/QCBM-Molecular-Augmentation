# augmentation generator placeholder
import numpy as np

from molaug_qcbm.augmentation.retrieval import (
    EmpiricalClusterRetriever
)


class MolecularGenerator:
    """
    Unified molecule generation interface.

    Supports:

    - QCBM
    - Classical Born
    - Markov Born
    - QGAN
    - QVAE
    - RL-QCBM
    """

    def __init__(
        self,
        smiles_list,
        cluster_labels
    ):

        self.retriever = (
            EmpiricalClusterRetriever(
                smiles_list=smiles_list,
                cluster_labels=cluster_labels
            )
        )

    def generate_from_distribution(
        self,
        probabilities,
        n_samples=1000,
        random_state=42
    ):
        """
        Generate molecules from
        learned probability distribution.
        """

        return self.retriever.retrieve_from_distribution(
            distribution=probabilities,
            n_samples=n_samples,
            random_state=random_state
        )

    def generate_from_counts(
        self,
        cluster_counts,
        random_state=42
    ):
        """
        Generate molecules from
        explicit cluster counts.

        Example:

        {
            0:100,
            1:50,
            2:25
        }
        """

        return self.retriever.retrieve_with_counts(
            cluster_counts=cluster_counts,
            random_state=random_state
        )

    def generate_from_sampled_states(
        self,
        sampled_states,
        random_state=42
    ):
        """
        Input:

        [0,1,5,7,7,7,12,...]

        Useful when a model directly
        returns sampled cluster IDs.
        """

        return self.retriever.retrieve_batch(
            cluster_ids=sampled_states,
            random_state=random_state
        )

    def generate_dataframe(
        self,
        generated_smiles
    ):

        import pandas as pd

        return pd.DataFrame(
            {
                "SMILES": generated_smiles
            }
        )

    def generation_summary(
        self,
        generated_smiles
    ):

        unique_smiles = len(
            set(generated_smiles)
        )

        total_smiles = len(
            generated_smiles
        )

        uniqueness = (
            unique_smiles /
            total_smiles
            if total_smiles > 0
            else 0
        )

        return {
            "generated": total_smiles,
            "unique": unique_smiles,
            "uniqueness": uniqueness
        }