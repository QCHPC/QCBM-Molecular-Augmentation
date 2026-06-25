# augmentation reconstruction placeholder
import random

import pandas as pd
import numpy as np


class MoleculeReconstructor:
    """
    Reconstruction layer for molecular augmentation.

    Pipeline

    QCBM State
        ↓
    Cluster
        ↓
    Candidate Molecules
        ↓
    Reconstruction
    """

    def __init__(
        self,
        random_state=42
    ):
        self.random_state = random_state

        random.seed(
            random_state
        )

    def state_to_molecules(
        self,
        state,
        cluster_ids,
        smiles_list
    ):
        """
        Retrieve all molecules belonging
        to a sampled state.
        """

        molecules = [

            smiles

            for smiles, cluster

            in zip(
                smiles_list,
                cluster_ids
            )

            if cluster == state
        ]

        return molecules

    def reconstruct_single(
        self,
        state,
        cluster_ids,
        smiles_list
    ):
        """
        Return one representative molecule.
        """

        candidates = (
            self.state_to_molecules(
                state,
                cluster_ids,
                smiles_list
            )
        )

        if len(candidates) == 0:

            return None

        return random.choice(
            candidates
        )

    def reconstruct_multiple(
        self,
        states,
        cluster_ids,
        smiles_list
    ):
        """
        Reconstruct many sampled states.
        """

        molecules = []

        for state in states:

            mol = (
                self.reconstruct_single(
                    state,
                    cluster_ids,
                    smiles_list
                )
            )

            if mol is not None:

                molecules.append(
                    mol
                )

        return molecules

    def reconstruct_topk(
        self,
        state,
        cluster_ids,
        smiles_list,
        k=10
    ):
        """
        Return up to k molecules
        from a cluster.
        """

        candidates = (
            self.state_to_molecules(
                state,
                cluster_ids,
                smiles_list
            )
        )

        return candidates[:k]

    def cluster_size(
        self,
        state,
        cluster_ids
    ):
        """
        Number of molecules in cluster.
        """

        return int(
            np.sum(
                np.array(
                    cluster_ids
                ) == state
            )
        )

    def cluster_statistics(
        self,
        cluster_ids
    ):
        """
        Cluster population report.
        """

        unique, counts = np.unique(
            cluster_ids,
            return_counts=True
        )

        return pd.DataFrame(
            {
                "Cluster":
                    unique,

                "Count":
                    counts
            }
        )

    def reconstruction_report(
        self,
        states,
        cluster_ids,
        smiles_list
    ):
        """
        Report reconstructed molecules.
        """

        rows = []

        for state in states:

            mol = (
                self.reconstruct_single(
                    state,
                    cluster_ids,
                    smiles_list
                )
            )

            rows.append(
                {
                    "State":
                        int(state),

                    "Cluster_Size":
                        self.cluster_size(
                            state,
                            cluster_ids
                        ),

                    "Molecule":
                        mol
                }
            )

        return pd.DataFrame(
            rows
        )

    def sample_from_cluster(
        self,
        state,
        cluster_ids,
        smiles_list,
        n_samples=10
    ):
        """
        Randomly sample molecules
        from a cluster.
        """

        candidates = (
            self.state_to_molecules(
                state,
                cluster_ids,
                smiles_list
            )
        )

        if len(candidates) == 0:

            return []

        n_samples = min(
            n_samples,
            len(candidates)
        )

        return random.sample(
            candidates,
            n_samples
        )

    def reconstruct_distribution(
        self,
        sampled_states,
        cluster_ids,
        smiles_list
    ):
        """
        Reconstruct an entire generated
        molecular distribution.
        """

        generated = []

        for state in sampled_states:

            molecule = (
                self.reconstruct_single(
                    state,
                    cluster_ids,
                    smiles_list
                )
            )

            if molecule is not None:

                generated.append(
                    molecule
                )

        return generated