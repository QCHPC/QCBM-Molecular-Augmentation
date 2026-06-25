# rl qcbm placeholder
import numpy as np
import pandas as pd

from molaug_qcbm.proposed.kmeans_qcbm import (
    KMeansQCBM
)

from molaug_qcbm.evaluation.syba import (
    syba_score_smiles
)

from molaug_qcbm.evaluation.qed import (
    qed_score
)

from molaug_qcbm.evaluation.novelty import (
    novelty_score
)

from molaug_qcbm.augmentation.reconstruction import (
    MoleculeReconstructor
)

from molaug_qcbm.config import (
    REWARD_WEIGHT_SYBA,
    REWARD_WEIGHT_QED,
    REWARD_WEIGHT_NOVELTY
)


class RLQCBM:
    """
    RL + QCBM

    Pipeline

    SMILES
        ↓
    KMeans
        ↓
    QCBM
        ↓
    Sample States
        ↓
    Reconstruct Molecules
        ↓
    Reward Evaluation
        ↓
    Reinforced Sampling

    Reward:

        SYBA
        + QED
        + Novelty
    """

    def __init__(
        self,
        reward_weight_syba=REWARD_WEIGHT_SYBA,
        reward_weight_qed=REWARD_WEIGHT_QED,
        reward_weight_novelty=REWARD_WEIGHT_NOVELTY
    ):

        self.reward_weight_syba = (
            reward_weight_syba
        )

        self.reward_weight_qed = (
            reward_weight_qed
        )

        self.reward_weight_novelty = (
            reward_weight_novelty
        )

        self.qcbm_model = None

        self.reconstructor = (
            MoleculeReconstructor()
        )

        self.training_smiles = None

        self.cluster_ids = None

        self.reward_table = None

        self.generated_smiles = None

    def reward(
        self,
        smiles,
        training_smiles=None
    ):
        """
        RL reward function.
        """

        syba = (
            syba_score_smiles(
                smiles
            )
            or 0.0
        )

        qed = (
            qed_score(
                smiles
            )
            or 0.0
        )

        novelty = 0.0

        if training_smiles is not None:

            novelty = (
                novelty_score(
                    smiles,
                    training_smiles
                )
                or 0.0
            )

        reward = (

            self.reward_weight_syba
            * syba

            +

            self.reward_weight_qed
            * qed

            +

            self.reward_weight_novelty
            * novelty
        )

        return float(
            reward
        )

    def fit(
        self,
        latent_vectors,
        smiles_list
    ):
        """
        Train KMeans+QCBM.
        """

        self.training_smiles = (
            smiles_list
        )

        self.qcbm_model = (
            KMeansQCBM()
        )

        self.qcbm_model.fit(
            latent_vectors
        )

        self.cluster_ids = (
            self.qcbm_model.cluster_ids
        )

        return self

    def generate(
        self,
        n_samples=1000
    ):
        """
        Generate molecules from QCBM.
        """

        self._check_fitted()

        sampled_states = (
            self.qcbm_model.sample_states(
                n_samples=n_samples
            )
        )

        generated = (
            self.reconstructor
            .reconstruct_distribution(
                sampled_states,
                self.cluster_ids,
                self.training_smiles
            )
        )

        self.generated_smiles = (
            generated
        )

        return generated

    def evaluate_rewards(
        self,
        generated_smiles=None
    ):
        """
        Compute reward table.
        """

        if generated_smiles is None:

            generated_smiles = (
                self.generated_smiles
            )

        rows = []

        for smiles in generated_smiles:

            syba = (
                syba_score_smiles(
                    smiles
                )
            )

            qed = (
                qed_score(
                    smiles
                )
            )

            novelty = (
                novelty_score(
                    smiles,
                    self.training_smiles
                )
            )

            reward = self.reward(
                smiles,
                self.training_smiles
            )

            rows.append(
                {
                    "SMILES":
                        smiles,

                    "SYBA":
                        syba,

                    "QED":
                        qed,

                    "Novelty":
                        novelty,

                    "Reward":
                        reward
                }
            )

        self.reward_table = (
            pd.DataFrame(
                rows
            )
        )

        return self.reward_table

    def reinforced_generation(
        self,
        n_samples=1000,
        top_fraction=0.2
    ):
        """
        Generate and keep only
        high reward molecules.
        """

        generated = (
            self.generate(
                n_samples
            )
        )

        reward_df = (
            self.evaluate_rewards(
                generated
            )
        )

        reward_df = (
            reward_df
            .sort_values(
                "Reward",
                ascending=False
            )
        )

        keep_n = max(
            1,
            int(
                top_fraction
                *
                len(reward_df)
            )
        )

        reward_df = (
            reward_df
            .head(
                keep_n
            )
        )

        return (
            reward_df[
                "SMILES"
            ]
            .tolist()
        )

    def reward_statistics(
        self
    ):
        """
        Reward diagnostics.
        """

        self._check_rewards()

        rewards = (
            self.reward_table[
                "Reward"
            ]
            .values
        )

        return {

            "mean":
                float(
                    np.mean(
                        rewards
                    )
                ),

            "std":
                float(
                    np.std(
                        rewards
                    )
                ),

            "min":
                float(
                    np.min(
                        rewards
                    )
                ),

            "max":
                float(
                    np.max(
                        rewards
                    )
                )
        }

    def top_molecules(
        self,
        top_k=50
    ):
        """
        Highest reward molecules.
        """

        self._check_rewards()

        return (

            self.reward_table
            .sort_values(
                "Reward",
                ascending=False
            )
            .head(
                top_k
            )
            .reset_index(
                drop=True
            )
        )

    def benchmark(
        self,
        n_samples=1000
    ):
        """
        RL-QCBM benchmark.
        """

        generated = (
            self.reinforced_generation(
                n_samples=n_samples
            )
        )

        stats = (
            self.reward_statistics()
        )

        return {

            "method":
                "RL+QCBM",

            "generated":
                len(
                    generated
                ),

            "reward_statistics":
                stats
        }

    def summary(
        self
    ):
        """
        Model summary.
        """

        return {

            "method":
                "RL+QCBM",

            "reward_weight_syba":
                self.reward_weight_syba,

            "reward_weight_qed":
                self.reward_weight_qed,

            "reward_weight_novelty":
                self.reward_weight_novelty,

            "fitted":
                self.qcbm_model
                is not None
        }

    def _check_fitted(
        self
    ):
        if self.qcbm_model is None:

            raise RuntimeError(
                "Call fit() first."
            )

    def _check_rewards(
        self
    ):
        if self.reward_table is None:

            raise RuntimeError(
                "Run evaluate_rewards() first."
            )