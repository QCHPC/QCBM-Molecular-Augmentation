# rl_generator.py

import random
from typing import List

import numpy as np
import pandas as pd

from molaug_qcbm.evaluation.syba import (
    syba_score_smiles,
)

from molaug_qcbm.evaluation.qed import (
    qed_score,
)


class RLMoleculeGenerator:
    """
    Classical reward-guided molecular generator.

    Reward Function
    ---------------
    Reward =

        w1 × SYBA
      + w2 × QED

    Novelty is intentionally NOT used during training.

    Novelty should only be computed after generation during
    benchmarking, since every training molecule has novelty = 0
    with respect to the training set.

    This implementation is intended as a lightweight
    classical baseline for comparison against QCBM.
    """

    def __init__(
        self,
        reward_weight_syba=1.0,
        reward_weight_qed=1.0,
        reward_weight_novelty=1.0,
        random_state=42,
    ):

        self.reward_weight_syba = reward_weight_syba
        self.reward_weight_qed = reward_weight_qed

        # kept only for API compatibility
        self.reward_weight_novelty = reward_weight_novelty

        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        self.training_smiles = None
        self.reward_table = None

    ####################################################################
    # Reward
    ####################################################################

    def reward(
        self,
        smiles,
    ):
        """
        Compute reward for one molecule.

        Reward =
            SYBA + QED
        """

        syba = syba_score_smiles(smiles)

        if syba is None or np.isnan(syba):
            syba = 0.0

        qed = qed_score(smiles)

        if qed is None or np.isnan(qed):
            qed = 0.0

        reward = (

            self.reward_weight_syba
            * syba

            +

            self.reward_weight_qed
            * qed

        )

        return float(reward)

    ####################################################################
    # Fit
    ####################################################################

    def fit(
        self,
        smiles_list: List[str],
    ):
        """
        Build reward table.

        Complexity:
            O(N)

        Previous implementation used novelty inside the
        reward function, producing O(N²) behaviour.
        """

        self.training_smiles = list(smiles_list)

        rows = []

        for i, smiles in enumerate(smiles_list):

            reward = self.reward(smiles)

            rows.append(
                {
                    "SMILES": smiles,
                    "SYBA": syba_score_smiles(smiles),
                    "QED": qed_score(smiles),
                    "Reward": reward,
                }
            )

            if (i + 1) % 5000 == 0:

                print(
                    f"Processed {i+1:,} molecules..."
                )

        self.reward_table = (
            pd.DataFrame(rows)
            .sort_values(
                "Reward",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return self

    ####################################################################
    # Generation
    ####################################################################

    def generate(
        self,
        n_samples=1000,
        exploration_rate=0.10,
    ):
        """
        Generate molecules.

        Exploration:
            random molecule

        Exploitation:
            sample from top 10%
        """

        self._check_fitted()

        smiles_pool = (
            self.reward_table["SMILES"]
            .tolist()
        )

        top_n = max(
            10,
            int(
                0.10
                * len(smiles_pool)
            ),
        )

        elite_pool = smiles_pool[:top_n]

        generated = []

        for _ in range(n_samples):

            if random.random() < exploration_rate:

                generated.append(
                    random.choice(smiles_pool)
                )

            else:

                generated.append(
                    random.choice(elite_pool)
                )

        return generated

    ####################################################################
    # Analysis
    ####################################################################

    def reward_distribution(
        self,
    ):

        self._check_fitted()

        rewards = self.reward_table[
            "Reward"
        ].values

        return {

            "mean":
                float(np.mean(rewards)),

            "std":
                float(np.std(rewards)),

            "min":
                float(np.min(rewards)),

            "max":
                float(np.max(rewards)),

            "median":
                float(np.median(rewards)),
        }

    def top_molecules(
        self,
        top_k=100,
    ):

        self._check_fitted()

        return self.reward_table.head(top_k)

    def benchmark(
        self,
        n_samples=1000,
    ):

        generated = self.generate(
            n_samples=n_samples
        )

        return {

            "method":
                "RL Generator",

            "generated":
                len(generated),

            "reward_statistics":
                self.reward_distribution(),
        }

    ####################################################################
    # Utilities
    ####################################################################

    def save_rewards(
        self,
        filepath,
    ):

        self._check_fitted()

        self.reward_table.to_csv(
            filepath,
            index=False,
        )

    def summary(
        self,
    ):

        return {

            "method":
                "RL Generator",

            "fitted":
                self.reward_table is not None,

            "training_molecules":
                0
                if self.training_smiles is None
                else len(self.training_smiles),

            "reward_weight_syba":
                self.reward_weight_syba,

            "reward_weight_qed":
                self.reward_weight_qed,

            "reward_weight_novelty":
                self.reward_weight_novelty,
        }

    ####################################################################
    # Internal
    ####################################################################

    def _check_fitted(
        self,
    ):

        if self.reward_table is None:

            raise RuntimeError(
                "Call fit() before generate()."
            )