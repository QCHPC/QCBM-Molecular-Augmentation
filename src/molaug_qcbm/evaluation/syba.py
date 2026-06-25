from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from rdkit import Chem

try:
    from syba.syba import SybaClassifier

    SYBA_AVAILABLE = True

except Exception:

    SYBA_AVAILABLE = False


class SYBAScorer:
    """
    Wrapper around SYBA synthetic accessibility scoring.

    Interpretation
    --------------
    score > 0 :
        Easier to synthesize

    score < 0 :
        Harder to synthesize

    score ~= 0 :
        Borderline
    """

    def __init__(self):

        self.model = None

        if SYBA_AVAILABLE:

            try:

                self.model = SybaClassifier()
                self.model.fitDefaultScore()

            except Exception:

                self.model = None

    def score_smiles(
        self,
        smiles: str
    ) -> float:
        """
        Compute SYBA score from SMILES.
        """

        if self.model is None:

            return np.nan

        try:

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:

                return np.nan

            Chem.SanitizeMol(mol)

            return float(
                self.model.predict(
                    mol=mol
                )
            )

        except Exception:

            return np.nan

    def score_molecule(
        self,
        mol
    ) -> float:
        """
        Compute SYBA score from RDKit Mol.
        """

        if self.model is None:

            return np.nan

        if mol is None:

            return np.nan

        try:

            return float(
                self.model.predict(
                    mol=mol
                )
            )

        except Exception:

            return np.nan

    def batch_score(
        self,
        smiles_list: List[str]
    ) -> List[float]:
        """
        Compute SYBA scores for multiple molecules.
        """

        return [
            self.score_smiles(smiles)
            for smiles in smiles_list
        ]

    def score_dataframe(
        self,
        df: pd.DataFrame,
        smiles_column: str = "SMILES"
    ) -> pd.DataFrame:
        """
        Add SYBA scores to a dataframe.
        """

        result = df.copy()

        result["SYBA"] = result[
            smiles_column
        ].apply(
            self.score_smiles
        )

        return result

    def summary_statistics(
        self,
        smiles_list: List[str]
    ) -> Dict:
        """
        Summary statistics for a set of molecules.
        """

        scores = np.array(
            self.batch_score(
                smiles_list
            ),
            dtype=float
        )

        scores = scores[
            ~np.isnan(scores)
        ]

        if len(scores) == 0:

            return {
                "n_molecules": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None
            }

        return {
            "n_molecules": int(len(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }


# ============================================================
# Global scorer instance
# ============================================================

syba_scorer = SYBAScorer()


# ============================================================
# Convenience functions
# ============================================================

def syba_score_smiles(
    smiles: str
) -> float:
    """
    Module-level SYBA scorer.

    Example
    -------
    from molaug_qcbm.evaluation.syba import syba_score_smiles

    score = syba_score_smiles("CCO")
    """

    return syba_scorer.score_smiles(
        smiles
    )


def syba_score_molecule(
    mol
) -> float:
    """
    Compute SYBA score from RDKit Mol.
    """

    return syba_scorer.score_molecule(
        mol
    )


def batch_syba_scores(
    smiles_list: List[str]
) -> List[float]:
    """
    Compute SYBA scores for a list of SMILES.
    """

    return syba_scorer.batch_score(
        smiles_list
    )


def mean_syba_score(
    smiles_list: List[str]
) -> Optional[float]:
    """
    Mean SYBA score.
    """

    scores = np.array(
        batch_syba_scores(
            smiles_list
        ),
        dtype=float
    )

    scores = scores[
        ~np.isnan(scores)
    ]

    if len(scores) == 0:

        return None

    return float(
        np.mean(scores)
    )


def syba_report(
    smiles_list: List[str]
) -> Dict:
    """
    Generate summary report.
    """

    return syba_scorer.summary_statistics(
        smiles_list
    )