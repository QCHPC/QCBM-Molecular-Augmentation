from typing import List
from typing import Optional
from typing import Dict

import numpy as np
import pandas as pd

from rdkit import Chem

from molaug_qcbm.utils import sascorer


def smiles_to_mol(
    smiles: str
):
    """
    Convert SMILES to RDKit molecule.
    """

    if smiles is None:
        return None

    try:

        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        return mol

    except Exception:

        return None


def compute_sa(
    smiles: str
) -> Optional[float]:
    """
    Compute Synthetic Accessibility score.

    Scale:
        1 = easy to synthesize
        10 = difficult to synthesize
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    try:

        score = sascorer.calculateScore(
            mol
        )

        return float(score)

    except Exception:

        return None


def compute_sa_batch(
    smiles_list: List[str]
) -> List[Optional[float]]:
    """
    Compute SA score for multiple molecules.
    """

    return [
        compute_sa(smiles)
        for smiles in smiles_list
    ]


def compute_mean_sa(
    smiles_list: List[str]
) -> Optional[float]:
    """
    Mean SA score.
    """

    scores = [
        s
        for s in compute_sa_batch(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return None

    return float(
        np.mean(scores)
    )


def compute_median_sa(
    smiles_list: List[str]
) -> Optional[float]:
    """
    Median SA score.
    """

    scores = [
        s
        for s in compute_sa_batch(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return None

    return float(
        np.median(scores)
    )


def sa_dataframe(
    smiles_list: List[str]
) -> pd.DataFrame:
    """
    Create dataframe containing SA scores.
    """

    rows = []

    for smiles in smiles_list:

        rows.append(
            {
                "SMILES": smiles,
                "SA_Score": compute_sa(
                    smiles
                )
            }
        )

    return pd.DataFrame(
        rows
    )


def easiest_molecules(
    smiles_list: List[str],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Lowest SA molecules.
    """

    df = sa_dataframe(
        smiles_list
    )

    df = df.dropna()

    return (
        df
        .sort_values(
            by="SA_Score",
            ascending=True
        )
        .head(top_k)
        .reset_index(
            drop=True
        )
    )


def hardest_molecules(
    smiles_list: List[str],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Highest SA molecules.
    """

    df = sa_dataframe(
        smiles_list
    )

    df = df.dropna()

    return (
        df
        .sort_values(
            by="SA_Score",
            ascending=False
        )
        .head(top_k)
        .reset_index(
            drop=True
        )
    )


def sa_report(
    smiles_list: List[str]
) -> Dict:
    """
    Summary statistics.
    """

    scores = [
        s
        for s in compute_sa_batch(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return {
            "n_molecules": 0,
            "mean_sa": None,
            "median_sa": None,
            "std_sa": None,
            "min_sa": None,
            "max_sa": None
        }

    return {

        "n_molecules":
            len(scores),

        "mean_sa":
            float(
                np.mean(scores)
            ),

        "median_sa":
            float(
                np.median(scores)
            ),

        "std_sa":
            float(
                np.std(scores)
            ),

        "min_sa":
            float(
                np.min(scores)
            ),

        "max_sa":
            float(
                np.max(scores)
            )
    }


def compare_sa(
    training_smiles: List[str],
    generated_smiles: List[str]
):
    """
    Compare train vs generated molecules.
    """

    train_sa = compute_mean_sa(
        training_smiles
    )

    generated_sa = compute_mean_sa(
        generated_smiles
    )

    difference = None

    if (
        train_sa is not None
        and generated_sa is not None
    ):

        difference = (
            generated_sa
            - train_sa
        )

    return {

        "training_mean_sa":
            train_sa,

        "generated_mean_sa":
            generated_sa,

        "difference":
            difference
    }
    
# ============================================================
# Backward Compatibility
# ============================================================

def sa_score(
    smiles: str
):
    """
    Backward-compatible wrapper.

    Existing modules may import:

        from molaug_qcbm.evaluation.sa_score import sa_score
    """

    return compute_sa(smiles)