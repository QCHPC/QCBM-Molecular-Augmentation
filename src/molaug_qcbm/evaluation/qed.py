# qed placeholder
from rdkit import Chem
from rdkit.Chem import QED

import pandas as pd
import numpy as np


def smiles_to_mol(
    smiles
):
    """
    Convert SMILES to RDKit Mol.
    """

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        return mol

    except Exception:

        return None


def qed_score(
    smiles
):
    """
    Compute QED score.

    Range:
        0 → 1

    Higher is better.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:

        return None

    try:

        return float(
            QED.qed(
                mol
            )
        )

    except Exception:

        return None


def batch_qed(
    smiles_list
):
    """
    Compute QED for many molecules.
    """

    scores = []

    for smiles in smiles_list:

        score = qed_score(
            smiles
        )

        scores.append(
            score
        )

    return scores


def average_qed(
    smiles_list
):
    """
    Average QED score.
    """

    scores = [
        s
        for s in batch_qed(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return 0.0

    return float(
        np.mean(
            scores
        )
    )


def median_qed(
    smiles_list
):
    """
    Median QED score.
    """

    scores = [
        s
        for s in batch_qed(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return 0.0

    return float(
        np.median(
            scores
        )
    )


def max_qed(
    smiles_list
):
    """
    Best QED molecule.
    """

    scores = [
        s
        for s in batch_qed(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return 0.0

    return float(
        np.max(
            scores
        )
    )


def min_qed(
    smiles_list
):
    """
    Worst QED molecule.
    """

    scores = [
        s
        for s in batch_qed(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return 0.0

    return float(
        np.min(
            scores
        )
    )


def qed_dataframe(
    smiles_list
):
    """
    Molecule-wise QED table.
    """

    rows = []

    for smiles in smiles_list:

        rows.append(
            {
                "SMILES":
                    smiles,

                "QED":
                    qed_score(
                        smiles
                    )
            }
        )

    return pd.DataFrame(
        rows
    )


def top_qed_molecules(
    smiles_list,
    top_k=20
):
    """
    Return best molecules by QED.
    """

    df = qed_dataframe(
        smiles_list
    )

    df = df.dropna()

    return (
        df
        .sort_values(
            "QED",
            ascending=False
        )
        .head(top_k)
        .reset_index(
            drop=True
        )
    )


def qed_distribution(
    smiles_list,
    bins=20
):
    """
    Histogram-ready QED distribution.
    """

    scores = [
        s
        for s in batch_qed(
            smiles_list
        )
        if s is not None
    ]

    hist, bin_edges = np.histogram(
        scores,
        bins=bins
    )

    return {
        "histogram":
            hist,

        "bin_edges":
            bin_edges
    }


def qed_report(
    smiles_list
):
    """
    Full QED summary.
    """

    scores = [
        s
        for s in batch_qed(
            smiles_list
        )
        if s is not None
    ]

    if len(scores) == 0:

        return {

            "n_molecules": 0,

            "mean_qed": 0.0,

            "median_qed": 0.0,

            "max_qed": 0.0,

            "min_qed": 0.0
        }

    return {

        "n_molecules":
            len(scores),

        "mean_qed":
            float(
                np.mean(
                    scores
                )
            ),

        "median_qed":
            float(
                np.median(
                    scores
                )
            ),

        "max_qed":
            float(
                np.max(
                    scores
                )
            ),

        "min_qed":
            float(
                np.min(
                    scores
                )
            )
    }


def compare_qed(
    training_smiles,
    generated_smiles
):
    """
    Compare train vs generated.
    """

    return {

        "training_qed":
            average_qed(
                training_smiles
            ),

        "generated_qed":
            average_qed(
                generated_smiles
            ),

        "difference":
            (
                average_qed(
                    generated_smiles
                )
                -
                average_qed(
                    training_smiles
                )
            )
    }