# fingerprints placeholder
from rdkit import Chem

from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

import numpy as np


def smiles_to_mol(
    smiles
):
    """
    Convert SMILES to RDKit molecule.
    """

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        return mol

    except Exception:

        return None


def morgan_fingerprint(
    smiles,
    radius=2,
    n_bits=2048
):
    """
    Generate Morgan fingerprint.

    Default:
        ECFP4
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:

        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    return fp


def fingerprint_array(
    smiles,
    radius=2,
    n_bits=2048
):
    """
    Convert fingerprint
    into numpy array.
    """

    fp = morgan_fingerprint(
        smiles,
        radius=radius,
        n_bits=n_bits
    )

    if fp is None:

        return None

    arr = np.zeros(
        (n_bits,),
        dtype=int
    )

    DataStructs.ConvertToNumpyArray(
        fp,
        arr
    )

    return arr


def batch_fingerprints(
    smiles_list,
    radius=2,
    n_bits=2048
):
    """
    Generate fingerprints
    for many molecules.
    """

    fps = []

    valid_smiles = []

    for smiles in smiles_list:

        arr = fingerprint_array(
            smiles,
            radius=radius,
            n_bits=n_bits
        )

        if arr is not None:

            fps.append(
                arr
            )

            valid_smiles.append(
                smiles
            )

    return (
        np.array(fps),
        valid_smiles
    )


def tanimoto_similarity(
    smiles_a,
    smiles_b,
    radius=2,
    n_bits=2048
):
    """
    Similarity between
    two molecules.
    """

    fp1 = morgan_fingerprint(
        smiles_a,
        radius,
        n_bits
    )

    fp2 = morgan_fingerprint(
        smiles_b,
        radius,
        n_bits
    )

    if fp1 is None or fp2 is None:

        return 0.0

    return float(
        DataStructs.TanimotoSimilarity(
            fp1,
            fp2
        )
    )


def bulk_tanimoto_similarity(
    query_smiles,
    smiles_list,
    radius=2,
    n_bits=2048
):
    """
    Similarity between query
    and many molecules.
    """

    query_fp = morgan_fingerprint(
        query_smiles,
        radius,
        n_bits
    )

    if query_fp is None:

        return []

    similarities = []

    for smiles in smiles_list:

        fp = morgan_fingerprint(
            smiles,
            radius,
            n_bits
        )

        if fp is None:

            similarities.append(
                0.0
            )

            continue

        score = (
            DataStructs.TanimotoSimilarity(
                query_fp,
                fp
            )
        )

        similarities.append(
            float(score)
        )

    return similarities


def most_similar_molecules(
    query_smiles,
    smiles_list,
    top_k=10
):
    """
    Retrieve most similar molecules.
    """

    similarities = (
        bulk_tanimoto_similarity(
            query_smiles,
            smiles_list
        )
    )

    ranked = sorted(
        zip(
            smiles_list,
            similarities
        ),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


def fingerprint_similarity_matrix(
    smiles_list
):
    """
    Compute pairwise similarity matrix.
    """

    n = len(
        smiles_list
    )

    matrix = np.zeros(
        (
            n,
            n
        )
    )

    fps = []

    for smiles in smiles_list:

        fps.append(
            morgan_fingerprint(
                smiles
            )
        )

    for i in range(n):

        for j in range(
            i,
            n
        ):

            if (
                fps[i] is None
                or
                fps[j] is None
            ):

                sim = 0.0

            else:

                sim = (
                    DataStructs.TanimotoSimilarity(
                        fps[i],
                        fps[j]
                    )
                )

            matrix[i, j] = sim
            matrix[j, i] = sim

    return matrix


def average_diversity(
    smiles_list
):
    """
    Molecular diversity.

    Diversity =
        1 - Avg Similarity
    """

    matrix = (
        fingerprint_similarity_matrix(
            smiles_list
        )
    )

    n = len(
        smiles_list
    )

    if n <= 1:

        return 0.0

    upper = []

    for i in range(n):

        for j in range(
            i + 1,
            n
        ):

            upper.append(
                matrix[i, j]
            )

    avg_similarity = (
        np.mean(upper)
    )

    return float(
        1.0 - avg_similarity
    )