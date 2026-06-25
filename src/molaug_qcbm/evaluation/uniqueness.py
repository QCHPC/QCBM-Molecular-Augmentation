# uniqueness metric placeholder
from rdkit import Chem


def canonicalize_smiles(
    smiles
):
    """
    Convert SMILES to canonical form.

    Equivalent molecules become identical.
    """

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True
        )

    except Exception:

        return None


def unique_molecules(
    smiles_list
):
    """
    Return set of unique canonical molecules.
    """

    unique = set()

    for smiles in smiles_list:

        canonical = canonicalize_smiles(
            smiles
        )

        if canonical is not None:

            unique.add(
                canonical
            )

    return unique


def uniqueness_score(
    smiles_list
):
    """
    Fraction of unique molecules.

    Returns value in [0,1].
    """

    if len(smiles_list) == 0:

        return 0.0

    unique = unique_molecules(
        smiles_list
    )

    return (
        len(unique)
        /
        len(smiles_list)
    )


def uniqueness_percentage(
    smiles_list
):
    """
    Uniqueness as percentage.
    """

    return (
        uniqueness_score(
            smiles_list
        )
        * 100
    )


def duplicate_count(
    smiles_list
):
    """
    Number of duplicate molecules.
    """

    unique = unique_molecules(
        smiles_list
    )

    return (
        len(smiles_list)
        - len(unique)
    )


def uniqueness_report(
    smiles_list
):
    """
    Full uniqueness statistics.
    """

    unique = unique_molecules(
        smiles_list
    )

    total = len(
        smiles_list
    )

    unique_count = len(
        unique
    )

    duplicates = (
        total - unique_count
    )

    return {
        "total":
            total,

        "unique":
            unique_count,

        "duplicates":
            duplicates,

        "uniqueness":
            (
                unique_count
                / total
                if total > 0
                else 0
            ),

        "uniqueness_percent":
            (
                100
                * unique_count
                / total
                if total > 0
                else 0
            )
    }


def duplicate_molecules(
    smiles_list
):
    """
    Return molecules appearing more than once.
    """

    counts = {}

    for smiles in smiles_list:

        canonical = canonicalize_smiles(
            smiles
        )

        if canonical is None:
            continue

        counts[canonical] = (
            counts.get(
                canonical,
                0
            )
            + 1
        )

    duplicates = {
        smi: count
        for smi, count
        in counts.items()
        if count > 1
    }

    return duplicates