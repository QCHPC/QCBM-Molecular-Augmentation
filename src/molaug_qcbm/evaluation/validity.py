# validity metric placeholder
from rdkit import Chem


def is_valid_smiles(
    smiles
):
    """
    Check whether a SMILES string
    represents a valid molecule.
    """

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return False

        Chem.SanitizeMol(
            mol
        )

        return True

    except Exception:

        return False


def validity_score(
    smiles_list
):
    """
    Fraction of valid molecules.

    Returns value in [0,1].
    """

    if len(smiles_list) == 0:

        return 0.0

    valid_count = sum(
        is_valid_smiles(s)
        for s in smiles_list
    )

    return (
        valid_count
        /
        len(smiles_list)
    )


def validity_percentage(
    smiles_list
):
    """
    Validity in percentage.
    """

    return (
        validity_score(
            smiles_list
        )
        * 100
    )


def valid_molecules(
    smiles_list
):
    """
    Return only valid molecules.
    """

    return [
        s
        for s in smiles_list
        if is_valid_smiles(s)
    ]


def invalid_molecules(
    smiles_list
):
    """
    Return invalid molecules.
    """

    return [
        s
        for s in smiles_list
        if not is_valid_smiles(s)
    ]


def validity_report(
    smiles_list
):
    """
    Complete validity statistics.
    """

    valid = valid_molecules(
        smiles_list
    )

    invalid = invalid_molecules(
        smiles_list
    )

    total = len(
        smiles_list
    )

    return {
        "total":
            total,

        "valid":
            len(valid),

        "invalid":
            len(invalid),

        "validity":
            (
                len(valid)
                /
                total
                if total > 0
                else 0
            ),

        "validity_percent":
            (
                100
                * len(valid)
                / total
                if total > 0
                else 0
            )
    }