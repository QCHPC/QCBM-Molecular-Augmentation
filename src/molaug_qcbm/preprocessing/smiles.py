from rdkit import Chem

def is_valid_smiles(smiles):
    return valid_smiles(smiles)

def valid_smiles(smiles):

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


def is_valid_smiles(smiles):
    """
    Compatibility wrapper.

    Expected by:
        validity.py
        filtering.py
        other modules
    """

    return valid_smiles(
        smiles
    )


def sanitize_smiles(smiles):

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return None, None, False

        Chem.SanitizeMol(
            mol
        )

        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False
        )

        return (
            mol,
            canonical,
            True
        )

    except Exception:

        return (
            None,
            None,
            False
        )