# novelty metric placeholder
from rdkit import Chem


def canonicalize_smiles(
    smiles
):
    """
    Convert SMILES to canonical form.

    Used to ensure equivalent
    molecules are treated identically.
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


def training_set(
    training_smiles
):
    """
    Convert training molecules
    into canonical set.
    """

    return {
        canonicalize_smiles(s)
        for s in training_smiles
        if canonicalize_smiles(s) is not None
    }


def novel_molecules(
    generated_smiles,
    training_smiles
):
    """
    Return molecules that do not
    exist in training data.
    """

    train_set = training_set(
        training_smiles
    )

    novel = []

    for smiles in generated_smiles:

        canonical = (
            canonicalize_smiles(
                smiles
            )
        )

        if canonical is None:
            continue

        if canonical not in train_set:

            novel.append(
                canonical
            )

    return novel


def novelty_score(
    generated_smiles,
    training_smiles
):
    """
    Novelty metric.

    Returns value in [0,1].
    """

    if len(generated_smiles) == 0:

        return 0.0

    novel = novel_molecules(
        generated_smiles,
        training_smiles
    )

    return (
        len(novel)
        /
        len(generated_smiles)
    )


def novelty_percentage(
    generated_smiles,
    training_smiles
):
    """
    Novelty as percentage.
    """

    return (
        novelty_score(
            generated_smiles,
            training_smiles
        )
        * 100
    )


def seen_molecules(
    generated_smiles,
    training_smiles
):
    """
    Molecules already present
    in training data.
    """

    train_set = training_set(
        training_smiles
    )

    seen = []

    for smiles in generated_smiles:

        canonical = (
            canonicalize_smiles(
                smiles
            )
        )

        if canonical is None:
            continue

        if canonical in train_set:

            seen.append(
                canonical
            )

    return seen


def novelty_report(
    generated_smiles,
    training_smiles
):
    """
    Complete novelty statistics.
    """

    total_generated = len(
        generated_smiles
    )

    if total_generated == 0:

        return {
            "generated": 0,
            "novel": 0,
            "seen": 0,
            "novelty": 0.0,
            "novelty_percent": 0.0
        }

    novel = novel_molecules(
        generated_smiles,
        training_smiles
    )

    seen = seen_molecules(
        generated_smiles,
        training_smiles
    )

    return {

        "generated":
            total_generated,

        "novel":
            len(novel),

        "seen":
            len(seen),

        "novelty":
            len(novel)
            /
            total_generated,

        "novelty_percent":
            (
                100
                * len(novel)
                / total_generated
            )
    }


def novelty_dataframe(
    generated_smiles,
    training_smiles
):
    """
    Create dataframe showing
    whether each generated molecule
    is novel.
    """

    import pandas as pd

    train_set = training_set(
        training_smiles
    )

    rows = []

    for smiles in generated_smiles:

        canonical = (
            canonicalize_smiles(
                smiles
            )
        )

        if canonical is None:
            continue

        rows.append(
            {
                "SMILES":
                    canonical,

                "Novel":
                    canonical
                    not in train_set
            }
        )

    return pd.DataFrame(
        rows
    )


def novelty_summary(
    generated_smiles,
    training_smiles
):
    """
    Compact benchmark output.
    """

    report = novelty_report(
        generated_smiles,
        training_smiles
    )

    return {
        "Novelty":
            report["novelty"],

        "Novelty (%)":
            report["novelty_percent"],

        "Novel Molecules":
            report["novel"]
    }