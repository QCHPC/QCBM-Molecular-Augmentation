# descriptors placeholder
from rdkit import Chem

from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors

import pandas as pd


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


def molecular_weight(
    smiles
):
    """
    Molecular Weight.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return float(
        Descriptors.MolWt(
            mol
        )
    )


def logp(
    smiles
):
    """
    Octanol-water partition coefficient.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return float(
        Crippen.MolLogP(
            mol
        )
    )


def h_bond_donors(
    smiles
):
    """
    Hydrogen Bond Donors.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return int(
        Lipinski.NumHDonors(
            mol
        )
    )


def h_bond_acceptors(
    smiles
):
    """
    Hydrogen Bond Acceptors.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return int(
        Lipinski.NumHAcceptors(
            mol
        )
    )


def rotatable_bonds(
    smiles
):
    """
    Rotatable bonds.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return int(
        Lipinski.NumRotatableBonds(
            mol
        )
    )


def tpsa(
    smiles
):
    """
    Topological Polar Surface Area.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return float(
        rdMolDescriptors.CalcTPSA(
            mol
        )
    )


def ring_count(
    smiles
):
    """
    Number of rings.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return int(
        rdMolDescriptors.CalcNumRings(
            mol
        )
    )


def heavy_atom_count(
    smiles
):
    """
    Number of heavy atoms.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:
        return None

    return int(
        mol.GetNumHeavyAtoms()
    )


def descriptor_dict(
    smiles
):
    """
    Compute all descriptors.
    """

    return {

        "SMILES":
            smiles,

        "MolWt":
            molecular_weight(
                smiles
            ),

        "LogP":
            logp(
                smiles
            ),

        "HBD":
            h_bond_donors(
                smiles
            ),

        "HBA":
            h_bond_acceptors(
                smiles
            ),

        "TPSA":
            tpsa(
                smiles
            ),

        "RotatableBonds":
            rotatable_bonds(
                smiles
            ),

        "RingCount":
            ring_count(
                smiles
            ),

        "HeavyAtoms":
            heavy_atom_count(
                smiles
            )
    }


def descriptor_dataframe(
    smiles_list
):
    """
    Convert molecule list
    into descriptor dataframe.
    """

    rows = []

    for smiles in smiles_list:

        row = descriptor_dict(
            smiles
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def descriptor_summary(
    smiles_list
):
    """
    Dataset-level descriptor statistics.
    """

    df = descriptor_dataframe(
        smiles_list
    )

    numeric_cols = [
        "MolWt",
        "LogP",
        "HBD",
        "HBA",
        "TPSA",
        "RotatableBonds",
        "RingCount",
        "HeavyAtoms"
    ]

    return (
        df[numeric_cols]
        .describe()
        .T
    )


def compare_descriptor_distributions(
    training_smiles,
    generated_smiles
):
    """
    Compare training vs generated.
    """

    train_df = (
        descriptor_dataframe(
            training_smiles
        )
    )

    gen_df = (
        descriptor_dataframe(
            generated_smiles
        )
    )

    metrics = [
        "MolWt",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotatableBonds",
        "RingCount",
        "HeavyAtoms"
    ]

    rows = []

    for metric in metrics:

        rows.append({

            "Descriptor":
                metric,

            "Training Mean":
                train_df[
                    metric
                ].mean(),

            "Generated Mean":
                gen_df[
                    metric
                ].mean()
        })

    return pd.DataFrame(
        rows
    )