# graph builder placeholder
from rdkit import Chem

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I"
]


BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3
}


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


def atom_features(
    atom
):
    """
    Node feature vector.

    Features:
    - atom type one-hot
    - degree
    - formal charge
    - aromaticity
    """

    symbol = atom.GetSymbol()

    atom_type = [
        int(symbol == a)
        for a in ATOM_TYPES
    ]

    features = atom_type + [

        atom.GetDegree(),

        atom.GetFormalCharge(),

        int(
            atom.GetIsAromatic()
        )
    ]

    return features


def bond_feature(
    bond
):
    """
    Bond encoding.
    """

    bond_type = bond.GetBondType()

    return BOND_TYPES.get(
        bond_type,
        0
    )


def build_node_features(
    mol
):
    """
    Build node feature matrix.
    """

    features = []

    for atom in mol.GetAtoms():

        features.append(
            atom_features(
                atom
            )
        )

    return torch.tensor(
        features,
        dtype=torch.float
    )


def build_edge_index(
    mol
):
    """
    PyTorch Geometric edge index.
    """

    edges = []

    for bond in mol.GetBonds():

        start = bond.GetBeginAtomIdx()

        end = bond.GetEndAtomIdx()

        edges.append(
            [start, end]
        )

        edges.append(
            [end, start]
        )

    if len(edges) == 0:

        return torch.empty(
            (2, 0),
            dtype=torch.long
        )

    return torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()


def build_edge_attr(
    mol
):
    """
    Edge features.
    """

    attrs = []

    for bond in mol.GetBonds():

        value = bond_feature(
            bond
        )

        attrs.append(
            [value]
        )

        attrs.append(
            [value]
        )

    if len(attrs) == 0:

        return torch.empty(
            (0, 1),
            dtype=torch.float
        )

    return torch.tensor(
        attrs,
        dtype=torch.float
    )


def smiles_to_graph(
    smiles
):
    """
    Convert SMILES into graph tensors.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:

        return None

    graph = {

        "x":
            build_node_features(
                mol
            ),

        "edge_index":
            build_edge_index(
                mol
            ),

        "edge_attr":
            build_edge_attr(
                mol
            )
    }

    return graph


def smiles_to_pyg(
    smiles
):
    """
    Convert SMILES directly into
    PyTorch Geometric Data object.
    """

    if Data is None:

        raise ImportError(
            "torch_geometric not installed."
        )

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:

        return None

    x = build_node_features(
        mol
    )

    edge_index = build_edge_index(
        mol
    )

    edge_attr = build_edge_attr(
        mol
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )


def batch_graphs(
    smiles_list
):
    """
    Convert many molecules into
    graph objects.
    """

    graphs = []

    for smiles in smiles_list:

        graph = smiles_to_graph(
            smiles
        )

        if graph is not None:

            graphs.append(
                graph
            )

    return graphs


def batch_pyg_graphs(
    smiles_list
):
    """
    Convert many molecules into
    PyG Data objects.
    """

    graphs = []

    for smiles in smiles_list:

        graph = smiles_to_pyg(
            smiles
        )

        if graph is not None:

            graphs.append(
                graph
            )

    return graphs


def graph_statistics(
    smiles
):
    """
    Basic graph statistics.
    """

    mol = smiles_to_mol(
        smiles
    )

    if mol is None:

        return None

    return {

        "nodes":
            mol.GetNumAtoms(),

        "edges":
            mol.GetNumBonds(),

        "rings":
            mol.GetRingInfo().NumRings(),

        "heavy_atoms":
            mol.GetNumHeavyAtoms()
    }