# gnn embedding placeholder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.loader import DataLoader
except ImportError:
    GCNConv = None
    global_mean_pool = None
    DataLoader = None

from molaug_qcbm.preprocessing.graph_builder import (
    batch_pyg_graphs
)


class SimpleGNNEncoder(nn.Module):
    """
    Simple GCN encoder for molecular graph embeddings.

    Input:
        PyTorch Geometric molecular graph

    Output:
        Fixed-size molecular embedding
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        embedding_dim=32
    ):
        super().__init__()

        if GCNConv is None:
            raise ImportError(
                "torch_geometric is not installed. "
                "Install with: pip install torch torch-geometric"
            )

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim
        )

        self.proj = nn.Linear(
            hidden_dim,
            embedding_dim
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):
        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        pooled = global_mean_pool(
            x,
            batch
        )

        embedding = self.proj(
            pooled
        )

        return embedding


class GNNEmbeddingEncoder:
    """
    Converts SMILES into GNN molecular embeddings.

    Pipeline:
        SMILES
        ↓
        Molecular Graph
        ↓
        GNN
        ↓
        Embedding Vector
        ↓
        KMeans / QCBM
    """

    def __init__(
        self,
        hidden_dim=64,
        embedding_dim=32,
        batch_size=64,
        device=None
    ):
        if DataLoader is None:
            raise ImportError(
                "torch_geometric is not installed. "
                "Install with: pip install torch torch-geometric"
            )

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = None
        self.input_dim = None

    def build_model(
        self,
        input_dim
    ):
        self.input_dim = input_dim

        self.model = SimpleGNNEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        return self.model

    def smiles_to_graphs(
        self,
        smiles_list
    ):
        graphs = batch_pyg_graphs(
            smiles_list
        )

        return graphs

    def fit(
        self,
        smiles_list
    ):
        """
        For Version 1, no supervised training.

        The GNN is randomly initialized and used as
        a structural embedding extractor.

        Later this can be replaced with:
            - contrastive pretraining
            - autoencoder reconstruction
            - property prediction pretraining
        """

        graphs = self.smiles_to_graphs(
            smiles_list
        )

        if len(graphs) == 0:
            raise ValueError(
                "No valid molecular graphs found."
            )

        input_dim = graphs[0].x.shape[1]

        self.build_model(
            input_dim
        )

        return self

    def transform(
        self,
        smiles_list
    ):
        """
        Generate embeddings for molecules.
        """

        if self.model is None:
            self.fit(
                smiles_list
            )

        graphs = self.smiles_to_graphs(
            smiles_list
        )

        loader = DataLoader(
            graphs,
            batch_size=self.batch_size,
            shuffle=False
        )

        embeddings = []

        self.model.eval()

        with torch.no_grad():

            for batch in loader:

                batch = batch.to(
                    self.device
                )

                emb = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.batch
                )

                embeddings.append(
                    emb.cpu().numpy()
                )

        if len(embeddings) == 0:
            return np.empty(
                (
                    0,
                    self.embedding_dim
                )
            )

        return np.vstack(
            embeddings
        )

    def fit_transform(
        self,
        smiles_list
    ):
        self.fit(
            smiles_list
        )

        return self.transform(
            smiles_list
        )

    def save(
        self,
        filepath
    ):
        if self.model is None:
            raise RuntimeError(
                "Model has not been initialized."
            )

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "input_dim":
                    self.input_dim,

                "hidden_dim":
                    self.hidden_dim,

                "embedding_dim":
                    self.embedding_dim
            },
            filepath
        )

    def load(
        self,
        filepath
    ):
        checkpoint = torch.load(
            filepath,
            map_location=self.device
        )

        self.hidden_dim = checkpoint[
            "hidden_dim"
        ]

        self.embedding_dim = checkpoint[
            "embedding_dim"
        ]

        self.input_dim = checkpoint[
            "input_dim"
        ]

        self.build_model(
            self.input_dim
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        return self

    def summary(
        self
    ):
        return {
            "method": "GNN Embedding",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_initialized": self.model is not None
        }