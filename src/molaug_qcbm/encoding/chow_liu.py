# chow liu encoding placeholder
import numpy as np
import networkx as nx

from sklearn.metrics import mutual_info_score


class ChowLiuEncoder:
    """
    Chow-Liu Tree Encoder

    Learns dependency structure
    between latent dimensions.

    Pipeline:

    SELFIES
        ↓
    Token Count Vectors
        ↓
    Mutual Information
        ↓
    Maximum Spanning Tree
        ↓
    Structured Features
    """

    def __init__(
        self
    ):

        self.mi_matrix = None

        self.tree = None

        self.edges = None

    def compute_mutual_information(
        self,
        X
    ):
        """
        Compute pairwise mutual information.

        X shape:
            (n_samples, n_features)
        """

        n_features = X.shape[1]

        mi = np.zeros(
            (
                n_features,
                n_features
            )
        )

        for i in range(
            n_features
        ):

            for j in range(
                i,
                n_features
            ):

                score = mutual_info_score(
                    X[:, i],
                    X[:, j]
                )

                mi[i, j] = score
                mi[j, i] = score

        self.mi_matrix = mi

        return mi

    def build_tree(
        self,
        X
    ):
        """
        Build maximum spanning tree.
        """

        mi = self.compute_mutual_information(
            X
        )

        G = nx.Graph()

        n_features = X.shape[1]

        for i in range(
            n_features
        ):

            for j in range(
                i + 1,
                n_features
            ):

                G.add_edge(
                    i,
                    j,
                    weight=mi[i, j]
                )

        tree = nx.maximum_spanning_tree(
            G
        )

        self.tree = tree

        self.edges = list(
            tree.edges()
        )

        return tree

    def fit(
        self,
        X
    ):
        """
        Learn Chow-Liu tree.
        """

        self.build_tree(
            X
        )

        return self

    def transform(
        self,
        X
    ):
        """
        Generate dependency features.

        For each tree edge:

            feature_i * feature_j

        becomes a new feature.
        """

        if self.tree is None:

            raise RuntimeError(
                "Call fit() first."
            )

        features = []

        for u, v in self.edges:

            interaction = (
                X[:, u]
                *
                X[:, v]
            )

            features.append(
                interaction.reshape(
                    -1,
                    1
                )
            )

        if len(features) == 0:

            return X

        interaction_features = (
            np.hstack(features)
        )

        return np.hstack(
            [
                X,
                interaction_features
            ]
        )

    def fit_transform(
        self,
        X
    ):

        self.fit(X)

        return self.transform(X)

    def adjacency_matrix(
        self
    ):
        """
        Tree adjacency matrix.
        """

        if self.tree is None:

            raise RuntimeError(
                "Call fit() first."
            )

        return nx.to_numpy_array(
            self.tree
        )

    def get_edges(
        self
    ):
        """
        Return tree edges.
        """

        if self.edges is None:

            raise RuntimeError(
                "Call fit() first."
            )

        return self.edges

    def summary(
        self
    ):

        if self.tree is None:

            return {
                "fitted": False
            }

        return {

            "fitted": True,

            "nodes":
                self.tree.number_of_nodes(),

            "edges":
                self.tree.number_of_edges()
        }