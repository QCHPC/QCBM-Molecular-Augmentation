import networkx as nx


class HardwareTopology:
    """
    Utilities for IBM backend topology.

    Used for:

    - Extracting coupling maps
    - Finding connected paths
    - Selecting qubits for QCBM
    """

    def __init__(
        self,
        backend
    ):
        self.backend = backend

    def coupling_map(
        self
    ):
        """
        Return backend coupling map.
        """

        try:

            return (
                self.backend
                .configuration()
                .coupling_map
            )

        except Exception:

            return (
                self.backend
                .coupling_map
                .get_edges()
            )

    def edges(
        self
    ):
        """
        Return list of edges.
        """

        return list(
            self.coupling_map()
        )

    def graph(
        self
    ):
        """
        Convert coupling map
        into NetworkX graph.
        """

        G = nx.Graph()

        G.add_edges_from(
            self.edges()
        )

        return G

    def num_qubits(
        self
    ):
        """
        Backend qubit count.
        """

        try:

            return (
                self.backend
                .configuration()
                .num_qubits
            )

        except Exception:

            return (
                self.backend
                .num_qubits
            )

    def connected_components(
        self
    ):
        """
        Connected subgraphs.
        """

        G = self.graph()

        return list(
            nx.connected_components(
                G
            )
        )

    def largest_component(
        self
    ):
        """
        Largest connected region.
        """

        components = (
            self.connected_components()
        )

        return max(
            components,
            key=len
        )

    def find_linear_path(
        self,
        path_length=5
    ):
        """
        Find connected chain of qubits.

        Example:

        [10,11,12,13,14]
        """

        G = self.graph()

        for start in G.nodes():

            stack = [
                (
                    start,
                    [start]
                )
            ]

            while stack:

                node, path = (
                    stack.pop()
                )

                if (
                    len(path)
                    == path_length
                ):
                    return path

                for neighbor in (
                    G.neighbors(node)
                ):

                    if (
                        neighbor
                        not in path
                    ):

                        stack.append(
                            (
                                neighbor,
                                path + [
                                    neighbor
                                ]
                            )
                        )

        raise ValueError(
            f"No path of length {path_length} found."
        )

    def find_multiple_paths(
        self,
        path_length=5,
        max_paths=10
    ):
        """
        Return several valid paths.
        """

        G = self.graph()

        paths = []

        for start in G.nodes():

            stack = [
                (
                    start,
                    [start]
                )
            ]

            while stack:

                node, path = (
                    stack.pop()
                )

                if (
                    len(path)
                    == path_length
                ):

                    paths.append(
                        path
                    )

                    if (
                        len(paths)
                        >= max_paths
                    ):
                        return paths

                for neighbor in (
                    G.neighbors(node)
                ):

                    if (
                        neighbor
                        not in path
                    ):

                        stack.append(
                            (
                                neighbor,
                                path + [
                                    neighbor
                                ]
                            )
                        )

        return paths

    def best_qcbm_path(
        self,
        n_qubits=5
    ):
        """
        Main helper used by QCBM.

        Returns one connected chain.
        """

        return self.find_linear_path(
            path_length=n_qubits
        )

    def summary(
        self
    ):
        """
        Topology statistics.
        """

        return {
            "backend":
                self.backend.name,

            "num_qubits":
                self.num_qubits(),

            "num_edges":
                len(
                    self.edges()
                ),

            "connected_components":
                len(
                    self.connected_components()
                )
        }