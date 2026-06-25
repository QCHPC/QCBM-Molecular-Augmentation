from qiskit import QuantumCircuit


class HardwareQCBMCircuit:
    """
    Hardware-aware QCBM circuit.

    Uses a connected path of physical qubits
    discovered from backend topology.

    Example:

        physical_path = [32,33,34,35,36]

    Logical:
        q0-q1-q2-q3-q4

    Physical:
        32-33-34-35-36
    """

    def __init__(
        self,
        physical_qubits,
        n_layers=2,
        measure=True
    ):

        self.physical_qubits = (
            list(physical_qubits)
        )

        self.n_qubits = len(
            self.physical_qubits
        )

        self.n_layers = n_layers

        self.measure = measure

        self.n_parameters = (
            self.n_layers
            * self.n_qubits
            * 2
        )

    def parameter_count(
        self
    ):

        return self.n_parameters

    def build(
        self,
        theta
    ):
        """
        Build hardware-ready QCBM.

        Parameters:

            theta shape =
            n_layers*n_qubits*2
        """

        qc = QuantumCircuit(
            self.n_qubits,
            self.n_qubits if self.measure else 0
        )

        idx = 0

        for _ in range(
            self.n_layers
        ):

            #
            # Single-qubit rotations
            #

            for q in range(
                self.n_qubits
            ):

                qc.ry(
                    theta[idx],
                    q
                )
                idx += 1

                qc.rz(
                    theta[idx],
                    q
                )
                idx += 1

            #
            # Hardware-native CZ chain
            #

            for q in range(
                self.n_qubits - 1
            ):

                qc.cz(
                    q,
                    q + 1
                )

        if self.measure:

            qc.measure(
                range(self.n_qubits),
                range(self.n_qubits)
            )

        return qc

    def build_unmeasured(
        self,
        theta
    ):
        """
        Statevector-compatible version.
        """

        qc = QuantumCircuit(
            self.n_qubits
        )

        idx = 0

        for _ in range(
            self.n_layers
        ):

            for q in range(
                self.n_qubits
            ):

                qc.ry(
                    theta[idx],
                    q
                )
                idx += 1

                qc.rz(
                    theta[idx],
                    q
                )
                idx += 1

            for q in range(
                self.n_qubits - 1
            ):

                qc.cz(
                    q,
                    q + 1
                )

        return qc

    def logical_to_physical_map(
        self
    ):
        """
        Return mapping.

        Example:

        {
            0:32,
            1:33,
            2:34,
            3:35,
            4:36
        }
        """

        return {
            logical: physical
            for logical, physical
            in enumerate(
                self.physical_qubits
            )
        }

    def summary(
        self
    ):

        return {
            "n_qubits":
                self.n_qubits,

            "n_layers":
                self.n_layers,

            "parameters":
                self.n_parameters,

            "physical_qubits":
                self.physical_qubits,

            "measurement":
                self.measure
        }