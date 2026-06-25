# circuit placeholder
from qiskit import QuantumCircuit


class QCBMCircuit:
    """
    Generic Quantum Circuit Born Machine.

    Supports:

    - Statevector simulation
    - Aer simulation
    - IBM Hardware

    Parameters are supplied externally.
    """

    def __init__(
        self,
        n_qubits,
        n_layers=2,
        entanglement="linear"
    ):

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement

        self.n_parameters = (
            n_layers *
            n_qubits *
            2
        )

    def build(
        self,
        theta
    ):
        """
        Build QCBM circuit.

        theta shape:

        n_layers * n_qubits * 2
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

            self._apply_entanglement(
                qc
            )

        return qc

    def _apply_entanglement(
        self,
        qc
    ):

        if (
            self.entanglement
            == "linear"
        ):

            for q in range(
                self.n_qubits - 1
            ):

                qc.cz(
                    q,
                    q + 1
                )

        elif (
            self.entanglement
            == "circular"
        ):

            for q in range(
                self.n_qubits - 1
            ):

                qc.cz(
                    q,
                    q + 1
                )

            qc.cz(
                self.n_qubits - 1,
                0
            )

        elif (
            self.entanglement
            == "full"
        ):

            for i in range(
                self.n_qubits
            ):

                for j in range(
                    i + 1,
                    self.n_qubits
                ):

                    qc.cz(
                        i,
                        j
                    )

        else:

            raise ValueError(
                f"Unknown entanglement: {self.entanglement}"
            )

    def parameter_count(
        self
    ):

        return self.n_parameters

class HardwareQCBMCircuit:
    """
    Hardware-aware QCBM.

    Uses physical qubit path.
    """

    def __init__(
        self,
        physical_qubits
    ):

        self.physical_qubits = (
            list(
                physical_qubits
            )
        )

        self.n_qubits = len(
            physical_qubits
        )

    def build(
        self,
        theta
    ):

        qc = QuantumCircuit(
            self.n_qubits
        )

        idx = 0

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

        for i in range(
            self.n_qubits - 1
        ):

            qc.cz(
                i,
                i + 1
            )

        return qc