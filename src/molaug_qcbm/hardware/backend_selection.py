from qiskit_ibm_runtime import (
    QiskitRuntimeService
)


class IBMBackendSelector:
    """
    IBM Quantum backend utilities.

    Supports:

    - Backend discovery
    - Backend filtering
    - Least busy backend selection
    """

    def __init__(
        self,
        channel="ibm_quantum"
    ):

        self.service = (
            QiskitRuntimeService(
                channel=channel
            )
        )

    def list_backends(
        self
    ):
        """
        Return all available backends.
        """

        backends = (
            self.service.backends()
        )

        return sorted(
            [
                backend.name
                for backend in backends
            ]
        )

    def operational_backends(
        self
    ):
        """
        Only operational hardware.
        """

        backends = (
            self.service.backends(
                operational=True,
                simulator=False
            )
        )

        return sorted(
            [
                backend.name
                for backend in backends
            ]
        )

    def backend_info(
        self,
        backend_name
    ):
        """
        Detailed backend metadata.
        """

        backend = (
            self.service.backend(
                backend_name
            )
        )

        config = backend.configuration()

        return {
            "name":
                backend.name,

            "num_qubits":
                config.num_qubits,

            "basis_gates":
                config.basis_gates,

            "coupling_map":
                config.coupling_map
        }

    def get_backend(
        self,
        backend_name
    ):
        """
        Return backend object.
        """

        return self.service.backend(
            backend_name
        )

    def suitable_backends(
        self,
        min_qubits=5
    ):
        """
        Hardware with enough qubits.
        """

        candidates = []

        for backend in self.service.backends(
            operational=True,
            simulator=False
        ):

            try:

                n_qubits = (
                    backend.configuration()
                    .num_qubits
                )

                if n_qubits >= min_qubits:

                    candidates.append(
                        {
                            "name":
                                backend.name,

                            "num_qubits":
                                n_qubits
                        }
                    )

            except Exception:
                pass

        return candidates

    def best_backend(
        self,
        min_qubits=5
    ):
        """
        Choose least busy backend.
        """

        candidates = []

        for backend in self.service.backends(
            operational=True,
            simulator=False
        ):

            try:

                if (
                    backend.configuration()
                    .num_qubits
                    >= min_qubits
                ):

                    pending_jobs = (
                        backend.status()
                        .pending_jobs
                    )

                    candidates.append(
                        (
                            backend,
                            pending_jobs
                        )
                    )

            except Exception:
                pass

        if len(candidates) == 0:

            raise RuntimeError(
                "No suitable backend found."
            )

        candidates.sort(
            key=lambda x: x[1]
        )

        return candidates[0][0]

    def print_backend_summary(
        self,
        min_qubits=5
    ):
        """
        Console summary.
        """

        backends = (
            self.suitable_backends(
                min_qubits=min_qubits
            )
        )

        print(
            "\nAvailable IBM Backends\n"
        )

        for backend in backends:

            print(
                f"{backend['name']} "
                f"({backend['num_qubits']} qubits)"
            )