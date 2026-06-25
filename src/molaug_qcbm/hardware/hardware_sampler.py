import numpy as np

from qiskit import transpile

from qiskit_ibm_runtime import (
    SamplerV2
)


class HardwareSampler:
    """
    Execute QCBM circuits on real IBM hardware.

    Uses:

    - SamplerV2
    - Runtime backend
    - HardwareQCBMCircuit
    """

    def __init__(
        self,
        backend
    ):

        self.backend = backend

    def run(
        self,
        circuit,
        shots=4096
    ):
        """
        Execute circuit on IBM hardware.

        Returns:
            Runtime job object
        """

        transpiled = transpile(
            circuit,
            backend=self.backend,
            optimization_level=1
        )

        sampler = SamplerV2(
            mode=self.backend
        )

        job = sampler.run(
            [transpiled],
            shots=shots
        )

        return job

    def run_and_wait(
        self,
        circuit,
        shots=4096
    ):
        """
        Submit job and return result.
        """

        job = self.run(
            circuit=circuit,
            shots=shots
        )

        result = job.result()

        return result

    def extract_counts(
        self,
        result
    ):
        """
        Extract bitstring counts.

        Runtime API can vary slightly
        between versions.
        """

        pub_result = result[0]

        try:

            counts = (
                pub_result
                .data
                .meas
                .get_counts()
            )

            return counts

        except Exception:

            try:

                return (
                    pub_result
                    .data
                    .c
                    .get_counts()
                )

            except Exception as e:

                raise RuntimeError(
                    f"Unable to extract counts: {e}"
                )

    def counts_to_probabilities(
        self,
        counts,
        n_qubits
    ):
        """
        Convert hardware counts
        into probability vector.
        """

        n_states = (
            2 ** n_qubits
        )

        probs = np.zeros(
            n_states,
            dtype=float
        )

        total_shots = sum(
            counts.values()
        )

        for bitstring, count in counts.items():

            state = int(
                bitstring,
                2
            )

            probs[state] = (
                count /
                total_shots
            )

        return probs

    def execute_distribution(
        self,
        circuit,
        n_qubits,
        shots=4096
    ):
        """
        Full hardware pipeline.

        Circuit
            ↓
        Runtime
            ↓
        Counts
            ↓
        Distribution
        """

        result = self.run_and_wait(
            circuit,
            shots=shots
        )

        counts = (
            self.extract_counts(
                result
            )
        )

        probabilities = (
            self.counts_to_probabilities(
                counts,
                n_qubits
            )
        )

        return {
            "counts": counts,
            "probabilities": probabilities,
            "shots": shots
        }