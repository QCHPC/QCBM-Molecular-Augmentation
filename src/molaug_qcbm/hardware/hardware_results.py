import numpy as np
import pandas as pd

from molaug_qcbm.evaluation.benchmark_metrics import (
    distribution_report
)


class HardwareResults:
    """
    Utilities for processing
    IBM Quantum hardware outputs.
    """

    @staticmethod
    def counts_to_distribution(
        counts,
        n_qubits
    ):
        """
        Convert counts dictionary
        to probability distribution.

        Example:

        {
            '00000':1988,
            '11111':2022
        }
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

        for bitstring, count in (
            counts.items()
        ):

            state = int(
                bitstring,
                2
            )

            probs[state] = (
                count /
                total_shots
            )

        return probs

    @staticmethod
    def distribution_to_counts(
        probabilities,
        shots=4096
    ):
        """
        Synthetic counts from probabilities.
        Useful for debugging.
        """

        probabilities = (
            np.asarray(
                probabilities,
                dtype=float
            )
        )

        probabilities = (
            probabilities /
            probabilities.sum()
        )

        counts = {}

        n_qubits = int(
            np.log2(
                len(probabilities)
            )
        )

        for idx, prob in enumerate(
            probabilities
        ):

            count = int(
                round(
                    prob * shots
                )
            )

            if count > 0:

                counts[
                    format(
                        idx,
                        f"0{n_qubits}b"
                    )
                ] = count

        return counts

    @staticmethod
    def compare_to_target(
        target_distribution,
        hardware_distribution,
        model_name="Hardware QCBM"
    ):
        """
        Benchmark hardware
        distribution against target.
        """

        return distribution_report(
            model_name,
            hardware_distribution,
            target_distribution
        )

    @staticmethod
    def compare_multiple(
        target_distribution,
        distributions
    ):
        """
        distributions:

        {
            "Ideal QCBM": probs1,
            "Hardware QCBM": probs2,
            "Born": probs3
        }
        """

        rows = []

        for name, probs in (
            distributions.items()
        ):

            row = distribution_report(
                name,
                probs,
                target_distribution
            )

            rows.append(row)

        return pd.DataFrame(
            rows
        )

    @staticmethod
    def hardware_vs_statevector(
        hardware_distribution,
        ideal_distribution
    ):
        """
        Compare hardware output
        against ideal simulation.
        """

        return distribution_report(
            "Hardware vs Ideal",
            hardware_distribution,
            ideal_distribution
        )

    @staticmethod
    def top_states(
        probabilities,
        top_k=10
    ):
        """
        Most probable states.
        """

        probabilities = (
            np.asarray(
                probabilities
            )
        )

        idx = np.argsort(
            probabilities
        )[::-1]

        idx = idx[:top_k]

        results = []

        n_qubits = int(
            np.log2(
                len(probabilities)
            )
        )

        for state in idx:

            results.append(
                {
                    "state":
                        int(state),

                    "bitstring":
                        format(
                            state,
                            f"0{n_qubits}b"
                        ),

                    "probability":
                        float(
                            probabilities[
                                state
                            ]
                        )
                }
            )

        return pd.DataFrame(
            results
        )

    @staticmethod
    def benchmark_table(
        target_distribution,
        ideal_qcbm,
        hardware_qcbm,
        classical_born=None,
        markov_born=None
    ):
        """
        Final paper-ready table.
        """

        models = {
            "Ideal QCBM":
                ideal_qcbm,

            "Hardware QCBM":
                hardware_qcbm
        }

        if classical_born is not None:

            models[
                "Classical Born"
            ] = classical_born

        if markov_born is not None:

            models[
                "Markov Born"
            ] = markov_born

        rows = []

        for name, probs in (
            models.items()
        ):

            row = distribution_report(
                name,
                probs,
                target_distribution
            )

            rows.append(row)

        return pd.DataFrame(
            rows
        )

    @staticmethod
    def save_results(
        dataframe,
        filepath
    ):
        """
        Save benchmark table.
        """

        dataframe.to_csv(
            filepath,
            index=False
        )

        return filepath