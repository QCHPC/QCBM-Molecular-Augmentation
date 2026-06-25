# sampler placeholder
import numpy as np


class QCBMSampler:
    """
    Sampling utilities for trained QCBM.

    Supports:

    - State sampling
    - Bitstring sampling
    - Empirical distribution generation
    """

    def __init__(
        self,
        probability_distribution
    ):

        self.probabilities = (
            np.asarray(
                probability_distribution,
                dtype=float
            )
        )

        self.probabilities = (
            self.probabilities
            /
            self.probabilities.sum()
        )

        self.n_states = len(
            self.probabilities
        )

        self.n_qubits = int(
            np.log2(
                self.n_states
            )
        )

    def sample_states(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample integer states.

        Example:
            [3,7,1,4,4,2]
        """

        rng = np.random.default_rng(
            random_state
        )

        return rng.choice(
            np.arange(
                self.n_states
            ),
            size=n_samples,
            p=self.probabilities
        )

    def sample_bitstrings(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample bitstrings.

        Example:
            ['00011','11100']
        """

        states = self.sample_states(
            n_samples=n_samples,
            random_state=random_state
        )

        return [
            format(
                state,
                f"0{self.n_qubits}b"
            )
            for state in states
        ]

    def empirical_distribution(
        self,
        n_samples=10000,
        random_state=None
    ):
        """
        Generate empirical distribution
        from sampled states.
        """

        states = self.sample_states(
            n_samples=n_samples,
            random_state=random_state
        )

        counts = np.bincount(
            states,
            minlength=self.n_states
        )

        return (
            counts /
            counts.sum()
        )

    def state_counts(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Return counts dictionary.

        Example:
        {
            0:123,
            1:55,
            ...
        }
        """

        states = self.sample_states(
            n_samples=n_samples,
            random_state=random_state
        )

        unique, counts = np.unique(
            states,
            return_counts=True
        )

        return dict(
            zip(
                unique.tolist(),
                counts.tolist()
            )
        )

    def bitstring_counts(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Return bitstring counts.

        Example:
        {
            '00000':120,
            '00001':33
        }
        """

        bitstrings = (
            self.sample_bitstrings(
                n_samples=n_samples,
                random_state=random_state
            )
        )

        counts = {}

        for b in bitstrings:

            counts[b] = (
                counts.get(
                    b,
                    0
                )
                + 1
            )

        return counts

    def most_probable_states(
        self,
        top_k=10
    ):
        """
        Return top states.
        """

        idx = np.argsort(
            self.probabilities
        )[::-1]

        idx = idx[:top_k]

        return [
            (
                int(state),
                float(
                    self.probabilities[
                        state
                    ]
                )
            )
            for state in idx
        ]

    def summary(
        self
    ):

        return {
            "n_qubits":
                self.n_qubits,

            "n_states":
                self.n_states,

            "max_probability":
                float(
                    np.max(
                        self.probabilities
                    )
                ),

            "min_probability":
                float(
                    np.min(
                        self.probabilities
                    )
                )
        }