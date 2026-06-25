# classical born placeholder
import numpy as np


def int_to_bits(
    x,
    n_bits
):
    """
    Convert integer state to binary vector.

    Example:
        int_to_bits(3, 5) -> [0, 0, 0, 1, 1]
    """

    return np.array(
        list(
            format(
                int(x),
                f"0{n_bits}b"
            )
        ),
        dtype=int
    )


def bits_to_int(
    bits
):
    """
    Convert binary vector to integer state.

    Example:
        bits_to_int([0, 0, 0, 1, 1]) -> 3
    """

    return int(
        "".join(
            str(int(b))
            for b in bits
        ),
        2
    )


def normalize_distribution(
    p
):
    """
    Normalize probability distribution.
    """

    p = np.asarray(
        p,
        dtype=float
    )

    p = np.clip(
        p,
        1e-12,
        None
    )

    return p / p.sum()


class ClassicalBornMachine:
    """
    Independent Bernoulli Born Machine.

    This is the simplest classical Born baseline.

    Assumption:
        Each bit is independent.

    Distribution:
        P(x) = Π P(x_i)
    """

    def __init__(
        self,
        n_bits
    ):
        self.n_bits = n_bits
        self.bit_probs = None

    def fit(
        self,
        target_distribution
    ):
        """
        Estimate independent bit probabilities
        from target distribution over 2^n states.
        """

        p_data = normalize_distribution(
            target_distribution
        )

        n_states = 2 ** self.n_bits

        if len(p_data) != n_states:
            raise ValueError(
                f"Expected distribution of length {n_states}, "
                f"got {len(p_data)}"
            )

        bit_probs = np.zeros(
            self.n_bits,
            dtype=float
        )

        for state in range(n_states):

            bits = int_to_bits(
                state,
                self.n_bits
            )

            bit_probs += (
                p_data[state] * bits
            )

        self.bit_probs = np.clip(
            bit_probs,
            1e-6,
            1 - 1e-6
        )

        return self

    def sample_states(
        self,
        n_samples,
        random_state=None
    ):
        """
        Sample integer states.
        """

        if self.bit_probs is None:
            raise RuntimeError(
                "Model must be fitted before sampling."
            )

        rng = np.random.default_rng(
            random_state
        )

        samples = []

        for _ in range(n_samples):

            bits = rng.binomial(
                1,
                self.bit_probs,
                size=self.n_bits
            )

            samples.append(
                bits_to_int(bits)
            )

        return np.asarray(
            samples,
            dtype=int
        )

    def probabilities(
        self
    ):
        """
        Compute full probability distribution
        over all 2^n states.
        """

        if self.bit_probs is None:
            raise RuntimeError(
                "Model must be fitted before probability computation."
            )

        n_states = 2 ** self.n_bits
        probs = np.zeros(
            n_states,
            dtype=float
        )

        for state in range(n_states):

            bits = int_to_bits(
                state,
                self.n_bits
            )

            prob = 1.0

            for bit, p_bit in zip(
                bits,
                self.bit_probs
            ):

                if bit == 1:
                    prob *= p_bit
                else:
                    prob *= (
                        1 - p_bit
                    )

            probs[state] = prob

        return normalize_distribution(
            probs
        )

    def sample_distribution(
        self,
        n_samples,
        random_state=None
    ):
        """
        Sample states and return empirical distribution.
        """

        sampled_states = self.sample_states(
            n_samples=n_samples,
            random_state=random_state
        )

        counts = np.bincount(
            sampled_states,
            minlength=2 ** self.n_bits
        ).astype(float)

        return normalize_distribution(
            counts
        )