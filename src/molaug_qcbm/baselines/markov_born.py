# markov born placeholder
import numpy as np


def int_to_bits(
    x,
    n_bits
):
    """
    Convert integer state to binary vector.

    Example:
        5 -> [0,0,1,0,1]
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


class MarkovBornMachine:
    """
    Markov Chain Born Machine.

    Distribution:

    P(x) =
        P(x1)
        P(x2|x1)
        P(x3|x2)
        ...
        P(xn|xn-1)

    Stronger than Independent Bernoulli Born Machine
    because local bit correlations are learned.
    """

    def __init__(
        self,
        n_bits
    ):

        self.n_bits = n_bits

        self.p_first = None

        self.transitions = None

    def fit(
        self,
        target_distribution
    ):
        """
        Learn Markov probabilities
        from target distribution.
        """

        p_data = normalize_distribution(
            target_distribution
        )

        n_states = 2 ** self.n_bits

        if len(p_data) != n_states:

            raise ValueError(
                f"Expected {n_states} states"
            )

        first_bit_counts = np.zeros(
            2
        )

        transition_counts = np.zeros(
            (
                self.n_bits - 1,
                2,
                2
            )
        )

        for state in range(n_states):

            bits = int_to_bits(
                state,
                self.n_bits
            )

            weight = p_data[state]

            first_bit_counts[
                bits[0]
            ] += weight

            for i in range(
                self.n_bits - 1
            ):

                prev_bit = bits[i]
                next_bit = bits[i + 1]

                transition_counts[
                    i,
                    prev_bit,
                    next_bit
                ] += weight

        self.p_first = (
            first_bit_counts
            /
            first_bit_counts.sum()
        )

        self.transitions = np.zeros_like(
            transition_counts
        )

        for i in range(
            self.n_bits - 1
        ):

            for prev_bit in [0, 1]:

                row_sum = (
                    transition_counts[
                        i,
                        prev_bit
                    ].sum()
                )

                if row_sum == 0:

                    self.transitions[
                        i,
                        prev_bit
                    ] = np.array(
                        [0.5, 0.5]
                    )

                else:

                    self.transitions[
                        i,
                        prev_bit
                    ] = (
                        transition_counts[
                            i,
                            prev_bit
                        ]
                        /
                        row_sum
                    )

        return self

    def state_probability(
        self,
        bits
    ):
        """
        Probability of a bitstring.
        """

        prob = self.p_first[
            bits[0]
        ]

        for i in range(
            self.n_bits - 1
        ):

            prob *= (
                self.transitions[
                    i,
                    bits[i],
                    bits[i + 1]
                ]
            )

        return prob

    def probabilities(
        self
    ):
        """
        Full distribution over all states.
        """

        n_states = (
            2 ** self.n_bits
        )

        probs = np.zeros(
            n_states
        )

        for state in range(
            n_states
        ):

            bits = int_to_bits(
                state,
                self.n_bits
            )

            probs[state] = (
                self.state_probability(
                    bits
                )
            )

        return normalize_distribution(
            probs
        )

    def sample_states(
        self,
        n_samples,
        random_state=None
    ):
        """
        Generate integer states.
        """

        rng = np.random.default_rng(
            random_state
        )

        samples = []

        for _ in range(
            n_samples
        ):

            bits = np.zeros(
                self.n_bits,
                dtype=int
            )

            bits[0] = rng.choice(
                [0, 1],
                p=self.p_first
            )

            for i in range(
                self.n_bits - 1
            ):

                bits[
                    i + 1
                ] = rng.choice(
                    [0, 1],
                    p=self.transitions[
                        i,
                        bits[i]
                    ]
                )

            samples.append(
                bits_to_int(bits)
            )

        return np.asarray(
            samples,
            dtype=int
        )

    def sample_distribution(
        self,
        n_samples,
        random_state=None
    ):
        """
        Empirical distribution
        from generated samples.
        """

        states = self.sample_states(
            n_samples=n_samples,
            random_state=random_state
        )

        counts = np.bincount(
            states,
            minlength=2 ** self.n_bits
        )

        return normalize_distribution(
            counts
        )

    def transition_matrix(
        self
    ):
        """
        Inspect learned transitions.
        """

        return self.transitions

    def summary(
        self
    ):

        return {
            "n_bits": self.n_bits,
            "p_first": self.p_first,
            "transitions_shape":
                self.transitions.shape
        }