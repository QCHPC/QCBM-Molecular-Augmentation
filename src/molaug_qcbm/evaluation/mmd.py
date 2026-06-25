# mmd placeholder
import numpy as np
from .kl_divergence import normalize


def mmd_rbf(
    p,
    q,
    gammas=(0.1, 1, 10)
):

    p = normalize(p)
    q = normalize(q)

    x = np.arange(
        len(p)
    ).reshape(-1, 1)

    K = np.zeros(
        (len(p), len(p))
    )

    for g in gammas:

        dist2 = (
            x - x.T
        ) ** 2

        K += np.exp(
            -g * dist2
        )

    K /= len(gammas)

    diff = p - q

    return float(
        diff.T @ K @ diff
    )