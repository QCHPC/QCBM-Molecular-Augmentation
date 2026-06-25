# kl divergence placeholder
import numpy as np


def normalize(p):

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


def kl_divergence(p, q):

    p = normalize(p)
    q = normalize(q)

    return float(
        np.sum(
            p * np.log(p / q)
        )
    )