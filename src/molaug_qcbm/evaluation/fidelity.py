# fidelity placeholder
import numpy as np
from .kl_divergence import normalize


def fidelity(p, q):

    p = normalize(p)
    q = normalize(q)

    return float(
        np.sum(
            np.sqrt(p * q)
        ) ** 2
    )