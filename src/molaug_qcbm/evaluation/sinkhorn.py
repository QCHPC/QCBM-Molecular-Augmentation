# sinkhorn placeholder
from scipy.stats import wasserstein_distance
from .kl_divergence import normalize
import numpy as np


def sinkhorn_1d(
    p,
    q
):

    x = np.arange(len(p))

    return float(
        wasserstein_distance(
            x,
            x,
            normalize(p),
            normalize(q)
        )
    )