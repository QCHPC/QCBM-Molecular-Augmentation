# losses placeholder
import numpy as np
from scipy.stats import wasserstein_distance


def normalize(
    p
):
    """
    Normalize probability vector.
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


def kl_loss(
    target,
    generated
):
    """
    KL(target || generated)
    """

    target = normalize(
        target
    )

    generated = normalize(
        generated
    )

    return float(
        np.sum(
            target *
            np.log(
                target / generated
            )
        )
    )


def total_variation_loss(
    target,
    generated
):
    """
    TV distance.
    """

    target = normalize(
        target
    )

    generated = normalize(
        generated
    )

    return float(
        0.5 *
        np.sum(
            np.abs(
                target -
                generated
            )
        )
    )


def fidelity_score(
    target,
    generated
):
    """
    Bhattacharyya fidelity.
    """

    target = normalize(
        target
    )

    generated = normalize(
        generated
    )

    return float(
        (
            np.sum(
                np.sqrt(
                    target *
                    generated
                )
            )
        ) ** 2
    )


def fidelity_loss(
    target,
    generated
):
    """
    Optimization version.

    Minimize:
        1 - fidelity
    """

    return (
        1.0 -
        fidelity_score(
            target,
            generated
        )
    )


def mmd_loss(
    target,
    generated,
    gammas=(0.1, 1, 10)
):
    """
    Multi-kernel MMD.
    """

    target = normalize(
        target
    )

    generated = normalize(
        generated
    )

    x = np.arange(
        len(target)
    ).reshape(-1, 1)

    kernel = np.zeros(
        (
            len(target),
            len(target)
        )
    )

    for gamma in gammas:

        dist2 = (
            x - x.T
        ) ** 2

        kernel += np.exp(
            -gamma * dist2
        )

    kernel /= len(
        gammas
    )

    diff = (
        target -
        generated
    )

    return float(
        diff.T @ kernel @ diff
    )


def sinkhorn_loss(
    target,
    generated
):
    """
    1D Wasserstein approximation.
    """

    target = normalize(
        target
    )

    generated = normalize(
        generated
    )

    x = np.arange(
        len(target)
    )

    return float(
        wasserstein_distance(
            x,
            x,
            target,
            generated
        )
    )


def qcbm_loss(
    target,
    generated,
    metric="mmd"
):
    """
    Main optimization loss.

    Supported:

    mmd
    kl
    tv
    fidelity
    sinkhorn
    """

    metric = metric.lower()

    if metric == "mmd":

        return mmd_loss(
            target,
            generated
        )

    elif metric == "kl":

        return kl_loss(
            target,
            generated
        )

    elif metric == "tv":

        return total_variation_loss(
            target,
            generated
        )

    elif metric == "fidelity":

        return fidelity_loss(
            target,
            generated
        )

    elif metric == "sinkhorn":

        return sinkhorn_loss(
            target,
            generated
        )

    else:

        raise ValueError(
            f"Unknown metric: {metric}"
        )


def benchmark_losses(
    target,
    generated
):
    """
    Compute all metrics.

    Used for benchmark tables.
    """

    return {
        "KL":
            kl_loss(
                target,
                generated
            ),

        "TV":
            total_variation_loss(
                target,
                generated
            ),

        "Fidelity":
            fidelity_score(
                target,
                generated
            ),

        "MMD":
            mmd_loss(
                target,
                generated
            ),

        "Sinkhorn":
            sinkhorn_loss(
                target,
                generated
            )
    }