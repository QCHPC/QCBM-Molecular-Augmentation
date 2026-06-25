# benchmark metrics placeholder
from .kl_divergence import *
from .fidelity import *
from .mmd import *
from .sinkhorn import *


def total_variation(
    p,
    q
):

    p = normalize(p)
    q = normalize(q)

    return float(
        0.5 *
        abs(
            p - q
        ).sum()
    )


def distribution_report(
    name,
    p,
    target
):

    return {
        "method": name,
        "KL": kl_divergence(
            target,
            p
        ),
        "TV": total_variation(
            target,
            p
        ),
        "Fidelity": fidelity(
            target,
            p
        ),
        "MMD": mmd_rbf(
            target,
            p
        ),
        "Sinkhorn": sinkhorn_1d(
            target,
            p
        )
    }