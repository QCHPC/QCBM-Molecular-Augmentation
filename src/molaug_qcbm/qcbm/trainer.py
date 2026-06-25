# trainer placeholder
import numpy as np

from scipy.optimize import minimize

from qiskit.quantum_info import Statevector

from molaug_qcbm.qcbm.losses import qcbm_loss


class QCBMTrainer:
    """
    Train QCBM using classical optimization.

    Supports:

    - Statevector training
    - MMD loss
    - KL loss
    - TV loss
    - Fidelity loss
    - Sinkhorn loss
    """

    def __init__(
        self,
        circuit_builder,
        loss_metric="mmd",
        optimizer="COBYLA",
        maxiter=200
    ):

        self.circuit_builder = (
            circuit_builder
        )

        self.loss_metric = (
            loss_metric
        )

        self.optimizer = (
            optimizer
        )

        self.maxiter = (
            maxiter
        )

        self.best_theta = None
        self.best_loss = None

        self.loss_history = []

    def statevector_probabilities(
        self,
        theta
    ):
        """
        Generate probability distribution
        from QCBM parameters.
        """

        qc = self.circuit_builder.build(
            theta
        )

        state = (
            Statevector.from_instruction(
                qc
            )
        )

        probs = np.abs(
            state.data
        ) ** 2

        return probs

    def objective(
        self,
        theta,
        target_distribution
    ):
        """
        Optimization objective.
        """

        generated_distribution = (
            self.statevector_probabilities(
                theta
            )
        )

        loss = qcbm_loss(
            target_distribution,
            generated_distribution,
            metric=self.loss_metric
        )

        self.loss_history.append(
            loss
        )

        return loss

    def fit(
        self,
        target_distribution,
        initial_theta=None
    ):
        """
        Train QCBM.
        """

        n_params = (
            self.circuit_builder
            .parameter_count()
        )

        if initial_theta is None:

            initial_theta = (
                np.random.uniform(
                    0,
                    2 * np.pi,
                    n_params
                )
            )

        result = minimize(
            fun=self.objective,
            x0=initial_theta,
            args=(
                target_distribution,
            ),
            method=self.optimizer,
            options={
                "maxiter":
                self.maxiter
            }
        )

        self.best_theta = (
            result.x
        )

        self.best_loss = (
            result.fun
        )

        return result

    def probabilities(
        self
    ):
        """
        Return learned distribution.
        """

        if self.best_theta is None:

            raise RuntimeError(
                "Train model first."
            )

        return (
            self.statevector_probabilities(
                self.best_theta
            )
        )

    def get_parameters(
        self
    ):

        return self.best_theta

    def get_loss_history(
        self
    ):

        return self.loss_history

    def summary(
        self
    ):

        return {
            "loss_metric":
                self.loss_metric,

            "optimizer":
                self.optimizer,

            "best_loss":
                self.best_loss,

            "iterations":
                len(
                    self.loss_history
                )
        }