# qcbm core placeholder
import numpy as np

from molaug_qcbm.qcbm.circuit import QCBMCircuit
from molaug_qcbm.qcbm.trainer import QCBMTrainer
from molaug_qcbm.qcbm.sampler import QCBMSampler


class QCBM:
    """
    High-level Quantum Circuit Born Machine.

    Wraps:

    - circuit construction
    - training
    - probability generation
    - sampling

    Intended usage:

        model = QCBM(...)
        model.fit(target_distribution)
        probs = model.probabilities()
        samples = model.sample_states(...)
    """

    def __init__(
        self,
        n_qubits,
        n_layers=3,
        entanglement="linear",
        loss_metric="mmd",
        optimizer="COBYLA",
        maxiter=200
    ):

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.circuit_builder = (
            QCBMCircuit(
                n_qubits=n_qubits,
                n_layers=n_layers,
                entanglement=entanglement
            )
        )

        self.trainer = (
            QCBMTrainer(
                circuit_builder=self.circuit_builder,
                loss_metric=loss_metric,
                optimizer=optimizer,
                maxiter=maxiter
            )
        )

        self._trained = False

    def fit(
        self,
        target_distribution,
        initial_theta=None
    ):
        """
        Train QCBM.
        """

        result = self.trainer.fit(
            target_distribution=target_distribution,
            initial_theta=initial_theta
        )

        self._trained = True

        return result

    def probabilities(
        self
    ):
        """
        Learned probability distribution.
        """

        self._check_trained()

        return self.trainer.probabilities()

    def parameters(
        self
    ):
        """
        Learned circuit parameters.
        """

        self._check_trained()

        return self.trainer.get_parameters()

    def loss_history(
        self
    ):

        self._check_trained()

        return self.trainer.get_loss_history()

    def best_loss(
        self
    ):

        self._check_trained()

        return self.trainer.best_loss

    def sample_states(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample integer states.
        """

        self._check_trained()

        sampler = QCBMSampler(
            self.probabilities()
        )

        return sampler.sample_states(
            n_samples=n_samples,
            random_state=random_state
        )

    def sample_bitstrings(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        Sample bitstrings.
        """

        self._check_trained()

        sampler = QCBMSampler(
            self.probabilities()
        )

        return sampler.sample_bitstrings(
            n_samples=n_samples,
            random_state=random_state
        )

    def empirical_distribution(
        self,
        n_samples=10000,
        random_state=None
    ):
        """
        Sample-based distribution.
        """

        self._check_trained()

        sampler = QCBMSampler(
            self.probabilities()
        )

        return sampler.empirical_distribution(
            n_samples=n_samples,
            random_state=random_state
        )

    def state_counts(
        self,
        n_samples=1000,
        random_state=None
    ):
        """
        State frequency counts.
        """

        self._check_trained()

        sampler = QCBMSampler(
            self.probabilities()
        )

        return sampler.state_counts(
            n_samples=n_samples,
            random_state=random_state
        )

    def summary(
        self
    ):

        if not self._trained:

            return {
                "trained": False,
                "n_qubits": self.n_qubits,
                "n_layers": self.n_layers
            }

        return {
            "trained": True,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "best_loss": self.best_loss(),
            "parameters": len(
                self.parameters()
            )
        }

    def _check_trained(
        self
    ):

        if not self._trained:

            raise RuntimeError(
                "Model must be trained first."
            )