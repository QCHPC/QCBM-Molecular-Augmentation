import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumNoiseGenerator:
    """
    Simple quantum generator.

    Random latent vector
        ↓
    Parameterized quantum circuit
        ↓
    State probabilities
        ↓
    Generated feature vector
    """

    def __init__(
        self,
        n_qubits=5,
        n_layers=2
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_states = 2 ** n_qubits
        self.n_parameters = (
            n_qubits *
            n_layers *
            2
        )

    def circuit(
        self,
        theta
    ):
        qc = QuantumCircuit(
            self.n_qubits
        )

        idx = 0

        for _ in range(
            self.n_layers
        ):
            for q in range(
                self.n_qubits
            ):
                qc.ry(
                    float(
                        theta[idx]
                    ),
                    q
                )
                idx += 1

                qc.rz(
                    float(
                        theta[idx]
                    ),
                    q
                )
                idx += 1

            for q in range(
                self.n_qubits - 1
            ):
                qc.cx(
                    q,
                    q + 1
                )

        return qc

    def generate_probabilities(
        self,
        theta
    ):
        qc = self.circuit(
            theta
        )

        state = Statevector.from_instruction(
            qc
        )

        probs = np.abs(
            state.data
        ) ** 2

        return probs

    def generate_batch(
        self,
        batch_size
    ):
        outputs = []

        for _ in range(
            batch_size
        ):
            theta = np.random.uniform(
                0,
                2 * np.pi,
                self.n_parameters
            )

            probs = (
                self.generate_probabilities(
                    theta
                )
            )

            outputs.append(
                probs
            )

        return np.array(
            outputs,
            dtype=np.float32
        )


class QGANGenerator(
    nn.Module
):
    """
    Maps quantum probability vectors
    into molecular feature vectors.
    """

    def __init__(
        self,
        quantum_dim,
        output_dim
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                quantum_dim,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                output_dim
            )
        )

    def forward(
        self,
        z
    ):
        return self.network(
            z
        )


class QGANDiscriminator(
    nn.Module
):
    """
    Classical discriminator.
    """

    def __init__(
        self,
        input_dim
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Linear(
                256,
                128
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Linear(
                128,
                1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        x
    ):
        return self.network(
            x
        )


class QuantumGAN:
    """
    Hybrid Quantum GAN baseline.

    Quantum circuit
        ↓
    Probability vector
        ↓
    Classical generator
        ↓
    Molecular feature vector
        ↓
    Classical discriminator
    """

    def __init__(
        self,
        n_qubits=5,
        n_layers=2,
        batch_size=128,
        epochs=30,
        learning_rate=2e-4,
        device=None
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.quantum_generator = (
            QuantumNoiseGenerator(
                n_qubits=n_qubits,
                n_layers=n_layers
            )
        )

        self.generator = None
        self.discriminator = None
        self.input_dim = None

        self.g_losses = []
        self.d_losses = []

    def fit(
        self,
        X
    ):
        """
        Train QGAN on molecular feature vectors.
        """

        X = np.asarray(
            X,
            dtype=np.float32
        )

        self.input_dim = X.shape[1]

        self.generator = (
            QGANGenerator(
                quantum_dim=2 ** self.n_qubits,
                output_dim=self.input_dim
            )
            .to(
                self.device
            )
        )

        self.discriminator = (
            QGANDiscriminator(
                input_dim=self.input_dim
            )
            .to(
                self.device
            )
        )

        dataset = TensorDataset(
            torch.tensor(
                X,
                dtype=torch.float32
            )
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.BCELoss()

        optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate
        )

        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate
        )

        for _ in range(
            self.epochs
        ):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            for batch in loader:
                real = batch[0].to(
                    self.device
                )

                batch_size = real.shape[0]

                real_labels = torch.ones(
                    batch_size,
                    1
                ).to(
                    self.device
                )

                fake_labels = torch.zeros(
                    batch_size,
                    1
                ).to(
                    self.device
                )

                # -------------------------
                # Train discriminator
                # -------------------------

                optimizer_d.zero_grad()

                real_output = (
                    self.discriminator(
                        real
                    )
                )

                real_loss = criterion(
                    real_output,
                    real_labels
                )

                q_noise = (
                    self.quantum_generator
                    .generate_batch(
                        batch_size
                    )
                )

                q_noise = torch.tensor(
                    q_noise,
                    dtype=torch.float32
                ).to(
                    self.device
                )

                fake = self.generator(
                    q_noise
                )

                fake_output = (
                    self.discriminator(
                        fake.detach()
                    )
                )

                fake_loss = criterion(
                    fake_output,
                    fake_labels
                )

                d_loss = (
                    real_loss
                    +
                    fake_loss
                )

                d_loss.backward()

                optimizer_d.step()

                # -------------------------
                # Train generator
                # -------------------------

                optimizer_g.zero_grad()

                q_noise = (
                    self.quantum_generator
                    .generate_batch(
                        batch_size
                    )
                )

                q_noise = torch.tensor(
                    q_noise,
                    dtype=torch.float32
                ).to(
                    self.device
                )

                fake = self.generator(
                    q_noise
                )

                output = self.discriminator(
                    fake
                )

                g_loss = criterion(
                    output,
                    real_labels
                )

                g_loss.backward()

                optimizer_g.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

            self.g_losses.append(
                epoch_g_loss / len(loader)
            )

            self.d_losses.append(
                epoch_d_loss / len(loader)
            )

        return self

    def generate(
        self,
        n_samples=1000
    ):
        """
        Generate molecular feature vectors.
        """

        self._check_trained()

        self.generator.eval()

        with torch.no_grad():

            q_noise = (
                self.quantum_generator
                .generate_batch(
                    n_samples
                )
            )

            q_noise = torch.tensor(
                q_noise,
                dtype=torch.float32
            ).to(
                self.device
            )

            generated = self.generator(
                q_noise
            )

        return (
            generated
            .cpu()
            .numpy()
        )

    def discriminator_score(
        self,
        X
    ):
        """
        Score generated or real vectors.
        """

        self._check_trained()

        X = torch.tensor(
            X,
            dtype=torch.float32
        ).to(
            self.device
        )

        self.discriminator.eval()

        with torch.no_grad():

            scores = self.discriminator(
                X
            )

        return (
            scores
            .cpu()
            .numpy()
            .flatten()
        )

    def training_history(
        self
    ):
        return {
            "generator_loss":
                self.g_losses,

            "discriminator_loss":
                self.d_losses
        }

    def summary(
        self
    ):
        return {
            "model":
                "QGAN",

            "n_qubits":
                self.n_qubits,

            "n_layers":
                self.n_layers,

            "epochs":
                self.epochs,

            "batch_size":
                self.batch_size,

            "device":
                self.device,

            "trained":
                self.generator
                is not None
        }

    def _check_trained(
        self
    ):
        if self.generator is None:
            raise RuntimeError(
                "Call fit() first."
            )