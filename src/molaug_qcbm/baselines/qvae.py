import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumLatentLayer:
    """
    Simple quantum latent layer.

    Maps latent vectors into
    quantum state probabilities.
    """

    def __init__(
        self,
        n_qubits=5
    ):

        self.n_qubits = n_qubits

        self.n_states = (
            2 ** n_qubits
        )

    def encode(
        self,
        latent_vector
    ):

        qc = QuantumCircuit(
            self.n_qubits
        )

        for i in range(
            min(
                self.n_qubits,
                len(latent_vector)
            )
        ):

            qc.ry(
                float(
                    latent_vector[i]
                ),
                i
            )

        for i in range(
            self.n_qubits - 1
        ):

            qc.cx(
                i,
                i + 1
            )

        statevector = (
            Statevector
            .from_instruction(
                qc
            )
        )

        probs = (
            np.abs(
                statevector.data
            ) ** 2
        )

        return probs


class QVAEModel(
    nn.Module
):
    """
    Hybrid Quantum VAE.
    """

    def __init__(
        self,
        input_dim,
        latent_dim=8
    ):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU()
        )

        self.mu_layer = nn.Linear(
            128,
            latent_dim
        )

        self.logvar_layer = nn.Linear(
            128,
            latent_dim
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                32,
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
                input_dim
            )
        )

    def encode(
        self,
        x
    ):

        h = self.encoder(
            x
        )

        mu = (
            self.mu_layer(
                h
            )
        )

        logvar = (
            self.logvar_layer(
                h
            )
        )

        return (
            mu,
            logvar
        )

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(
            std
        )

        return (
            mu +
            eps * std
        )

    def decode(
        self,
        z
    ):

        return self.decoder(
            z
        )

    def forward(
        self,
        x,
        quantum_layer
    ):

        mu, logvar = (
            self.encode(
                x
            )
        )

        z = (
            self.reparameterize(
                mu,
                logvar
            )
        )

        quantum_features = []

        z_np = (
            z.detach()
            .cpu()
            .numpy()
        )

        for sample in z_np:

            probs = (
                quantum_layer.encode(
                    sample
                )
            )

            quantum_features.append(
                probs
            )

        quantum_features = (
            torch.tensor(
                np.array(
                    quantum_features
                ),
                dtype=torch.float32
            )
            .to(
                x.device
            )
        )

        reconstruction = (
            self.decode(
                quantum_features
            )
        )

        return (
            reconstruction,
            mu,
            logvar
        )


class QuantumVAE:
    """
    Quantum Variational
    Autoencoder baseline.

    Token Counts
         ↓
      Encoder
         ↓
    Quantum Latent
         ↓
      Decoder
         ↓
    Generated Molecules
    """

    def __init__(
        self,
        latent_dim=8,
        n_qubits=5,
        batch_size=128,
        epochs=20,
        learning_rate=1e-3,
        device=None
    ):

        self.latent_dim = (
            latent_dim
        )

        self.n_qubits = (
            n_qubits
        )

        self.batch_size = (
            batch_size
        )

        self.epochs = epochs

        self.learning_rate = (
            learning_rate
        )

        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = None

        self.quantum_layer = (
            QuantumLatentLayer(
                n_qubits
            )
        )

        self.loss_history = []

    def qvae_loss(
        self,
        reconstruction,
        x,
        mu,
        logvar
    ):

        recon_loss = (
            nn.functional.mse_loss(
                reconstruction,
                x
            )
        )

        kl_loss = -0.5 * torch.mean(

            1
            + logvar
            - mu.pow(2)
            - logvar.exp()
        )

        return (
            recon_loss
            + kl_loss
        )

    def fit(
        self,
        X
    ):

        X = np.asarray(
            X,
            dtype=np.float32
        )

        input_dim = (
            X.shape[1]
        )

        self.model = (
            QVAEModel(
                input_dim,
                self.latent_dim
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

        optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
        )

        for epoch in range(
            self.epochs
        ):

            epoch_loss = 0.0

            for batch in loader:

                x = (
                    batch[0]
                    .to(
                        self.device
                    )
                )

                optimizer.zero_grad()

                reconstruction, mu, logvar = (
                    self.model(
                        x,
                        self.quantum_layer
                    )
                )

                loss = (
                    self.qvae_loss(
                        reconstruction,
                        x,
                        mu,
                        logvar
                    )
                )

                loss.backward()

                optimizer.step()

                epoch_loss += (
                    loss.item()
                )

            epoch_loss /= len(
                loader
            )

            self.loss_history.append(
                epoch_loss
            )

        return self

    def encode(
        self,
        X
    ):

        X = torch.tensor(
            X,
            dtype=torch.float32
        ).to(
            self.device
        )

        self.model.eval()

        with torch.no_grad():

            mu, _ = (
                self.model.encode(
                    X
                )
            )

        return (
            mu.cpu()
            .numpy()
        )

    def generate(
        self,
        n_samples=1000
    ):

        self.model.eval()

        with torch.no_grad():

            latent = np.random.normal(
                0,
                1,
                (
                    n_samples,
                    self.latent_dim
                )
            )

            quantum_vectors = []

            for vec in latent:

                quantum_vectors.append(

                    self.quantum_layer.encode(
                        vec
                    )
                )

            quantum_vectors = (
                torch.tensor(
                    np.array(
                        quantum_vectors
                    ),
                    dtype=torch.float32
                )
                .to(
                    self.device
                )
            )

            generated = (
                self.model.decode(
                    quantum_vectors
                )
            )

        return (
            generated
            .cpu()
            .numpy()
        )

    def reconstruction_error(
        self,
        X
    ):

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32
        ).to(
            self.device
        )

        self.model.eval()

        with torch.no_grad():

            reconstruction, _, _ = (
                self.model(
                    X_tensor,
                    self.quantum_layer
                )
            )

        mse = np.mean(

            (
                X -
                reconstruction
                .cpu()
                .numpy()
            ) ** 2
        )

        return float(
            mse
        )

    def summary(
        self
    ):

        return {

            "model":
                "QVAE",

            "latent_dim":
                self.latent_dim,

            "n_qubits":
                self.n_qubits,

            "epochs":
                self.epochs,

            "batch_size":
                self.batch_size,

            "device":
                self.device,

            "trained":
                self.model is not None
        }