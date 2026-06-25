# autoencoder latent placeholder
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)


class MolecularAutoencoder(
    nn.Module
):
    """
    Feed-forward Autoencoder.

    Input:
        Token Count Vector

    Output:
        Reconstructed Vector
    """

    def __init__(
        self,
        input_dim,
        latent_dim=32
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

            nn.ReLU(),

            nn.Linear(
                128,
                latent_dim
            )
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
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
        return self.encoder(
            x
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
        x
    ):
        z = self.encode(
            x
        )

        x_hat = self.decode(
            z
        )

        return x_hat
        

class AutoencoderLatentEncoder:
    """
    Autoencoder-based latent encoder.

    Token Counts
        ↓
    Autoencoder
        ↓
    Latent Space
        ↓
    QCBM
    """

    def __init__(
        self,
        latent_dim=32,
        batch_size=256,
        epochs=25,
        learning_rate=1e-3,
        device=None
    ):

        self.latent_dim = latent_dim

        self.batch_size = batch_size

        self.epochs = epochs

        self.learning_rate = learning_rate

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = None

        self.input_dim = None

        self.training_loss = []

    def fit(
        self,
        X
    ):
        """
        Train autoencoder.
        """

        X = np.asarray(
            X,
            dtype=np.float32
        )

        self.input_dim = X.shape[1]

        self.model = (
            MolecularAutoencoder(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim
            )
            .to(self.device)
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

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        criterion = nn.MSELoss()

        self.model.train()

        for epoch in range(
            self.epochs
        ):

            epoch_loss = 0.0

            for batch in loader:

                x = (
                    batch[0]
                    .to(self.device)
                )

                optimizer.zero_grad()

                reconstruction = (
                    self.model(x)
                )

                loss = criterion(
                    reconstruction,
                    x
                )

                loss.backward()

                optimizer.step()

                epoch_loss += (
                    loss.item()
                )

            epoch_loss /= len(
                loader
            )

            self.training_loss.append(
                epoch_loss
            )

        return self

    def transform(
        self,
        X
    ):
        """
        Generate latent embeddings.
        """

        if self.model is None:

            raise RuntimeError(
                "Model not trained."
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        self.model.eval()

        with torch.no_grad():

            tensor = torch.tensor(
                X,
                dtype=torch.float32
            ).to(
                self.device
            )

            latent = (
                self.model
                .encode(
                    tensor
                )
            )

        return (
            latent
            .cpu()
            .numpy()
        )

    def fit_transform(
        self,
        X
    ):
        self.fit(X)

        return self.transform(
            X
        )

    def reconstruct(
        self,
        X
    ):
        """
        Reconstruct inputs.
        """

        if self.model is None:

            raise RuntimeError(
                "Model not trained."
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        self.model.eval()

        with torch.no_grad():

            tensor = torch.tensor(
                X,
                dtype=torch.float32
            ).to(
                self.device
            )

            reconstructed = (
                self.model(
                    tensor
                )
            )

        return (
            reconstructed
            .cpu()
            .numpy()
        )

    def reconstruction_error(
        self,
        X
    ):
        """
        Average reconstruction MSE.
        """

        reconstructed = (
            self.reconstruct(X)
        )

        mse = np.mean(
            (
                X -
                reconstructed
            ) ** 2
        )

        return float(
            mse
        )

    def save(
        self,
        filepath
    ):
        """
        Save model.
        """

        torch.save(
            {
                "state_dict":
                    self.model.state_dict(),

                "input_dim":
                    self.input_dim,

                "latent_dim":
                    self.latent_dim
            },
            filepath
        )

    def load(
        self,
        filepath
    ):
        """
        Load model.
        """

        checkpoint = torch.load(
            filepath,
            map_location=self.device
        )

        self.input_dim = checkpoint[
            "input_dim"
        ]

        self.latent_dim = checkpoint[
            "latent_dim"
        ]

        self.model = (
            MolecularAutoencoder(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim
            )
            .to(self.device)
        )

        self.model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

        self.model.eval()

        return self

    def summary(
        self
    ):
        return {

            "latent_dim":
                self.latent_dim,

            "epochs":
                self.epochs,

            "batch_size":
                self.batch_size,

            "device":
                self.device,

            "trained":
                self.model is not None
        }