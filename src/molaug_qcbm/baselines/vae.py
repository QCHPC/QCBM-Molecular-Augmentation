import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)


class VariationalAutoencoder(
    nn.Module
):
    """
    Simple VAE for molecular
    token-count vectors.
    """

    def __init__(
        self,
        input_dim,
        latent_dim=32
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

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

        h = self.encoder(
            x
        )

        mu = self.mu_layer(
            h
        )

        logvar = (
            self.logvar_layer(
                h
            )
        )

        return mu, logvar

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
        x
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

        reconstruction = (
            self.decode(
                z
            )
        )

        return (
            reconstruction,
            mu,
            logvar
        )


class MolecularVAE:
    """
    Molecular VAE baseline.
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

        self.input_dim = None

        self.loss_history = []

    def vae_loss(
        self,
        reconstruction,
        x,
        mu,
        logvar
    ):

        recon_loss = (
            nn.functional.mse_loss(
                reconstruction,
                x,
                reduction="mean"
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

        self.input_dim = (
            X.shape[1]
        )

        self.model = (
            VariationalAutoencoder(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim
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

                reconstruction, mu, logvar = (
                    self.model(
                        x
                    )
                )

                loss = (
                    self.vae_loss(
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

            z = torch.randn(
                n_samples,
                self.latent_dim
            ).to(
                self.device
            )

            generated = (
                self.model.decode(
                    z
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
                    X_tensor
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