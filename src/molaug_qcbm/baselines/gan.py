import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)


class Generator(
    nn.Module
):
    """
    GAN Generator
    """

    def __init__(
        self,
        latent_dim,
        output_dim
    ):
        super().__init__()

        self.network = nn.Sequential(

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


class Discriminator(
    nn.Module
):
    """
    GAN Discriminator
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


class MolecularGAN:
    """
    Classical GAN baseline.

    Molecular Token Vectors
            ↓
        Generator
            ↓
        Discriminator
            ↓
       New Molecules
    """

    def __init__(
        self,
        latent_dim=32,
        batch_size=256,
        epochs=50,
        learning_rate=2e-4,
        device=None
    ):

        self.latent_dim = (
            latent_dim
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
        Train GAN.
        """

        X = np.asarray(
            X,
            dtype=np.float32
        )

        self.input_dim = (
            X.shape[1]
        )

        self.generator = (
            Generator(
                self.latent_dim,
                self.input_dim
            )
            .to(
                self.device
            )
        )

        self.discriminator = (
            Discriminator(
                self.input_dim
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

        criterion = (
            nn.BCELoss()
        )

        optimizer_g = (
            torch.optim.Adam(
                self.generator.parameters(),
                lr=self.learning_rate
            )
        )

        optimizer_d = (
            torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.learning_rate
            )
        )

        for epoch in range(
            self.epochs
        ):

            epoch_g_loss = 0.0

            epoch_d_loss = 0.0

            for batch in loader:

                real = (
                    batch[0]
                    .to(
                        self.device
                    )
                )

                batch_size = (
                    real.shape[0]
                )

                real_labels = (
                    torch.ones(
                        batch_size,
                        1
                    )
                    .to(
                        self.device
                    )
                )

                fake_labels = (
                    torch.zeros(
                        batch_size,
                        1
                    )
                    .to(
                        self.device
                    )
                )

                # -----------------
                # Train D
                # -----------------

                optimizer_d.zero_grad()

                real_output = (
                    self.discriminator(
                        real
                    )
                )

                real_loss = (
                    criterion(
                        real_output,
                        real_labels
                    )
                )

                z = torch.randn(
                    batch_size,
                    self.latent_dim
                ).to(
                    self.device
                )

                fake = (
                    self.generator(
                        z
                    )
                )

                fake_output = (
                    self.discriminator(
                        fake.detach()
                    )
                )

                fake_loss = (
                    criterion(
                        fake_output,
                        fake_labels
                    )
                )

                d_loss = (
                    real_loss
                    +
                    fake_loss
                )

                d_loss.backward()

                optimizer_d.step()

                # -----------------
                # Train G
                # -----------------

                optimizer_g.zero_grad()

                z = torch.randn(
                    batch_size,
                    self.latent_dim
                ).to(
                    self.device
                )

                fake = (
                    self.generator(
                        z
                    )
                )

                output = (
                    self.discriminator(
                        fake
                    )
                )

                g_loss = (
                    criterion(
                        output,
                        real_labels
                    )
                )

                g_loss.backward()

                optimizer_g.step()

                epoch_g_loss += (
                    g_loss.item()
                )

                epoch_d_loss += (
                    d_loss.item()
                )

            epoch_g_loss /= len(
                loader
            )

            epoch_d_loss /= len(
                loader
            )

            self.g_losses.append(
                epoch_g_loss
            )

            self.d_losses.append(
                epoch_d_loss
            )

        return self

    def generate(
        self,
        n_samples=1000
    ):
        """
        Generate molecular vectors.
        """

        self._check_trained()

        self.generator.eval()

        with torch.no_grad():

            z = torch.randn(
                n_samples,
                self.latent_dim
            ).to(
                self.device
            )

            generated = (
                self.generator(
                    z
                )
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
        Evaluate realism.
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

            scores = (
                self.discriminator(
                    X
                )
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
                "GAN",

            "latent_dim":
                self.latent_dim,

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