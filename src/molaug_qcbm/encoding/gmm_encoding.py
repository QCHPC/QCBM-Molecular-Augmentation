# gmm encoding placeholder
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture


class GMMEncoder:
    """
    Gaussian Mixture Model Encoder.

    Molecules
        ↓
    Latent Vectors
        ↓
    GMM
        ↓
    Cluster States
        ↓
    Distribution
        ↓
    QCBM

    Compared with KMeans:

    KMeans:
        Hard assignment

    GMM:
        Soft assignment
    """

    def __init__(
        self,
        n_components=32,
        covariance_type="full",
        random_state=42
    ):

        self.n_components = n_components

        self.covariance_type = covariance_type

        self.random_state = random_state

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )

        self.fitted = False

    def fit(
        self,
        X
    ):
        """
        Train GMM.
        """

        self.model.fit(X)

        self.fitted = True

        return self

    def predict(
        self,
        X
    ):
        """
        Hard cluster assignment.
        """

        self._check_fitted()

        return self.model.predict(X)

    def predict_proba(
        self,
        X
    ):
        """
        Soft assignment.

        Returns:

        shape:
            (n_samples,
             n_components)
        """

        self._check_fitted()

        return self.model.predict_proba(
            X
        )

    def fit_predict(
        self,
        X
    ):
        """
        Fit and return clusters.
        """

        self.fit(X)

        return self.predict(X)

    def fit_predict_proba(
        self,
        X
    ):
        """
        Fit and return soft assignments.
        """

        self.fit(X)

        return self.predict_proba(
            X
        )

    def cluster_distribution(
        self,
        cluster_ids
    ):
        """
        Hard-cluster distribution.
        """

        counts = np.bincount(
            cluster_ids,
            minlength=self.n_components
        )

        probs = (
            counts
            /
            counts.sum()
        )

        return probs

    def soft_distribution(
        self,
        probabilities
    ):
        """
        Average soft assignment.

        Recommended for QCBM.
        """

        probs = np.mean(
            probabilities,
            axis=0
        )

        probs = (
            probs
            /
            probs.sum()
        )

        return probs

    def target_distribution(
        self,
        X,
        use_soft=True
    ):
        """
        Create QCBM target distribution.
        """

        self._check_fitted()

        if use_soft:

            probs = self.predict_proba(
                X
            )

            return self.soft_distribution(
                probs
            )

        cluster_ids = self.predict(
            X
        )

        return self.cluster_distribution(
            cluster_ids
        )

    def cluster_centers(
        self
    ):
        """
        GMM means.
        """

        self._check_fitted()

        return self.model.means_

    def covariances(
        self
    ):
        """
        GMM covariance matrices.
        """

        self._check_fitted()

        return self.model.covariances_

    def bic(
        self,
        X
    ):
        """
        Bayesian Information Criterion.
        """

        self._check_fitted()

        return self.model.bic(
            X
        )

    def aic(
        self,
        X
    ):
        """
        Akaike Information Criterion.
        """

        self._check_fitted()

        return self.model.aic(
            X
        )

    def score(
        self,
        X
    ):
        """
        Mean log-likelihood.
        """

        self._check_fitted()

        return self.model.score(
            X
        )

    def summary(
        self
    ):
        """
        Model summary.
        """

        return {

            "fitted":
                self.fitted,

            "n_components":
                self.n_components,

            "covariance_type":
                self.covariance_type
        }

    def cluster_dataframe(
        self,
        X
    ):
        """
        Molecule-cluster dataframe.
        """

        self._check_fitted()

        clusters = self.predict(
            X
        )

        probs = self.predict_proba(
            X
        )

        rows = []

        for idx in range(
            len(clusters)
        ):

            rows.append(
                {
                    "Molecule":
                        idx,

                    "Cluster":
                        int(
                            clusters[idx]
                        ),

                    "Confidence":
                        float(
                            np.max(
                                probs[idx]
                            )
                        )
                }
            )

        return pd.DataFrame(
            rows
        )

    def save(
        self,
        filepath
    ):
        """
        Save trained model.
        """

        import joblib

        joblib.dump(
            self.model,
            filepath
        )

    def load(
        self,
        filepath
    ):
        """
        Load trained model.
        """

        import joblib

        self.model = joblib.load(
            filepath
        )

        self.fitted = True

        return self

    def _check_fitted(
        self
    ):
        if not self.fitted:

            raise RuntimeError(
                "GMMEncoder has not been fitted."
            )