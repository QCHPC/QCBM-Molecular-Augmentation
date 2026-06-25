# kmeans encoding placeholder
import numpy as np
import selfies as sf

from collections import Counter
from sklearn.cluster import KMeans


def selfies_to_latent(
    selfies_list,
    max_tokens=32
):

    vocab_counter = Counter()

    for s in selfies_list:
        vocab_counter.update(
            sf.split_selfies(s)
        )

    vocab = [
        tok
        for tok, _
        in vocab_counter.most_common(max_tokens)
    ]

    token_to_idx = {
        tok: idx
        for idx, tok
        in enumerate(vocab)
    }

    X = np.zeros(
        (len(selfies_list), len(vocab))
    )

    for i, selfie in enumerate(selfies_list):

        counts = Counter(
            sf.split_selfies(selfie)
        )

        for token, count in counts.items():

            if token in token_to_idx:
                X[i, token_to_idx[token]] = count

    X = X / (
        np.linalg.norm(
            X,
            axis=1,
            keepdims=True
        ) + 1e-12
    )

    return X, vocab


class KMeansEncoder:

    def __init__(
        self,
        n_clusters=32,
        random_state=42
    ):

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20
        )

    def fit(self, X):

        return self.kmeans.fit(X)

    def fit_predict(self, X):

        return self.kmeans.fit_predict(X)

    def predict(self, X):

        return self.kmeans.predict(X)

    def target_distribution(
        self,
        bins,
        n_bins
    ):

        counts = np.zeros(n_bins)

        for b in bins:
            counts[b] += 1

        return counts / counts.sum()