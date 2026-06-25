# fragment encoding placeholder
from collections import Counter

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import BRICS

from sklearn.preprocessing import LabelEncoder


class FragmentEncoder:
    """
    BRICS Fragment Encoder

    SMILES
        ↓
    BRICS Decomposition
        ↓
    Fragment Vocabulary
        ↓
    Integer Encoding

    Useful for:

    Fragment Diversity
    Fragment-QCBM
    Drug Discovery Analysis
    Medicinal Chemistry
    """

    def __init__(
        self
    ):
        self.label_encoder = LabelEncoder()

        self.fragment_mapping_ = {}

        self.fitted = False

    def smiles_to_fragments(
        self,
        smiles
    ):
        """
        Generate BRICS fragments.
        """

        try:

            mol = Chem.MolFromSmiles(
                smiles
            )

            if mol is None:
                return []

            fragments = BRICS.BRICSDecompose(
                mol
            )

            return list(
                fragments
            )

        except Exception:

            return []

    def extract_fragments(
        self,
        smiles_list
    ):
        """
        Extract all fragments from dataset.
        """

        all_fragments = []

        for smiles in smiles_list:

            fragments = (
                self.smiles_to_fragments(
                    smiles
                )
            )

            all_fragments.extend(
                fragments
            )

        return all_fragments

    def fit(
        self,
        smiles_list
    ):
        """
        Build fragment vocabulary.
        """

        fragments = (
            self.extract_fragments(
                smiles_list
            )
        )

        unique_fragments = sorted(
            list(
                set(
                    fragments
                )
            )
        )

        self.label_encoder.fit(
            unique_fragments
        )

        self.fragment_mapping_ = {

            frag: idx

            for idx, frag

            in enumerate(
                self.label_encoder.classes_
            )
        }

        self.fitted = True

        return self

    def transform(
        self,
        smiles_list
    ):
        """
        Convert molecule fragments
        into fragment ids.
        """

        if not self.fitted:

            raise RuntimeError(
                "Call fit() first."
            )

        encoded = []

        for smiles in smiles_list:

            fragments = (
                self.smiles_to_fragments(
                    smiles
                )
            )

            fragment_ids = []

            for frag in fragments:

                if (
                    frag
                    in
                    self.fragment_mapping_
                ):

                    fragment_ids.append(
                        self.fragment_mapping_[
                            frag
                        ]
                    )

            encoded.append(
                fragment_ids
            )

        return encoded

    def fit_transform(
        self,
        smiles_list
    ):
        self.fit(
            smiles_list
        )

        return self.transform(
            smiles_list
        )

    def fragment_counts(
        self,
        smiles_list
    ):
        """
        Fragment frequency table.
        """

        fragments = (
            self.extract_fragments(
                smiles_list
            )
        )

        return Counter(
            fragments
        )

    def top_fragments(
        self,
        smiles_list,
        top_k=20
    ):
        """
        Most common fragments.
        """

        counts = (
            self.fragment_counts(
                smiles_list
            )
        )

        rows = []

        for fragment, count in (
            counts.most_common(
                top_k
            )
        ):

            rows.append(
                {
                    "Fragment":
                        fragment,

                    "Count":
                        count
                }
            )

        return pd.DataFrame(
            rows
        )

    def fragment_distribution(
        self,
        smiles_list
    ):
        """
        Probability distribution
        over fragments.
        """

        counts = (
            self.fragment_counts(
                smiles_list
            )
        )

        total = sum(
            counts.values()
        )

        if total == 0:

            return {}

        return {

            fragment:
            count / total

            for fragment,
            count

            in counts.items()
        }

    def fragment_vector(
        self,
        smiles
    ):
        """
        Multi-hot fragment vector.
        """

        if not self.fitted:

            raise RuntimeError(
                "Call fit() first."
            )

        vector = np.zeros(
            len(
                self.fragment_mapping_
            ),
            dtype=int
        )

        fragments = (
            self.smiles_to_fragments(
                smiles
            )
        )

        for frag in fragments:

            if (
                frag
                in
                self.fragment_mapping_
            ):

                idx = (
                    self.fragment_mapping_[
                        frag
                    ]
                )

                vector[idx] = 1

        return vector

    def fragment_matrix(
        self,
        smiles_list
    ):
        """
        Multi-hot matrix.

        Shape:
        (n_molecules,
         n_fragments)
        """

        rows = []

        for smiles in smiles_list:

            rows.append(
                self.fragment_vector(
                    smiles
                )
            )

        return np.vstack(
            rows
        )

    def fragment_dataframe(
        self,
        smiles_list
    ):
        """
        Molecule-fragment mapping.
        """

        rows = []

        for smiles in smiles_list:

            rows.append(
                {
                    "SMILES":
                        smiles,

                    "Fragments":
                        self.smiles_to_fragments(
                            smiles
                        )
                }
            )

        return pd.DataFrame(
            rows
        )

    def summary(
        self
    ):
        """
        Encoder statistics.
        """

        if not self.fitted:

            return {
                "fitted": False
            }

        return {

            "fitted":
                True,

            "unique_fragments":
                len(
                    self.fragment_mapping_
                )
        }