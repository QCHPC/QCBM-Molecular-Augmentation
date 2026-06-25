# scaffold encoding placeholder
from collections import Counter

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.preprocessing import LabelEncoder


class ScaffoldEncoder:
    """
    Murcko Scaffold Encoder

    SMILES
        ↓
    Murcko Scaffold
        ↓
    Integer State

    Useful for:

    Scaffold Diversity
    Scaffold-Based QCBM
    Medicinal Chemistry Analysis
    """

    def __init__(
        self
    ):
        self.label_encoder = LabelEncoder()

        self.scaffolds_ = None

        self.scaffold_mapping_ = None

        self.fitted = False

    def smiles_to_scaffold(
        self,
        smiles
    ):
        """
        Convert molecule to Murcko scaffold.
        """

        try:

            mol = Chem.MolFromSmiles(
                smiles
            )

            if mol is None:
                return None

            scaffold = (
                MurckoScaffold
                .MurckoScaffoldSmiles(
                    mol=mol
                )
            )

            if scaffold == "":
                return None

            return scaffold

        except Exception:

            return None

    def extract_scaffolds(
        self,
        smiles_list
    ):
        """
        Generate scaffold list.
        """

        scaffolds = []

        for smiles in smiles_list:

            scaffolds.append(
                self.smiles_to_scaffold(
                    smiles
                )
            )

        return scaffolds

    def fit(
        self,
        smiles_list
    ):
        """
        Learn scaffold vocabulary.
        """

        scaffolds = (
            self.extract_scaffolds(
                smiles_list
            )
        )

        valid_scaffolds = [

            scaffold

            for scaffold

            in scaffolds

            if scaffold is not None
        ]

        self.label_encoder.fit(
            valid_scaffolds
        )

        self.scaffolds_ = (
            valid_scaffolds
        )

        self.scaffold_mapping_ = {

            scaffold: idx

            for idx, scaffold

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
        Convert scaffolds to integer states.
        """

        if not self.fitted:

            raise RuntimeError(
                "Call fit() first."
            )

        encoded = []

        for smiles in smiles_list:

            scaffold = (
                self.smiles_to_scaffold(
                    smiles
                )
            )

            if scaffold is None:

                encoded.append(
                    -1
                )

                continue

            if (
                scaffold
                not in
                self.scaffold_mapping_
            ):

                encoded.append(
                    -1
                )

                continue

            encoded.append(
                self.scaffold_mapping_[
                    scaffold
                ]
            )

        return np.array(
            encoded
        )

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

    def scaffold_counts(
        self,
        smiles_list
    ):
        """
        Scaffold frequency table.
        """

        scaffolds = (
            self.extract_scaffolds(
                smiles_list
            )
        )

        return Counter(
            [
                s
                for s in scaffolds
                if s is not None
            ]
        )

    def top_scaffolds(
        self,
        smiles_list,
        top_k=20
    ):
        """
        Most common scaffolds.
        """

        counts = (
            self.scaffold_counts(
                smiles_list
            )
        )

        rows = []

        for scaffold, count in (
            counts.most_common(
                top_k
            )
        ):

            rows.append(
                {
                    "Scaffold":
                        scaffold,

                    "Count":
                        count
                }
            )

        return pd.DataFrame(
            rows
        )

    def scaffold_distribution(
        self,
        smiles_list
    ):
        """
        Probability distribution
        over scaffolds.
        """

        counts = (
            self.scaffold_counts(
                smiles_list
            )
        )

        total = sum(
            counts.values()
        )

        distribution = {

            scaffold:
            count / total

            for scaffold,
            count

            in counts.items()
        }

        return distribution

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

            "fitted": True,

            "unique_scaffolds":
                len(
                    self.scaffold_mapping_
                )
        }

    def scaffold_dataframe(
        self,
        smiles_list
    ):
        """
        Molecule-scaffold mapping.
        """

        rows = []

        for smiles in smiles_list:

            scaffold = (
                self.smiles_to_scaffold(
                    smiles
                )
            )

            rows.append(
                {
                    "SMILES":
                        smiles,

                    "Scaffold":
                        scaffold
                }
            )

        return pd.DataFrame(
            rows
        )