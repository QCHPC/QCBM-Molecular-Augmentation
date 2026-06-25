# augmentation filtering placeholder
import pandas as pd

from molaug_qcbm.evaluation.validity import (
    is_valid_smiles
)

from molaug_qcbm.evaluation.syba import (
    syba_score_smiles
)

from molaug_qcbm.evaluation.qed import (
    qed_score
)

from molaug_qcbm.evaluation.sa_score import (
    sa_score
)


class MoleculeFilter:
    """
    Molecular quality filtering.

    Pipeline:

    Validity
        ↓
    SYBA
        ↓
    QED
        ↓
    SA Score
    """

    def __init__(
        self,
        min_syba=0.0,
        min_qed=0.4,
        max_sa=6.0
    ):

        self.min_syba = min_syba

        self.min_qed = min_qed

        self.max_sa = max_sa

    def validity_filter(
        self,
        smiles_list
    ):
        """
        Keep valid molecules only.
        """

        return [
            smi
            for smi in smiles_list
            if is_valid_smiles(
                smi
            )
        ]

    def syba_filter(
        self,
        smiles_list
    ):
        """
        Keep molecules passing
        synthesizability threshold.
        """

        filtered = []

        for smi in smiles_list:

            score = (
                syba_score_smiles(
                    smi
                )
            )

            if score is None:
                continue

            if score >= self.min_syba:

                filtered.append(
                    smi
                )

        return filtered

    def qed_filter(
        self,
        smiles_list
    ):
        """
        Keep drug-like molecules.
        """

        filtered = []

        for smi in smiles_list:

            score = qed_score(
                smi
            )

            if score is None:
                continue

            if score >= self.min_qed:

                filtered.append(
                    smi
                )

        return filtered

    def sa_filter(
        self,
        smiles_list
    ):
        """
        Keep synthesizable molecules.

        Lower SA is better.
        """

        filtered = []

        for smi in smiles_list:

            score = sa_score(
                smi
            )

            if score is None:
                continue

            if score <= self.max_sa:

                filtered.append(
                    smi
                )

        return filtered

    def full_filter(
        self,
        smiles_list
    ):
        """
        Complete molecular pipeline.
        """

        molecules = (
            self.validity_filter(
                smiles_list
            )
        )

        molecules = (
            self.syba_filter(
                molecules
            )
        )

        molecules = (
            self.qed_filter(
                molecules
            )
        )

        molecules = (
            self.sa_filter(
                molecules
            )
        )

        return molecules

    def filter_report(
        self,
        smiles_list
    ):
        """
        Track molecules surviving
        each filtering stage.
        """

        initial = len(
            smiles_list
        )

        valid = self.validity_filter(
            smiles_list
        )

        syba = self.syba_filter(
            valid
        )

        qed = self.qed_filter(
            syba
        )

        sa = self.sa_filter(
            qed
        )

        return {

            "initial":
                initial,

            "valid":
                len(valid),

            "syba":
                len(syba),

            "qed":
                len(qed),

            "sa":
                len(sa),

            "final":
                len(sa)
        }

    def scored_dataframe(
        self,
        smiles_list
    ):
        """
        Create benchmark table
        with all molecular metrics.
        """

        rows = []

        for smi in smiles_list:

            rows.append({

                "SMILES":
                    smi,

                "Valid":
                    is_valid_smiles(
                        smi
                    ),

                "SYBA":
                    syba_score_smiles(
                        smi
                    ),

                "QED":
                    qed_score(
                        smi
                    ),

                "SA":
                    sa_score(
                        smi
                    )
            })

        return pd.DataFrame(
            rows
        )

    def top_molecules(
        self,
        smiles_list,
        top_k=50
    ):
        """
        Best molecules ranked
        by QED then SYBA.
        """

        df = self.scored_dataframe(
            smiles_list
        )

        df = df.dropna()

        df = df.sort_values(
            by=[
                "QED",
                "SYBA"
            ],
            ascending=False
        )

        return (
            df
            .head(top_k)
            .reset_index(
                drop=True
            )
        )