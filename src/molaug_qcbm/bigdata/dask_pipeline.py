from pathlib import Path

import pandas as pd

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

from molaug_qcbm.preprocessing.smiles import (
    valid_smiles
)

from molaug_qcbm.preprocessing.selfies import (
    safe_selfies
)


class DaskMoleculePipeline:
    """
    Dask-based molecular preprocessing pipeline.

    Designed for:
        1M+ molecules

    Pipeline:
        CSV
        ↓
        Dask DataFrame
        ↓
        SMILES validation
        ↓
        SELFIES generation
        ↓
        Parquet export
    """

    def __init__(
        self,
        npartitions=None
    ):
        if dd is None:
            raise ImportError(
                "dask is not installed. Install using: pip install dask[dataframe]"
            )

        self.npartitions = npartitions

    def load_csv(
        self,
        filepath,
        blocksize="64MB",
        assume_missing=True
    ):
        """
        Load large CSV using Dask.
        """

        return dd.read_csv(
            filepath,
            blocksize=blocksize,
            assume_missing=assume_missing
        )

    def select_smiles(
        self,
        ddf,
        smiles_column="SMILES"
    ):
        """
        Keep only SMILES column.
        """

        return ddf[
            [
                smiles_column
            ]
        ]

    def drop_null_smiles(
        self,
        ddf,
        smiles_column="SMILES"
    ):
        """
        Remove null SMILES.
        """

        return ddf.dropna(
            subset=[
                smiles_column
            ]
        )

    def validate_smiles(
        self,
        ddf,
        smiles_column="SMILES"
    ):
        """
        Add Valid column using RDKit validation.
        """

        ddf["Valid"] = ddf[
            smiles_column
        ].map(
            valid_smiles,
            meta=(
                "Valid",
                "bool"
            )
        )

        return ddf[
            ddf[
                "Valid"
            ]
        ]

    def generate_selfies(
        self,
        ddf,
        smiles_column="SMILES"
    ):
        """
        Add SELFIES column.
        """

        ddf["SELFIES"] = ddf[
            smiles_column
        ].map(
            safe_selfies,
            meta=(
                "SELFIES",
                "object"
            )
        )

        return ddf.dropna(
            subset=[
                "SELFIES"
            ]
        )

    def repartition(
        self,
        ddf,
        npartitions=None
    ):
        """
        Repartition Dask DataFrame.
        """

        partitions = (
            npartitions
            if npartitions is not None
            else self.npartitions
        )

        if partitions is None:
            return ddf

        return ddf.repartition(
            npartitions=partitions
        )

    def save_parquet(
        self,
        ddf,
        output_path,
        overwrite=True
    ):
        """
        Save processed data as parquet.
        """

        output_path = Path(
            output_path
        )

        ddf.to_parquet(
            output_path,
            engine="pyarrow",
            write_index=False,
            overwrite=overwrite
        )

        return output_path

    def save_csv(
        self,
        ddf,
        output_path,
        single_file=True
    ):
        """
        Save processed data as CSV.
        """

        output_path = Path(
            output_path
        )

        ddf.to_csv(
            str(output_path),
            single_file=single_file,
            index=False
        )

        return output_path

    def compute(
        self,
        ddf
    ):
        """
        Convert Dask DataFrame to pandas.
        Use only for smaller samples.
        """

        return ddf.compute()

    def sample(
        self,
        ddf,
        frac=0.01,
        random_state=42
    ):
        """
        Sample Dask DataFrame.
        """

        return ddf.sample(
            frac=frac,
            random_state=random_state
        )

    def head(
        self,
        ddf,
        n=5
    ):
        """
        Preview rows.
        """

        return ddf.head(
            n
        )

    def dataset_summary(
        self,
        ddf
    ):
        """
        Basic dataset statistics.
        """

        return {
            "rows":
                int(
                    ddf.shape[0].compute()
                ),

            "columns":
                len(
                    ddf.columns
                ),

            "column_names":
                list(
                    ddf.columns
                ),

            "npartitions":
                ddf.npartitions
        }

    def preprocess_pipeline(
        self,
        input_csv,
        output_parquet,
        smiles_column="SMILES",
        blocksize="64MB"
    ):
        """
        Complete Dask preprocessing pipeline.

        CSV
            ↓
        Select SMILES
            ↓
        Drop Nulls
            ↓
        Validate SMILES
            ↓
        Generate SELFIES
            ↓
        Save Parquet
        """

        ddf = self.load_csv(
            input_csv,
            blocksize=blocksize
        )

        ddf = self.select_smiles(
            ddf,
            smiles_column=smiles_column
        )

        ddf = self.drop_null_smiles(
            ddf,
            smiles_column=smiles_column
        )

        ddf = self.repartition(
            ddf
        )

        ddf = self.validate_smiles(
            ddf,
            smiles_column=smiles_column
        )

        ddf = self.generate_selfies(
            ddf,
            smiles_column=smiles_column
        )

        self.save_parquet(
            ddf,
            output_parquet
        )

        return ddf

    def parquet_to_dask(
        self,
        parquet_path
    ):
        """
        Load parquet as Dask DataFrame.
        """

        return dd.read_parquet(
            parquet_path,
            engine="pyarrow"
        )

    def parquet_to_pandas(
        self,
        parquet_path
    ):
        """
        Load parquet and compute to pandas.
        """

        return (
            self.parquet_to_dask(
                parquet_path
            )
            .compute()
        )