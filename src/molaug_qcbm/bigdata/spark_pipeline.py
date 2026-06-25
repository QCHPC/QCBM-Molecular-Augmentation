# spark pipeline placeholder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import pandas as pd

from molaug_qcbm.preprocessing.smiles import (
    valid_smiles
)

from molaug_qcbm.preprocessing.selfies import (
    safe_selfies
)


class SparkMoleculePipeline:
    """
    Distributed molecular preprocessing.

    Designed for:

        1M+ molecules

    Tasks:

        Load
        Clean
        Validate
        SELFIES Conversion
        Export
    """

    def __init__(
        self,
        app_name="MolAug-QCBM"
    ):

        self.app_name = app_name

        self.spark = None

    def start(
        self
    ):
        """
        Create Spark session.
        """

        self.spark = (
            SparkSession
            .builder
            .appName(
                self.app_name
            )
            .getOrCreate()
        )

        return self.spark

    def stop(
        self
    ):
        """
        Stop Spark session.
        """

        if self.spark is not None:

            self.spark.stop()

    def load_csv(
        self,
        filepath,
        header=True,
        infer_schema=True
    ):
        """
        Load molecular dataset.
        """

        self._check_session()

        return (
            self.spark.read
            .csv(
                filepath,
                header=header,
                inferSchema=infer_schema
            )
        )

    def select_smiles(
        self,
        df,
        smiles_column="SMILES"
    ):
        """
        Keep SMILES column.
        """

        return df.select(
            smiles_column
        )

    def drop_null_smiles(
        self,
        df,
        smiles_column="SMILES"
    ):
        """
        Remove missing SMILES.
        """

        return df.filter(
            col(
                smiles_column
            ).isNotNull()
        )

    def validate_smiles(
        self,
        df,
        smiles_column="SMILES"
    ):
        """
        Validate SMILES.

        Uses pandas UDF style processing.
        """

        pdf = df.toPandas()

        pdf = pdf[
            pdf[
                smiles_column
            ].apply(
                valid_smiles
            )
        ]

        return self.spark.createDataFrame(
            pdf
        )

    def generate_selfies(
        self,
        df,
        smiles_column="SMILES"
    ):
        """
        Create SELFIES column.
        """

        pdf = df.toPandas()

        pdf["SELFIES"] = (
            pdf[
                smiles_column
            ]
            .apply(
                safe_selfies
            )
        )

        return self.spark.createDataFrame(
            pdf
        )

    def save_parquet(
        self,
        df,
        output_path
    ):
        """
        Save dataframe.
        """

        (
            df.write
            .mode("overwrite")
            .parquet(
                output_path
            )
        )

    def dataset_summary(
        self,
        df
    ):
        """
        Dataset statistics.
        """

        return {

            "rows":
                df.count(),

            "columns":
                len(
                    df.columns
                ),

            "column_names":
                df.columns
        }

    def preprocess_pipeline(
        self,
        input_csv,
        output_parquet,
        smiles_column="SMILES"
    ):
        """
        Complete preprocessing.

        CSV
            ↓
        Remove Nulls
            ↓
        Validate SMILES
            ↓
        Generate SELFIES
            ↓
        Save Parquet
        """

        df = self.load_csv(
            input_csv
        )

        df = self.select_smiles(
            df,
            smiles_column
        )

        df = self.drop_null_smiles(
            df,
            smiles_column
        )

        df = self.validate_smiles(
            df,
            smiles_column
        )

        df = self.generate_selfies(
            df,
            smiles_column
        )

        self.save_parquet(
            df,
            output_parquet
        )

        return df

    def parquet_to_pandas(
        self,
        parquet_path
    ):
        """
        Convert parquet back
        to pandas.
        """

        self._check_session()

        return (

            self.spark.read
            .parquet(
                parquet_path
            )
            .toPandas()
        )

    def _check_session(
        self
    ):
        if self.spark is None:

            raise RuntimeError(
                "Call start() first."
            )