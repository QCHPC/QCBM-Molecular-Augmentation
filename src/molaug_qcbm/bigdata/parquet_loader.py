# parquet loader placeholder
import os
import pandas as pd


class MolecularDataLoader:
    """
    Large-scale molecular dataset loader.

    Supports:

    - CSV
    - Parquet
    - Chunked loading
    - Sampling
    - Column selection

    Designed for:
        1M+ molecules
    """

    def __init__(
        self,
        filepath
    ):

        self.filepath = filepath

        if not os.path.exists(
            filepath
        ):
            raise FileNotFoundError(
                filepath
            )

    def file_type(
        self
    ):

        ext = os.path.splitext(
            self.filepath
        )[1].lower()

        return ext

    def load(
        self,
        columns=None
    ):
        """
        Load full dataset.

        Use only when memory allows.
        """

        ext = self.file_type()

        if ext == ".csv":

            return pd.read_csv(
                self.filepath,
                usecols=columns
            )

        elif ext == ".parquet":

            return pd.read_parquet(
                self.filepath,
                columns=columns
            )

        else:

            raise ValueError(
                f"Unsupported file type: {ext}"
            )

    def sample(
        self,
        n=10000,
        random_state=42,
        columns=None
    ):
        """
        Sample rows.

        Useful for:
        - debugging
        - QCBM experiments
        """

        df = self.load(
            columns=columns
        )

        if n >= len(df):
            return df

        return df.sample(
            n=n,
            random_state=random_state
        )

    def csv_chunks(
        self,
        chunksize=100000,
        columns=None
    ):
        """
        Stream CSV in chunks.

        Example:

        for chunk in loader.csv_chunks():
            process(chunk)
        """

        if (
            self.file_type()
            != ".csv"
        ):
            raise ValueError(
                "Chunk loading only supported for CSV."
            )

        return pd.read_csv(
            self.filepath,
            chunksize=chunksize,
            usecols=columns
        )

    def parquet_batches(
        self,
        batch_size=100000
    ):
        """
        Stream parquet batches.

        Uses pyarrow.
        """

        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(
            self.filepath
        )

        for batch in parquet_file.iter_batches(
            batch_size=batch_size
        ):

            yield batch.to_pandas()

    def head(
        self,
        n=5
    ):

        return self.load().head(n)

    def shape(
        self
    ):

        df = self.load()

        return df.shape

    def memory_usage_mb(
        self
    ):

        df = self.load()

        return (
            df.memory_usage(
                deep=True
            ).sum()
            / (1024 ** 2)
        )

    def dataset_summary(
        self
    ):

        df = self.load()

        return {
            "rows":
                len(df),

            "columns":
                len(df.columns),

            "memory_mb":
                round(
                    self.memory_usage_mb(),
                    2
                ),

            "column_names":
                list(df.columns)
        }