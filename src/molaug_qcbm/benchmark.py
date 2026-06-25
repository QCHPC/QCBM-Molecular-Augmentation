# Placeholder benchmark module
import pandas as pd

from molaug_qcbm.config import (
    PRIMARY_DATASET,
    N_CLUSTERS,
    N_GENERATED_MOLECULES
)

from molaug_qcbm.pipeline import (
    MolecularAugmentationPipeline
)

from molaug_qcbm.baselines.classical_born import (
    ClassicalBornMachine
)

from molaug_qcbm.baselines.markov_born import (
    MarkovBornMachine
)

from molaug_qcbm.baselines.stoned import (
    STONEDGenerator
)

from molaug_qcbm.evaluation.benchmark_metrics import (
    distribution_report
)


class BenchmarkRunner:
    """
    Benchmark currently implemented models.

    Supported:

    - Classical Born
    - Markov Born
    - QCBM
    - STONED
    """

    def __init__(
        self,
        dataset_path=PRIMARY_DATASET
    ):

        self.dataset_path = dataset_path

        self.pipeline = None

        self.target_distribution = None

        self.results = []

    def prepare_pipeline(
        self
    ):

        self.pipeline = (
            MolecularAugmentationPipeline(
                dataset_path=self.dataset_path
            )
        )

        (
            self.pipeline
            .load_dataset()
            .smiles_to_selfies()
            .build_latent_space()
            .cluster_molecules()
        )

        self.target_distribution = (
            self.pipeline.target_distribution
        )

        return self

    def run_qcbm(
        self
    ):

        self.pipeline.train_qcbm()

        probs = (
            self.pipeline.qcbm.probabilities()
        )

        result = distribution_report(
            "QCBM",
            probs,
            self.target_distribution
        )

        self.results.append(
            result
        )

        return result

    def run_classical_born(
        self
    ):

        model = ClassicalBornMachine(
            n_bits=5
        )

        model.fit(
            self.target_distribution
        )

        probs = (
            model.probabilities()
        )

        result = distribution_report(
            "Classical Born",
            probs,
            self.target_distribution
        )

        self.results.append(
            result
        )

        return result

    def run_markov_born(
        self
    ):

        model = MarkovBornMachine(
            n_bits=5
        )

        model.fit(
            self.target_distribution
        )

        probs = (
            model.probabilities()
        )

        result = distribution_report(
            "Markov Born",
            probs,
            self.target_distribution
        )

        self.results.append(
            result
        )

        return result

    def run_stoned(
        self
    ):

        stoned = STONEDGenerator()

        generated = (
            stoned.classical_stoned_generate(
                smiles_list=
                self.pipeline.smiles_list,
                n_samples=
                N_GENERATED_MOLECULES
            )
        )

        result = {
            "method":
                "STONED",
            "generated":
                len(generated),
            "unique":
                len(set(generated)),
            "uniqueness":
                (
                    len(set(generated))
                    /
                    len(generated)
                    if len(generated) > 0
                    else 0
                )
        }

        return result

    def run_all(
        self
    ):

        self.prepare_pipeline()

        print(
            "\nRunning Classical Born..."
        )

        self.run_classical_born()

        print(
            "Running Markov Born..."
        )

        self.run_markov_born()

        print(
            "Running QCBM..."
        )

        self.run_qcbm()

        benchmark_df = pd.DataFrame(
            self.results
        )

        return benchmark_df

    def save_results(
        self,
        filepath
    ):

        df = self.run_all()

        df.to_csv(
            filepath,
            index=False
        )

        return df