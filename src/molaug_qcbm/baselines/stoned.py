import random
import selfies as sf

from rdkit import Chem


class STONEDGenerator:
    """
    STONED molecular augmentation baseline.

    Uses SELFIES mutations to generate
    valid molecular structures.
    """

    def __init__(
        self,
        alphabet=None,
        random_state=42
    ):

        self.rng = random.Random(
            random_state
        )

        if alphabet is None:

            self.alphabet = list(
                sf.get_semantic_robust_alphabet()
            )

        else:

            self.alphabet = alphabet

    def mutate_selfie(
        self,
        selfie
    ):
        """
        Single SELFIES mutation.
        """

        tokens = list(
            sf.split_selfies(selfie)
        )

        if len(tokens) == 0:
            return selfie

        mutation_type = self.rng.choice(
            [
                "replace",
                "insert",
                "delete"
            ]
        )

        if mutation_type == "replace":

            idx = self.rng.randrange(
                len(tokens)
            )

            tokens[idx] = self.rng.choice(
                self.alphabet
            )

        elif mutation_type == "insert":

            idx = self.rng.randrange(
                len(tokens) + 1
            )

            tokens.insert(
                idx,
                self.rng.choice(
                    self.alphabet
                )
            )

        elif mutation_type == "delete":

            if len(tokens) > 1:

                idx = self.rng.randrange(
                    len(tokens)
                )

                del tokens[idx]

        return "".join(tokens)

    def depth_n_selfies_mutation(
        self,
        selfie,
        depth=3
    ):
        """
        Multiple mutations.
        """

        current = selfie

        for _ in range(depth):

            current = self.mutate_selfie(
                current
            )

        return current

    def generate_single(
        self,
        smiles,
        depth=3
    ):
        """
        Generate one molecule.
        """

        try:

            selfie = sf.encoder(
                smiles
            )

            mutated = (
                self.depth_n_selfies_mutation(
                    selfie,
                    depth=depth
                )
            )

            new_smiles = sf.decoder(
                mutated
            )

            mol = Chem.MolFromSmiles(
                new_smiles
            )

            if mol is None:
                return None

            Chem.SanitizeMol(mol)

            return Chem.MolToSmiles(
                mol,
                canonical=True
            )

        except Exception:

            return None

    def classical_stoned_generate(
        self,
        smiles_list,
        n_samples=1000,
        depth=3
    ):
        """
        Main generation function.
        """

        generated = []

        for _ in range(n_samples):

            seed = self.rng.choice(
                smiles_list
            )

            new_mol = self.generate_single(
                seed,
                depth=depth
            )

            if new_mol is not None:

                generated.append(
                    new_mol
                )

        return generated

    def uniqueness(
        self,
        smiles_list
    ):

        if len(smiles_list) == 0:
            return 0

        return (
            len(set(smiles_list))
            / len(smiles_list)
        )

    def summary(
        self,
        generated_smiles
    ):

        return {
            "generated":
                len(generated_smiles),

            "unique":
                len(set(generated_smiles)),

            "uniqueness":
                self.uniqueness(
                    generated_smiles
                )
        }