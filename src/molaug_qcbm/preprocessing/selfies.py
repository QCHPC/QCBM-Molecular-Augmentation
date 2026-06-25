# selfies processing placeholder
import selfies as sf


def safe_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except:
        return None


def safe_decode_selfies(selfie):
    try:
        return sf.decoder(selfie)
    except:
        return None


def get_selfie_chars(selfie):
    return list(sf.split_selfies(selfie))