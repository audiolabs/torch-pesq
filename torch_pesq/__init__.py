from .bark import *
from .loudness import *
from .loss import PesqLoss

# emit warning for torchaudio < 2.0.0
import torchaudio
from packaging import version

if version.parse(torchaudio.__version__) < version.parse("2.0.0"):
    print(
        "Warning: your torchaudio seems to be older than version 2.0.0, the energy alignment routine may run slowly. See https://github.com/audiolabs/torch-pesq/issues/3 for further details."
    )

__all__ = [
    "PesqLoss",
    "BarkScale",
    "Loudness",
]
