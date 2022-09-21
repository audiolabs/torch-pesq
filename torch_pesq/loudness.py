import torch
from torch.nn import Parameter
import numpy as np

from torchtyping import TensorType
from typeguard import typechecked

from .bark import centre_of_band_bark_16k, interp

# fmt: off
abs_thresh_power_16k = [
    51286152.000000,     2454709.500000,     70794.593750,     4897.788574,     1174.897705,
    389.045166,     104.712860,     45.708820,     17.782795,     9.772372,
    4.897789,     3.090296,     1.905461,     1.258925,     0.977237,
    0.724436,     0.562341,     0.457088,     0.389045,     0.331131,
    0.295121,     0.269153,     0.257040,     0.251189,     0.251189,
    0.251189,     0.251189,     0.263027,     0.288403,     0.309030,
    0.338844,     0.371535,     0.398107,     0.436516,     0.467735,
    0.489779,     0.501187,     0.501187,     0.512861,     0.524807,
    0.524807,     0.524807,     0.512861,     0.478630,     0.426580,
    0.371535,     0.363078,     0.416869,     0.537032]
# fmt: on

zwicker_power = 0.23
Sl_16k = 1.866055e-001


class Loudness(torch.nn.Module):
    """Apply a loudness curve to the Bark spectrogram

    Parameters
    ----------
    nbark : int
        Number of bark bands

    Attributes
    ----------
    threshs : TensorType[1, 1, "band"]
        Hearing threshold per band; below a band is assumed to contain no significant energy
    exp : TensorType[1, 1, "band"]
        Exponent of each band
    """

    def __init__(self, nbark: int = 49):
        super(Loudness, self).__init__()

        self.threshs = Parameter(
            interp(abs_thresh_power_16k, nbark).unsqueeze(0).unsqueeze(0),
            requires_grad=False,
        )

        exp = 6 / (torch.tensor(centre_of_band_bark_16k) + 2.0)
        self.exp = Parameter(
            exp.clamp(min=1.0, max=2.0) ** 0.15 * zwicker_power, requires_grad=False
        )

    @typechecked
    def total_audible(
        self, tensor: TensorType["batch", "frame", "band"], factor: float = 1.0
    ) -> TensorType["batch", "frame"]:
        """Calculate total audible energy for each frame over all bands

        Parameters
        ----------
        tensor : TensorType["batch", "frame", "band"]
            A Bark scaled spectrogram with shape [batch_size, nframes, nbands]
        factor : float
            Scaling factor of the hearing threshold

        Returns
        -------
        TensorType["batch", "frame"]
            A tensor containing the hearable energy with shape [batch_size, nframes]
        """

        mask = tensor > self.threshs * factor

        tmp = (tensor * mask).sum(dim=2)
        return tmp

    @typechecked
    def time_avg_audible(
        self,
        tensor: TensorType["batch", "frame", "band"],
        silent: TensorType["batch", "frame"],
    ) -> TensorType["batch", "band"]:
        """Calculate arithmetic mean of audible energy for each band over all frames

        Parameters
        ----------
        tensor : TensorType["batch", "frame", "band"]
            A Bark scaled spectrogram with shape [batch_size, nframes, nbands]
        silent : TensorType["batch", "frame"]
            Indicates whether a frame is silent or not

        Returns
        -------
        TensorType["batch", "band"]
            A tensor containing the hearable energy with shape [batch_size, nbands]
        """

        mask = tensor > self.threshs * 100.0
        mask = mask * (~silent.unsqueeze(2))

        return (tensor * mask).mean(dim=1)

    @typechecked
    def forward(
        self, pow_dens: TensorType["batch", "frame", "band"]
    ) -> TensorType["batch", "frame", "band"]:
        """Transform Bark scaled power spectrogram to audible energy per band

        Parameters
        ----------
        pow_dens : TensorType["batch", "frame", "band"]
            A Bark scaled spectrogram with shape [batch_size, nframes, nbands]

        Returns
        -------
        TensorType["batch", "frame", "band"]
            A tensor containing the hearable energy with shape [batch_size, nframes, nbands]
        """

        loudness = (2.0 * self.threshs) ** self.exp * (
            (0.5 + 0.5 * pow_dens / self.threshs) ** self.exp - 1
        )
        loudness[pow_dens <= self.threshs] = 0.0

        return loudness * Sl_16k
