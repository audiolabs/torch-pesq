import math
import torch
from torch.nn.parameter import Parameter
from scipy import interpolate
import numpy as np

from typeguard import typechecked
from torchtyping import TensorType

# fmt: off
nr_of_hz_bands_per_bark_band_16k = [
    1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2,
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4,
    3, 4, 5, 4, 5, 6, 6, 7, 8, 9, 9, 12,12,15,16,
    18,21,25,20]

centre_of_band_bark_16k = [
    0.078672,     0.316341,     0.636559,     0.961246,     1.290450,
    1.624217,     1.962597,     2.305636,     2.653383,     3.005889,
    3.363201,     3.725371,     4.092449,     4.464486,     4.841533,
    5.223642,     5.610866,     6.003256,     6.400869,     6.803755,
    7.211971,     7.625571,     8.044611,     8.469146,     8.899232,
    9.334927,     9.776288,     10.223374,     10.676242,     11.134952,
    11.599563,     12.070135,     12.546731,     13.029408,     13.518232,
    14.013264,     14.514566,     15.022202,     15.536238,     16.056736,
    16.583761,     17.117382,     17.657663,     18.204674,     18.758478,
    19.319147,     19.886751,     20.461355,     21.043034]

centre_of_band_hz_16k = [
    7.867213,     31.634144,     63.655895,     96.124611,     129.044968,
    162.421738,     196.259659,     230.563568,     265.338348,     300.588867,
    336.320129,     372.537140,     409.244934,     446.448578,     484.568604,
    526.600586,     570.303833,     619.423340,     672.121643,     728.525696,
    785.675964,     846.835693,     909.691650,     977.063293,     1049.861694,
    1129.635986,     1217.257568,     1312.109497,     1412.501465,     1517.999390,
    1628.894165,     1746.194336,     1871.568848,     2008.776123,     2158.979248,
    2326.743164,     2513.787109,     2722.488770,     2952.586670,     3205.835449,
    3492.679932,     3820.219238,     4193.938477,     4619.846191,     5100.437012,
    5636.199219,     6234.313477,     6946.734863,     7796.473633]

width_of_band_bark_16k = [
    0.157344,     0.317994,     0.322441,     0.326934,     0.331474,
    0.336061,     0.340697,     0.345381,     0.350114,     0.354897,
    0.359729,     0.364611,     0.369544,     0.374529,     0.379565,
    0.384653,     0.389794,     0.394989,     0.400236,     0.405538,
    0.410894,     0.416306,     0.421773,     0.427297,     0.432877,
    0.438514,     0.444209,     0.449962,     0.455774,     0.461645,
    0.467577,     0.473569,     0.479621,     0.485736,     0.491912,
    0.498151,     0.504454,     0.510819,     0.517250,     0.523745,
    0.530308,     0.536934,     0.543629,     0.550390,     0.557220,
    0.564119,     0.571085,     0.578125,     0.585232]

width_of_band_hz_16k = [
    15.734426,     31.799433,     32.244064,     32.693359,     33.147385,
    33.606140,     34.069702,     34.538116,     35.011429,     35.489655,
    35.972870,     36.461121,     36.954407,     37.452911,     40.269653,
    42.311859,     45.992554,     51.348511,     55.040527,     56.775208,
    58.699402,     62.445862,     64.820923,     69.195374,     76.745667,
    84.016235,     90.825684,     97.931152,     103.348877,     107.801880,
    113.552246,     121.490601,     130.420410,     143.431763,     158.486816,
    176.872803,     198.314697,     219.549561,     240.600098,     268.702393,
    306.060059,     349.937012,     398.686279,     454.713867,     506.841797,
    564.863770,     637.261230,     794.717285,     931.068359]

pow_dens_correction_factor_16k = [
    100.000000,     99.999992,     100.000000,     100.000008,     100.000008,
    100.000015,     99.999992,     99.999969,     50.000027,     100.000000,
    99.999969,     100.000015,     99.999947,     100.000061,     53.047077,
    110.000046,     117.991989,     65.000000,     68.760147,     69.999931,
    71.428818,     75.000038,     76.843384,     80.968781,     88.646126,
    63.864388,     68.155350,     72.547775,     75.584831,     58.379192,
    80.950836,     64.135651,     54.384785,     73.821884,     64.437073,
    59.176456,     65.521278,     61.399822,     58.144047,     57.004543,
    64.126297,     54.311001,     61.114979,     55.077751,     56.849335,
    55.628868,     53.137054,     54.985844,     79.546974]
# fmt: on

Sp_16k = 6.910853e-006


def interp(values: list, nelms_new: int) -> TensorType:
    """Apply linear interpolation to the list of values

    Parameters
    ----------
    values : list
        The list of values to be interpolated
    nelms_new : int
        Number of values of returned list

    Returns
    -------
    TensorType
        a list of interpolated values
    """

    nelms = len(values)
    interp = interpolate.interp1d(np.arange(nelms), values)
    return torch.tensor(interp(np.linspace(0, 49.0, nelms_new, endpoint=False)))


class BarkScale(torch.nn.Module):
    """Bark filterbank according to P.862; can be extended with linear interpolation

    The ITU P.862 standard models perception with a Bark scaled filterbank. It uses
    rectangular filters and a constant width until 4kHz center frequency. This
    implementation uses interpolation to approximate the original parametrization when
    the number of band is different from the reference implementation.

    Parameters
    ----------
    nfreqs : int
        Number of frequency bins
    nbarks : int
        Number of Bark bands

    Attributes
    ----------
    pow_dens_correction : list
        Power density correction factors for each filter band
    width_hz : list
        Width of each filter in Hz
    width_bark : list
        Width of each filter in Bark
    centre : list
        Centre frequency of each band
    fbank : TensorType["band", "bark"]
        Filterbank matrix converting power spectrum to band powers
    """

    def __init__(self, nfreqs: int = 256, nbarks: int = 49):
        super(BarkScale, self).__init__()

        self.pow_dens_correction = Parameter(
            interp(pow_dens_correction_factor_16k, nbarks) * Sp_16k, requires_grad=False
        )
        self.width_hz = Parameter(
            interp(width_of_band_hz_16k, nbarks), requires_grad=False
        )
        self.width_bark = Parameter(
            interp(width_of_band_bark_16k, nbarks), requires_grad=False
        )
        self.centre = Parameter(
            interp(centre_of_band_hz_16k, nbarks), requires_grad=False
        )

        fbank = torch.zeros(nbarks, nfreqs)

        if nfreqs == 256 and nbarks == 49:
            # if default params are used, create filterbank matrix from given width

            current = 0
            # use filterbank width from reference for default band number
            for i in range(nbarks):
                end = current + nr_of_hz_bands_per_bark_band_16k[i]

                fbank[i, current:end] = 1.0
                current = end
        else:
            # otherwise generate one from number of barks and frequency bins

            prev, bin_width = 0, 8000.0 / nfreqs
            for i in range(nbarks):
                stride = self.width_hz[i] / bin_width
                centre = self.centre[i] / bin_width
                start, end = max(prev, int(math.floor(centre - stride / 2))), min(
                    nfreqs, int(math.ceil(centre + stride / 2))
                )
                fbank[i, start:end] = 1.0
                prev = end

        self.fbank = Parameter(fbank, requires_grad=False)
        self.total_width = self.width_bark[1:].sum()

    @typechecked
    def weighted_norm(
        self, tensor: TensorType["batch", "frame", "band"], p: float = 2
    ) -> TensorType["batch", "frame"]:
        """Calculates the p-norm taking band width into consideration

        Parameters
        ----------
        tensor : TensorType["batch", "frame", "band"]
            Power spectrogram with nfreqs frequency bins
        p : float
            Norm value

        Returns
        -------
        TensorType["batch", "frame"]
            scaled norm value
        """
        return self.total_width * (
            self.width_bark * tensor / self.total_width ** (1 / p)
        )[:, :, 1:].norm(p, dim=2)

    @typechecked
    def forward(
        self, tensor: TensorType["batch", "frame", "band"]
    ) -> TensorType["batch", "frame", "bark"]:
        """Converts a Hz-scaled spectrogram to a Bark-scaled spectrogram

        Parameters
        ----------
        tensor : TensorType["batch", "frame", "band"]
            A Hz-scaled power spectrogram

        Returns
        -------
        TensorType["batch", "frame", "bark"]
            A Bark-scaled power spectrogram
        """

        bark_powspec = torch.einsum("ij,klj->kli", self.fbank, tensor[:, :, :-1])
        return bark_powspec * self.pow_dens_correction
