import torch
import math
import numpy as np
import warnings

from scipy.signal import butter
from torch.nn.functional import unfold, pad
from torch.nn import Parameter
from torchaudio.functional import lfilter
from torchaudio.transforms import Spectrogram, Resample

from typeguard import typechecked
from torchtyping import TensorType

from .bark import BarkScale
from .loudness import Loudness


class PesqLoss(torch.nn.Module):
    """Perceptual Evaluation of Speech Quality

    Implementation of the PESQ score in the PyTorch framework, closely following the ITU P.862
    reference. There are two mayor difference: (1) no time alignment (2) energy normalization
    uses an IIR filter.

    Attributes
    ----------
    to_spec : torch.nn.Module
        Perform a Short-Time Fourier Transformation on the time signal returning the power spectral
        density
    fbank : torch.nn.Module
        Apply a Bark scaling to the power distribution
    loudness : torch.nn.Module
        Estimate perceived loudness of the Bark scaled spectrogram
    power_filter : torch.tensor
        IIR filter coefficients to calculate power in 325Hz to 3.25kHz band
    pre_filter : torch.tensor
        Pre-empasize filter, applied to reference and degraded signal

    Methods
    -------
    align_level(self, tensor)
        Align level of signal to 10**7 in band 325Hz to 3.25kHz
    preemphasize(self, tensor)
        Pre-empasize a signal
    mos(self, tensor, tensor)
        Calculate the Mean Opinion Score between 1.08 and 4.999
    forward(self, tensor, tensor)
        Calculate the MOS score usable as loss; drops compression to valid range and flip sign
    """

    factor: float

    def __init__(
        self,
        factor: float,
        sample_rate: int = 48000,
        nbarks: int = 49,
        win_length: int = 512,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        """
        Parameters
        ----------
        factor : float
            Scaling of the loss function
        sample_rate : int
            Sampling rate of the time signal, re-samples if different from 16kHz
        nbarks : int
            Number of bark bands
        win_length : int
            Window size used in the STFT
        n_fft : int
            Number of frequency bins
        hop_length : int
            Distance between different frames
        """
        super(PesqLoss, self).__init__()

        self.factor = factor
        self.source_sample_rate = sample_rate

        # resample to 16kHz
        if sample_rate != 16000:
            self.resampler = Resample(sample_rate, 16000)

        # PESQ specifications state 32ms, 50% overlap, Hamming window
        self.to_spec = Spectrogram(
            win_length=win_length,
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            power=2,
            normalized=False,
            center=False,
        )

        # use a Bark filterbank to model perceived frequency resolution
        self.fbank = BarkScale(n_fft // 2, nbarks)

        # set up loudness degation and calibration
        self.loudness = Loudness(nbarks)

        # design IIR bandpass filter for power degation between 325Hz to 3.25kHz
        out = np.asarray(butter(5, [300, 3500], fs=16000, btype="band"))
        self.power_filter = Parameter(
            torch.as_tensor(out, dtype=torch.float32), requires_grad=False
        )

        # use IIR filter for pre-emphasize
        self.pre_filter = Parameter(
            torch.tensor(
                [[2.740826, -5.4816519, 2.740826], [1.0, -1.9444777, 0.94597794]],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @typechecked
    def align_level(
        self, signal: TensorType["batch", "sample"]
    ) -> TensorType["batch", "sample"]:
        """Align power to 10**7 for band 325 to 3.25kHz

        Parameters
        ----------
        signal : TensorType["batch", "sample"]
            Input time signal with size [batch, sample]

        Returns
        -------
        Tensor containing the scaled time signal
        """

        filtered_signal = lfilter(
            signal, self.power_filter[1], self.power_filter[0], clamp=False
        )

        # calculate power with weird bugs in reference implementation
        power = (
            (filtered_signal**2).sum(dim=1, keepdim=True)
            / (filtered_signal.shape[1] + 5120)
            / 1.04684
        )

        # align power
        signal = signal * (10**7 / power).sqrt()

        return signal

    @typechecked
    def preemphasize(
        self, signal: TensorType["batch", "sample"]
    ) -> TensorType["batch", "sample"]:
        """Pre-empasize a signal

        This pre-emphasize filter is also applied in the reference implementation. The filter
        coefficients are taken from the reference.

        Parameters
        ----------
        signal : TensorType["batch", "sample"]
            Input time signal with size [batch, sample]

        Returns
        -------
        Tensor containing the pre-emphasized signal
        """

        emp = torch.linspace(0, 15, 16, device=signal.device)[1:] / 16.0
        signal[:, :15] *= emp
        signal[:, -15:] *= torch.flip(emp, dims=(0,))

        signal = lfilter(signal, self.pre_filter[1], self.pre_filter[0], clamp=False)

        return signal

    @typechecked
    def raw(
        self, ref: TensorType["batch", "sample"], deg: TensorType["batch", "sample"]
    ) -> (torch.tensor, torch.tensor):
        """Calculate symmetric and asymmetric distances"""
        deg, ref = torch.atleast_2d(deg), torch.atleast_2d(ref)

        # equalize to [-1, 1] range
        max_val = torch.max(
            torch.amax(deg.abs(), dim=1, keepdim=True),
            torch.amax(ref.abs(), dim=1, keepdim=True),
        )
        deg, ref = deg / max_val, ref / max_val

        # resample to 16kHz if required
        if self.source_sample_rate != 16000:
            deg, ref = self.resampler(deg), self.resampler(ref)

        ref, deg = self.align_level(ref), self.align_level(deg)
        ref, deg = self.preemphasize(ref), self.preemphasize(deg)

        # do weird alignments with reference implementation
        deg = torch.nn.functional.pad(deg, (0, deg.shape[1] % 256))
        ref = torch.nn.functional.pad(ref, (0, ref.shape[1] % 256))
        # deg[:, 0:-1] = deg[:, 1:].clone()

        # calculate spectrogram for ref and degated speech
        deg, ref = self.to_spec(deg).swapaxes(1, 2), self.to_spec(ref).swapaxes(1, 2)

        # we won't use energy feature
        deg[:, :, 0] = 0.0
        ref[:, :, 0] = 0.0

        # calculate power spectrum in bark scale and hearing threshold
        deg, ref = self.fbank(deg), self.fbank(ref)

        # degate silent frames
        silent = self.loudness.total_audible(ref, 1e2) < 1e7

        # average power densities for frames
        mean_deg_pow = self.loudness.time_avg_audible(deg, silent)
        mean_ref_pow = self.loudness.time_avg_audible(ref, silent)

        band_pow_ratio = (
            ((mean_deg_pow + 1000) / (mean_ref_pow + 1000))
            .unsqueeze(1)
            .clamp(min=0.01, max=100.0)
        )
        equ_ref = band_pow_ratio * ref

        # normalize power of degated signal, averaged over bands
        frame_pow_ratio = (self.loudness.total_audible(equ_ref, 1) + 5e3) / (
            self.loudness.total_audible(deg, 1) + 5e3
        )

        frame_pow_ratio[:, 1:] = (
            frame_pow_ratio[:, 1:] * 0.8 + frame_pow_ratio[:, :-1] * 0.2
        )

        frame_pow_ratio = frame_pow_ratio.clamp(min=3e-4, max=5.0)

        equ_deg = frame_pow_ratio.unsqueeze(2) * deg

        deg_loud, ref_loud = self.loudness(equ_deg), self.loudness(equ_ref)

        # calculate disturbance
        deadzone = 0.25 * torch.min(deg_loud, ref_loud)
        disturbance = deg_loud - ref_loud
        disturbance = disturbance.sign() * (disturbance.abs() - deadzone).clamp(min=0)

        # symmetrical disturbance
        symm_distu = self.fbank.weighted_norm(disturbance, p=2)
        symm_distu = symm_distu.clamp(min=1e-20)

        # asymmetrical disturbance
        asymm_scaling = ((equ_deg + 50.0) / (equ_ref + 50.0)) ** 1.2
        asymm_scaling[asymm_scaling < 3.0] = 0.0
        asymm_scaling = asymm_scaling.clamp(max=12.0)

        asymm_distu = self.fbank.weighted_norm(disturbance * asymm_scaling, p=1)
        asymm_distu = asymm_distu.clamp(min=1e-20)

        # weighting
        h = ((self.loudness.total_audible(equ_ref, 1) + 1e5) / 1e7) ** 0.04
        symm_distu, asymm_distu = (symm_distu / h).clamp(max=45.0), (
            asymm_distu / h
        ).clamp(max=45.0)

        # calculate overlapping sums
        psqm = (
            unfold(symm_distu.unsqueeze(1).unsqueeze(1), (1, 20), stride=10) ** 6
        ).mean(dim=1) ** (1.0 / 6)
        d_symm = psqm.square().mean(dim=1).sqrt()

        psqm = (
            unfold(asymm_distu.unsqueeze(1).unsqueeze(1), (1, 20), stride=10) ** 6
        ).mean(dim=1) ** (1.0 / 6)
        d_asymm = psqm.square().mean(dim=1).sqrt()

        return d_symm, d_asymm

    @typechecked
    def mos(
        self, ref: TensorType["batch", "sample"], deg: TensorType["batch", "sample"]
    ) -> torch.tensor:
        """Calculate Mean Opinion Score

        Parameters
        ----------
        ref : TensorType["batch", "sample"]
            Reference signal
        deg : TensorType["batch", "sample"]
            Degraded signal

        Returns
        ----------
        Mean Opinion Score in range (1.08, 4.999)
        """

        d_symm, d_asymm = self.raw(ref, deg)

        # calculate MOS as combination of symmetric and asymmetric distance
        mos = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm

        # apply compression curve to have MOS in proper range
        mos = 0.999 + 4 / (1 + torch.exp(-1.3669 * mos + 3.8224))

        return mos

    @typechecked
    def forward(
        self, ref: TensorType["batch", "sample"], deg: TensorType["batch", "sample"]
    ) -> torch.tensor:
        """Calculate a loss variant of the MOS score

        This function combines symmetric and asymmetric distances but does not apply a range
        compression and flip the sign in order to maximize the MOS.

        Parameters
        ----------
        ref : TensorType["batch", "sample"]
            Reference signal
        deg : TensorType["batch", "sample"]
            Degraded signal

        Returns
        ----------
        Loss value in range [0, inf)
        """
        d_symm, d_asymm = self.raw(ref, deg)

        return self.factor * (0.1 * d_symm + 0.0309 * d_asymm)
