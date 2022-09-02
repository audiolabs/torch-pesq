import torch
import math
import numpy as np
import warnings
from torchaudio.functional.functional import _create_triangular_filterbank
from torchaudio.functional import lfilter, DB_to_amplitude, filtfilt, biquad
from torchaudio.transforms import Spectrogram, Resample, InverseSpectrogram
import scipy
from scipy.signal import butter
from torch.nn.functional import unfold, pad
from torch.nn import Parameter

from .bark import BarkScale
from .loudness import Loudness
from .iir import prefilter_apply


class PesqLoss(torch.nn.Module):
    factor: float

    def __init__(
        self,
        factor,
        sample_rate=48000,
        nbarks=49,
        win_length=512,
        n_fft=512,
        hop_length=256,
    ):
        super(PesqLoss, self).__init__()

        self.factor = factor
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
        b, a = butter(4, [325, 3250], fs=16000, btype='band')
        self.power_filter = Parameter(torch.tensor([a, b], dtype=torch.float32), requires_grad=False)

        # use IIR filter for pre-emphasize
        self.pre_filter = Parameter(torch.tensor([[2.740826, -5.4816519, 2.740826], [1.0, -1.9444777, 0.94597794]], dtype=torch.float64), requires_grad=False)

    def align_level(self, signal):
        """Align level to 10**7 and apply IIR gain + correction factor of STFT"""

        filtered_signal = lfilter(signal, self.power_filter[0], self.power_filter[1], clamp=False)
        power = (filtered_signal ** 2).sum() / (filtered_signal.shape[1] + 5120) / 1.04684
        #print("{:.10}".format(power))
        signal = signal * (10**7 / power).sqrt()
        return signal

    def preemphasize(self, signal, filter=True):
        emp = np.linspace(0,15,16)[1:] / 16.
        signal[:, :15] *= emp
        signal[:, -15:] *= emp[::-1]

        if filter:
            signal = lfilter(signal.double(), self.pre_filter[1], self.pre_filter[0], clamp=False).float()

        return signal 

    def raw(self, deg, ref):
        # both signals should have same length, we don't support alignment
        assert deg.shape == ref.shape

        deg, ref = torch.atleast_2d(deg), torch.atleast_2d(ref)

        if hasattr(self, "resampler"):
            deg, ref = self.resampler(deg), self.resampler(ref)

        #ref = self.align_level(ref)
        #deg = self.align_level(deg)
        ref = ref * 32320.01953125
        deg = deg * 23455.2265625
        #deg = deg * math.sqrt(10.0 ** 7 / 0.018176937475800514221)

        #deg, ref = pad(deg, (4800, 4800)), pad(ref, (4800, 4800))
        deg, ref = self.preemphasize(deg, filter=True), self.preemphasize(ref)

        ## pad input
        #nsamples = deg.shape[1]

        #print(ref[0, 5000:5050])
        ## degate power

        #breakpoint()
        # calculate spectrogram for ref and degated speech
        deg = torch.nn.functional.pad(deg, (0, deg.shape[1] % 256))
        ref = torch.nn.functional.pad(ref, (0, ref.shape[1] % 256))
        deg[:, 0:-1] = deg[:, 1:].clone()
        deg, ref = self.to_spec(deg).swapaxes(1, 2), self.to_spec(ref).swapaxes(
            1, 2
        )
        deg[:,:,0] = 0.0
        ref[:,:,0] = 0.0


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
        #frame_pow_ratio[:, 0] += 0.2


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
        symm_distu, asymm_distu = (symm_distu / h).clamp(max=45.), (asymm_distu / h).clamp(max=45.)

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

    def mos(self, ref, deg):
        d_symm, d_asymm = self.raw(deg, ref)

        # calculate MOS as combination of symmetric and asymmetric distance
        mos = 4.5 - 0.1 * d_symm - 0.0309 * d_asymm

        # apply compression curve to have MOS in (1, 5)
        mos = 0.999 + 4 / (1 + torch.exp(-1.3669 * mos + 3.8224))

        return mos

    def forward(self, deg, ref):
        return self.factor * (-self.mos(deg, ref) + 4.5)
