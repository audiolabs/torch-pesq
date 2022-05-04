import torch
import warnings
from torchaudio.functional.functional import _create_triangular_filterbank
from torchaudio.functional import lfilter, DB_to_amplitude
from torchaudio.transforms import Spectrogram, Resample, InverseSpectrogram
from scipy.signal import butter
from torch.nn.functional import unfold
from torch.nn import Parameter

from .bark import BarkScale
from .loudness import Loudness

class Pesq(torch.nn.Module):
    factor: float

    def __init__(self, factor, sample_rate=48000, nbarks=49, win_length=512, n_fft=512, hop_length=256):
        super(Pesq, self).__init__()

        self.factor = factor
        # resample to 16kHz
        if sample_rate != 16000:
            self.resampler = Resample(sample_rate, 16000).cuda()
        # PESQ specifications state 32ms, 50% overlap, Hamming windowh
        self.to_spec = Spectrogram(win_length=win_length, n_fft=n_fft, hop_length=hop_length, window_fn=torch.hamming_window,power=None, normalized=False).cuda()
        # use a Bark filterbank to model perceived frequency resolution
        self.fbank = BarkScale(n_fft//2, nbarks).cuda()
        # set up loudness estimation and calibration
        self.loudness = Loudness(nbarks).cuda()

    def align_level(self, signal):
        """ Align level to 10**7 and apply IIR gain + correction factor of STFT"""
        power = (signal.abs() ** 2 / 512.)[:, :, 10:96].mean(dim=(1,2), keepdim=True)
        signal = signal * (10**7 / power).sqrt() * 2.47 * 1.074
        return signal

    def mos(self, estim, clean):
        if hasattr(self, 'resampler'):
            estim, clean = self.resampler(estim), self.resampler(clean)

        # calculate spectrogram for clean and estimated speech
        estim, clean = self.to_spec(estim).swapaxes(1, 2), self.to_spec(clean).swapaxes(1, 2)

        estim, clean = self.align_level(estim), self.align_level(clean)

        # calculate power spectrum in bark scale and hearing threshold
        estim, clean = self.fbank(estim), self.fbank(clean)

        # estimate silent frames
        silent = (self.loudness.total_audible(clean, 1e2) < 1e7)

        # average power densities for frames
        mean_estim_pow = self.loudness.time_avg_audible(estim, silent)
        mean_clean_pow = self.loudness.time_avg_audible(clean, silent)

        band_pow_ratio = ((mean_estim_pow + 1000) / (mean_clean_pow + 1000)).unsqueeze(1).clamp(min=0.01, max=100.0)
        equ_clean = band_pow_ratio * clean

        # normalize power of estimated signal, averaged over bands
        frame_pow_ratio = ((self.loudness.total_audible(clean, 1) + 5e3) / (self.loudness.total_audible(estim, 1) + 5e3))

        frame_pow_ratio[:, 1:] = frame_pow_ratio[:, 1:] * 0.8 + frame_pow_ratio[:, :-1] * 0.2
        frame_pow_ratio = frame_pow_ratio.clamp(min=3e-4,max=5.0)
        equ_estim = frame_pow_ratio.unsqueeze(2) * estim

        estim_loud, clean_loud = self.loudness(equ_estim), self.loudness(equ_clean)

        # calculate disturbance
        deadzone = 0.25 * torch.min(estim_loud, clean_loud)
        disturbance = ((estim_loud - clean_loud).abs() - deadzone).clamp(min=0)

        # symmetrical disturbance
        symm_distu = self.fbank.weighted_norm(disturbance, p=2)
        symm_distu = symm_distu.clamp(min=1e-20)

        # asymmetrical disturbance
        asymm_scaling = ((estim + 50.) / (clean + 50.)) ** 1.2 
        asymm_scaling[asymm_scaling < 3.] = 0.
        asymm_scaling = asymm_scaling.clamp(max=12.)

        asymm_distu = self.fbank.weighted_norm(disturbance * asymm_scaling, p=1)
        asymm_distu = asymm_distu.clamp(min=1e-20)

        # calculate overlapping sums
        psqm = (unfold(symm_distu.unsqueeze(1).unsqueeze(1), (1,20), stride=10) ** 6).mean(dim=1) ** (1./6)
        d_symm = psqm.square().mean(dim=1).sqrt().mean()

        psqm = (unfold(asymm_distu.unsqueeze(1).unsqueeze(1), (1,20), stride=10) ** 6).mean(dim=1) ** (1./6)
        d_asymm = psqm.square().mean(dim=1).sqrt().mean()

        return 4.5 - 0.1 * d_symm - 0.0309 * d_asymm 
    
    def forward(self, estim, clean):
        return self.factor * (-self.mos(estim, clean) + 4.5)
