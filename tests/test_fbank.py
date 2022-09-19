from torch_pesq import BarkScale, nr_of_hz_bands_per_bark_band_16k
import torch


def test_sums():
    tmp = BarkScale()

    assert torch.all(
        tmp.fbank.sum(dim=1) == torch.tensor(nr_of_hz_bands_per_bark_band_16k)
    )
    assert torch.all(tmp.fbank.sum(dim=0) == 1.0)
