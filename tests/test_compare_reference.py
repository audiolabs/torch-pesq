import random
import torch
import numpy as np
import pytest
import pathlib
import torchaudio
from joblib import Parallel, delayed
import scipy

from torch_pesq import PesqLoss
from pesq import pesq

DATA_DIR = pathlib.Path(__file__).parent / "samples"


random.seed(42)
np.random.seed(42)
torch.torch.manual_seed(42)


@pytest.fixture(params=DATA_DIR.glob("speech/*.flac"))
def speech(request, device):
    return torchaudio.load(request.param)[0].to(device)


@pytest.fixture(params=DATA_DIR.glob("noise/*.wav"))
def noise(request, device):
    return torchaudio.load(request.param)[0].to(device)


@pytest.fixture(params=["cuda", pytest.param("cpu", marks=pytest.mark.slow)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("No GPU installed")
    return request.param


def batched_pesq(ref, deg):
    def fnc(a, b):
        return pesq(16000, np.asarray(a.squeeze(0)), np.asarray(b), mode="wb")

    result = []
    result.extend(Parallel(n_jobs=-1)(delayed(fnc)(ref.cpu(), x) for x in deg.cpu()))

    return torch.as_tensor(result).to(ref.device)


def test_abs_error(speech, noise, device):
    loss = PesqLoss(1.0, sample_rate=16000).to(device)

    if noise.shape[1] > speech.shape[1]:
        noise = noise[:, : speech.shape[1]]

    steps = torch.linspace(0.00, 0.7, 50).unsqueeze(1).to(device)
    degraded = (1 - steps) * speech + steps * noise

    vals = loss.mos(speech.expand(50, -1), degraded)
    target = batched_pesq(speech, degraded)

    assert np.allclose(vals.cpu(), target.cpu(), atol=0.17)


def test_correlation(speech, noise, device):
    loss = PesqLoss(1.0, sample_rate=16000).to(device)

    if noise.shape[1] > speech.shape[1]:
        noise = noise[:, : speech.shape[1]]

    steps = torch.linspace(0.00, 0.7, 50).unsqueeze(1).to(device)
    degraded = (1 - steps) * speech + steps * noise

    vals = loss.mos(speech.expand(50, -1), degraded)
    target = batched_pesq(speech, degraded)

    val, p = scipy.stats.pearsonr(target.cpu(), vals.cpu())

    assert val > 0.97
    assert p < 0.05 / 2000.0
