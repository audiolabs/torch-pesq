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


@pytest.fixture(params=DATA_DIR.glob("speech/*.wav"))
def speech(request, device):
    return torchaudio.load(request.param)[0].to(device)


@pytest.fixture(params=DATA_DIR.glob("noise/*/*.wav"))
def noise(request, device):
    return torchaudio.load(request.param)[0].to(device)


@pytest.fixture(params=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return request.param


def batched_pesq(ref, deg):
    def fnc(a, b):
        return pesq(16000, np.asarray(a.squeeze(0)), np.asarray(b), mode="wb")

    result = []
    result.extend(Parallel(n_jobs=-1)(delayed(fnc)(ref.cpu(), x) for x in deg.cpu()))

    return torch.as_tensor(result).to(ref.device)


def test_abs_error(speech, noise, device):
    loss = PesqLoss(1.0, sample_rate=16000).to(device)

    if noise.numel() > speech.numel():
        noise = noise[:, : speech.numel()]

    steps = torch.linspace(0.00, 0.7, 50).unsqueeze(1).to(device)
    degraded = (1 - steps) * speech + steps * noise

    vals = loss.mos(speech.expand(50, -1), degraded)
    target = batched_pesq(speech, degraded)

    assert (vals - target).abs().max() < 0.17


def test_correlation(speech, noise, device):
    loss = PesqLoss(1.0, sample_rate=16000).to(device)

    if noise.numel() > speech.numel():
        noise = noise[:, : speech.numel()]

    steps = torch.linspace(0.00, 0.7, 50).unsqueeze(1).to(device)
    degraded = (1 - steps) * speech + steps * noise

    vals = loss.mos(speech.expand(50, -1), degraded)
    target = batched_pesq(speech, degraded)

    val, p = scipy.stats.pearsonr(target.cpu(), vals.cpu())
    # corr.write("{},{}\n".format(val, p))
    # corr.flush()

    # import os

    # name = os.environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    # with open("out/" + name + ".txt", "w") as f:
    #    for a, b in zip(target.cpu().numpy(), vals.cpu().numpy()):
    #        f.write("{},{}\n".format(a, b))

    assert val > 0.99
    assert p < 0.05 / 2000.0
