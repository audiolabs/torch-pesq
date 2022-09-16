from torch_pesq import PesqLoss
import torch


def test_autodiff():
    """Check that loss is differentiable"""

    pesq = PesqLoss(1.0, sample_rate=16000)

    ref, deg = torch.rand(1, 16000, requires_grad=True), torch.rand(
        1, 16000, requires_grad=True
    )
    loss = pesq(ref, deg)

    loss.backward()


def test_loss_range():
    """Check that loss is positive and zero for no degradation"""

    pesq = PesqLoss(1.0, sample_rate=16000)

    ref, deg = torch.rand(128, 16000, requires_grad=True), torch.rand(
        128, 16000, requires_grad=True
    )
    assert torch.all(pesq(ref, deg) > 0.0)
    assert torch.all(pesq(ref, ref) <= 1e-20)
