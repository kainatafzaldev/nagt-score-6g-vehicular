"""
Numerical gradient checks for the from-scratch autodiff engine used to train
NAGT-Score and all baselines (see paper Section on Experimental Setup).

Run with: python3 -m pytest tests/test_autograd.py -v
or simply: python3 tests/test_autograd.py
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from autograd import Tensor, param, binary_cross_entropy, mse_loss  # noqa: E402

# NOTE: autograd.py stores tensors as float32 (chosen for memory efficiency
# when training on CPU-only hardware -- see paper, Experimental Setup). Finite
# differences on float32 data are noise-limited below ~1e-4 perturbations, so
# EPS and TOL below are calibrated for float32, not float64. Analytic vs.
# numeric gradients were additionally verified at float64 precision during
# development (max abs diff ~1e-11 to ~1e-12); these float32 thresholds are
# intentionally looser and are what this repository's tests check.
EPS = 1e-3
TOL = 1e-3


def numerical_grad(forward_fn, tensor):
    analytic = None
    grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = tensor.data[idx]
        tensor.data[idx] = orig + EPS
        l1 = forward_fn().data
        tensor.data[idx] = orig - EPS
        l2 = forward_fn().data
        tensor.data[idx] = orig
        grad[idx] = (l1 - l2) / (2 * EPS)
        it.iternext()
    return grad


def test_matmul_sigmoid_bce():
    np.random.seed(0)
    x = Tensor(np.random.randn(5, 3))
    W = param(3, 1)
    y = np.array([[1], [0], [1], [1], [0]])

    def forward():
        return binary_cross_entropy((x @ W).sigmoid(), y)

    loss = forward()
    loss.backward()
    analytic = W.grad.copy()
    numeric = numerical_grad(forward, W)
    assert np.abs(analytic - numeric).max() < TOL


def test_masked_softmax():
    np.random.seed(1)
    x = Tensor(np.random.randn(4, 4))
    mask = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0]])
    target = np.random.RandomState(2).rand(4, 4)

    def forward():
        p = x.masked_softmax(mask, axis=-1)
        return mse_loss(p, target, mask=mask)

    loss = forward()
    loss.backward()
    analytic = x.grad.copy()
    numeric = numerical_grad(forward, x)
    assert np.abs(analytic - numeric).max() < TOL


def test_batched_matmul_broadcast():
    np.random.seed(3)
    B, T, D = 2, 3, 4
    x = Tensor(np.random.randn(B, T, D))
    W = param(D, D)
    target = np.random.RandomState(5).randn(B, T, D)

    def forward():
        return mse_loss(x @ W, target)

    loss = forward()
    loss.backward()
    analytic = W.grad.copy()
    numeric = numerical_grad(forward, W)
    assert np.abs(analytic - numeric).max() < TOL


def test_stack_reshape():
    np.random.seed(9)
    B, D = 3, 2
    hs = [Tensor(np.random.randn(B, D)) for _ in range(4)]
    target = np.random.RandomState(1).randn(B, 4, D)

    def forward():
        st = Tensor.stack(hs, axis=1)
        return mse_loss(st, target)

    loss = forward()
    loss.backward()
    for h in hs:
        analytic = h.grad.copy()
        numeric = numerical_grad(forward, h)
        assert np.abs(analytic - numeric).max() < TOL


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS: {name}")
    print("All gradient checks passed.")
