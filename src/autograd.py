"""
Minimal reverse-mode autodiff engine over numpy arrays.
Just enough ops to build: linear layers, sigmoid/tanh/relu/leaky-relu,
masked softmax, elementwise mul/add, matmul, mean/sum reductions,
concatenation -- sufficient to implement an LSTM, a Transformer encoder
layer, a dense graph-attention (GAT) layer, and multi-head MLP outputs,
all trained end-to-end with Adam.
"""
import numpy as np

class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape

    # ---------- helpers ----------
    def _unbroadcast(self, g, shape):
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and g.shape[i] != 1:
                g = g.sum(axis=i, keepdims=True)
        return g

    # ---------- ops ----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += other._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,))
        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(-other))

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        def _backward():
            self.grad += self._unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += other._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def pow(self, p):
        out = Tensor(self.data ** p, (self,))
        def _backward():
            self.grad += out.grad * p * (self.data ** (p - 1))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        def _backward():
            g_self = out.grad @ np.swapaxes(other.data, -1, -2)
            g_other = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += self._unbroadcast(g_self, self.data.shape)
            other.grad += other._unbroadcast(g_other, other.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else None
        out = Tensor(self.data.transpose(axes), (self,))
        def _backward():
            if axes is None:
                self.grad += out.grad.transpose()
            else:
                inv = np.argsort(axes)
                self.grad += out.grad.transpose(inv)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))
        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def leaky_relu(self, slope=0.2):
        out = Tensor(np.where(self.data > 0, self.data, slope * self.data), (self,))
        def _backward():
            self.grad += out.grad * np.where(self.data > 0, 1.0, slope)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -30, 30)))
        out = Tensor(s, (self,))
        def _backward():
            self.grad += out.grad * s * (1 - s)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))
        def _backward():
            self.grad += out.grad * (1 - t ** 2)
        out._backward = _backward
        return out

    def exp(self):
        e = np.exp(np.clip(self.data, -30, 30))
        out = Tensor(e, (self,))
        def _backward():
            self.grad += out.grad * e
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def _backward():
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def masked_softmax(self, mask, axis=-1):
        # mask: numpy array, 1 = keep, 0 = mask out. Softmax along `axis`.
        neg = (1 - mask) * (-1e9)
        z = self.data + neg
        z = z - z.max(axis=axis, keepdims=True)
        e = np.exp(z) * mask
        s = e.sum(axis=axis, keepdims=True) + 1e-12
        p = e / s
        out = Tensor(p, (self,))
        def _backward():
            # jacobian of softmax: dz_i = p_i * (g_i - sum_j g_j p_j)
            g = out.grad
            dot = (g * p).sum(axis=axis, keepdims=True)
            self.grad += p * (g - dot)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), (self,))
        def _backward():
            self.grad += out.grad.reshape(orig_shape)
        out._backward = _backward
        return out

    def stack(tensors, axis=1):
        expanded = [t.reshape(*(t.data.shape[:axis] + (1,) + t.data.shape[axis:])) for t in tensors]
        return Tensor.concat(expanded, axis=axis)
    stack = staticmethod(stack)

    def concat(tensors, axis=-1):
        datas = [t.data for t in tensors]
        out = Tensor(np.concatenate(datas, axis=axis), tuple(tensors))
        sizes = [d.shape[axis] for d in datas]
        def _backward():
            idx = 0
            for t, sz in zip(tensors, sizes):
                sl = [slice(None)] * out.grad.ndim
                sl[axis] = slice(idx, idx + sz)
                t.grad += out.grad[tuple(sl)]
                idx += sz
        out._backward = _backward
        return out
    concat = staticmethod(concat)

    def getitem(self, key):
        out = Tensor(self.data[key], (self,))
        def _backward():
            np.add.at(self.grad, key, out.grad)
        out._backward = _backward
        return out
    __getitem__ = getitem

    # ---------- backward ----------
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def binary_cross_entropy(p, y, mask=None, pos_weight=1.0):
    # p, y: Tensor / ndarray, p already sigmoid-activated probabilities.
    # pos_weight > 1 up-weights the positive class, useful for imbalanced heads.
    eps = 1e-7
    pc = Tensor(np.clip(p.data, eps, 1 - eps), (p,))
    def _backward():
        p.grad += pc.grad
    pc._backward = _backward
    pd = pc.data
    yd = np.asarray(y, dtype=np.float32)
    w = yd * pos_weight + (1 - yd)
    l = -w * (yd * np.log(pd) + (1 - yd) * np.log(1 - pd))
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        l = l * m
        denom = m.sum() + 1e-8
    else:
        denom = l.size
    out = Tensor(l.sum() / denom, (pc,))
    def _backward2():
        g = -w * (yd / pd - (1 - yd) / (1 - pd)) / denom
        if mask is not None:
            g = g * np.asarray(mask, dtype=np.float32)
        pc.grad += out.grad * g
    out._backward = _backward2
    return out


def mse_loss(pred, y, mask=None):
    yd = np.asarray(y, dtype=np.float32)
    diff = pred.data - yd
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        denom = m.sum() + 1e-8
        l = ((diff ** 2) * m).sum() / denom
    else:
        m = None
        denom = diff.size
        l = (diff ** 2).sum() / denom
    out = Tensor(l, (pred,))
    def _backward():
        g = 2 * diff / denom
        if m is not None:
            g = g * m
        pred.grad += out.grad * g
    out._backward = _backward
    return out


class Adam:
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def param(*shape, scale=None):
    scale = scale or (1.0 / np.sqrt(shape[0]))
    return Tensor(np.random.randn(*shape) * scale)
