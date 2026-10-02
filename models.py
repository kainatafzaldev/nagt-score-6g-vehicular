import numpy as np
from autograd import Tensor, param, binary_cross_entropy, mse_loss


def linear_params(d_in, d_out):
    return param(d_in, d_out), Tensor(np.zeros((1, d_out)))


def linear(x, W, b):
    return x @ W + b


class NoiseGate:
    """Learnable noise-aware feature weighting module.
    Takes per-message noise-magnitude estimates and produces a (0,1) gate
    applied elementwise to the corresponding raw kinematic features, so the
    network learns to down-weight features the sensing pipeline reports as
    unreliable rather than trusting all raw fields equally."""
    def __init__(self, d_feat, d_hidden=16):
        self.W1, self.b1 = linear_params(d_feat, d_hidden)
        self.W2, self.b2 = linear_params(d_hidden, d_feat)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, raw, noise):
        h = linear(noise, self.W1, self.b1).relu()
        gate = linear(h, self.W2, self.b2).sigmoid()
        return raw * gate, gate


class GATLayer:
    """Single-layer dense graph-attention layer, applied independently per
    message-window subgraph with parameters shared across windows."""
    def __init__(self, d_in, d_hidden):
        self.W = param(d_in, d_hidden)
        self.a_src = param(d_hidden, 1)
        self.a_dst = param(d_hidden, 1)

    def params(self):
        return [self.W, self.a_src, self.a_dst]

    def __call__(self, x, adj):
        # x: Tensor (n, d_in); adj: ndarray (n, n) with 1 where edge/self-loop exists
        h = x @ self.W                      # (n, hidden)
        s_src = h @ self.a_src              # (n, 1)
        s_dst = h @ self.a_dst              # (n, 1)
        scores = (s_src + s_dst.transpose()).leaky_relu(0.2)  # (n, n) broadcast add
        alpha = scores.masked_softmax(adj, axis=-1)
        out = alpha @ h                     # (n, hidden)
        return out.relu()


class LSTM:
    def __init__(self, d_in, d_hidden):
        self.d_hidden = d_hidden
        self.Wi, self.Ui, self.bi = param(d_in, d_hidden), param(d_hidden, d_hidden), Tensor(np.zeros((1, d_hidden)))
        self.Wf, self.Uf, self.bf = param(d_in, d_hidden), param(d_hidden, d_hidden), Tensor(np.ones((1, d_hidden)))
        self.Wg, self.Ug, self.bg = param(d_in, d_hidden), param(d_hidden, d_hidden), Tensor(np.zeros((1, d_hidden)))
        self.Wo, self.Uo, self.bo = param(d_in, d_hidden), param(d_hidden, d_hidden), Tensor(np.zeros((1, d_hidden)))

    def params(self):
        return [self.Wi, self.Ui, self.bi, self.Wf, self.Uf, self.bf,
                self.Wg, self.Ug, self.bg, self.Wo, self.Uo, self.bo]

    def forward_steps(self, x_t_list, mask_t_list):
        """x_t_list: list of length T of Tensors (B, d_in). Returns list of h_t (B, d_hidden)."""
        B = x_t_list[0].shape[0]
        h = Tensor(np.zeros((B, self.d_hidden)))
        c = Tensor(np.zeros((B, self.d_hidden)))
        hs = []
        for x_t in x_t_list:
            i_t = (linear(x_t, self.Wi, self.bi) + h @ self.Ui).sigmoid()
            f_t = (linear(x_t, self.Wf, self.bf) + h @ self.Uf).sigmoid()
            g_t = (linear(x_t, self.Wg, self.bg) + h @ self.Ug).tanh()
            o_t = (linear(x_t, self.Wo, self.bo) + h @ self.Uo).sigmoid()
            c = f_t * c + i_t * g_t
            h = o_t * c.tanh()
            hs.append(h)
        return hs


class SelfAttentionLayer:
    """Single-head self-attention encoder layer + position-wise FFN with
    residual connections (a minimal Transformer encoder block)."""
    def __init__(self, d_model, d_ff=None):
        d_ff = d_ff or d_model * 2
        self.Wq = param(d_model, d_model)
        self.Wk = param(d_model, d_model)
        self.Wv = param(d_model, d_model)
        self.W1 = param(d_model, d_ff)
        self.b1 = Tensor(np.zeros((1, 1, d_ff)))
        self.W2 = param(d_ff, d_model)
        self.b2 = Tensor(np.zeros((1, 1, d_model)))
        self.scale = 1.0 / np.sqrt(d_model)

    def params(self):
        return [self.Wq, self.Wk, self.Wv, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, x, pad_mask):
        # x: (B, T, D); pad_mask: (B, T) 1=valid
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv
        scores = (Q @ K.transpose(0, 2, 1)) * self.scale
        B, T = pad_mask.shape
        attn_mask = pad_mask[:, None, :] * pad_mask[:, :, None]  # (B,T,T)
        attn = scores.masked_softmax(attn_mask, axis=-1)
        ctx = attn @ V
        x2 = x + ctx
        ff = (linear(x2, self.W1, self.b1)).relu()
        ff = linear(ff, self.W2, self.b2)
        out = x2 + ff
        return out


class MultiHead:
    """Four task-specific linear prediction heads on top of a shared
    representation: anomaly/failure, service degradation, propagation
    state (all binary, sigmoid), and response timing (regression)."""
    def __init__(self, d_in):
        self.heads = {}
        for name in ["anomaly", "degradation", "propagation"]:
            self.heads[name] = linear_params(d_in, 1)
        self.heads["timing"] = linear_params(d_in, 1)

    def params(self):
        ps = []
        for W, b in self.heads.values():
            ps += [W, b]
        return ps

    def __call__(self, h):
        W, b = self.heads["anomaly"]
        p_anom = linear(h, W, b).sigmoid()
        W, b = self.heads["degradation"]
        p_deg = linear(h, W, b).sigmoid()
        W, b = self.heads["propagation"]
        p_prop = linear(h, W, b).sigmoid()
        W, b = self.heads["timing"]
        y_time = linear(h, W, b)
        return p_anom, p_deg, p_prop, y_time


def multitask_loss(preds, targets, mask, weights=(1.0, 1.0, 1.0, 0.5),
                    pos_weights=(4.0, 18.0, 4.0)):
    p_anom, p_deg, p_prop, y_time = preds
    y_anom, y_deg, y_prop, y_time_t = targets
    l1 = binary_cross_entropy(p_anom, y_anom, mask, pos_weight=pos_weights[0])
    l2 = binary_cross_entropy(p_deg, y_deg, mask, pos_weight=pos_weights[1])
    l3 = binary_cross_entropy(p_prop, y_prop, mask, pos_weight=pos_weights[2])
    l4 = mse_loss(y_time, y_time_t, mask)
    total = l1 * weights[0] + l2 * weights[1] + l3 * weights[2] + l4 * weights[3]
    return total, (l1.data, l2.data, l3.data, l4.data)
