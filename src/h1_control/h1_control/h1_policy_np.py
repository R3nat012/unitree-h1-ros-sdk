"""Unitree's pretrained H1 walking policy (unitree_rl_gym deploy/pre_train/h1/motion.pt)
re-implemented in numpy, so it runs without PyTorch.

Network: LSTM(41 -> 64) -> Linear(64, 32) -> ELU -> Linear(32, 10).
Weights were pulled out of the TorchScript file into h1_policy.npz.
"""
import numpy as np


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


class H1Policy:
    def __init__(self, npz_path):
        w = np.load(npz_path)
        self.w = {k: w[k].astype(np.float64) for k in w.files}
        self.reset()

    def reset(self):
        self.h = np.zeros(64)
        self.c = np.zeros(64)

    def __call__(self, obs):
        w = self.w
        g = (w["memory.weight_ih_l0"] @ obs + w["memory.bias_ih_l0"]
             + w["memory.weight_hh_l0"] @ self.h + w["memory.bias_hh_l0"])
        i, f, gg, o = np.split(g, 4)                 # PyTorch gate order
        self.c = _sigmoid(f) * self.c + _sigmoid(i) * np.tanh(gg)
        self.h = _sigmoid(o) * np.tanh(self.c)
        y = w["actor.0.weight"] @ self.h + w["actor.0.bias"]
        y = np.where(y > 0, y, np.expm1(y))          # ELU
        return w["actor.2.weight"] @ y + w["actor.2.bias"]
