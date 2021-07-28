from numba import int32, float32, int8, float64
from numba.experimental import jitclass
from scipy.special import softmax
import numpy as np

from helpers import SHORT_ALPHABET, ALPHABET, cross_entropy


spec2 = [
    ("counts", float64[:, :]),
    ("clades", float64[:, :, :]),
    ("clip_start", int32),
    ("clade_count", int32),
    ("reference_length", int32),
    ("alphabet_size", int8),
]
@jitclass(spec2)
class ModelS_jitted:
    def __init__(self, counts, clades, clip_start=0):
        self.counts = counts
        self.clades = clades
        self.clip_start = clip_start
        self.clade_count = clades.shape[0]
        self.reference_length = clades.shape[1]
        self.alphabet_size = 4

    def mixture(self, weights, pos):
        result = np.zeros(self.alphabet_size, np.float64)
        for cnum in np.arange(self.clade_count):
            for i in np.arange(self.alphabet_size):
                result[i] += weights[cnum] * self.clades[cnum][pos][i]
        return result

    @staticmethod
    def add_errors(rs, subst_rate):
        assert rs.shape == (4,)
        result = np.zeros(4, np.float64)

        for i in range(4):
            r = rs[i]
            result[i] = (1 - subst_rate) * r + subst_rate/3 * (1 - r)

        return result

    def ce(self, weights, subst_rate):
        result = 0
        total_length = 0

        for pos in np.arange(self.reference_length):
            if pos < self.clip_start:
                continue
            total_length += 1
            cc = self.counts[pos]
            mix = self.mixture(weights, pos)
            errd_mix = self.add_errors(mix, subst_rate)
            elem = self.cross_entropy(cc, errd_mix)
            result += elem
        return result/total_length

    def target_fun(self, weights, subst_rate):
        return self.ce(weights, subst_rate) + self.normalisation(subst_rate)

    @staticmethod
    def normalisation(subst_rate):
        return 0

    @staticmethod
    def softmax(x):
        expd = np.exp(x)
        total = np.sum(expd)
        return expd/total

    @staticmethod
    def cross_entropy(a, b):
        return -np.dot(a.astype(np.float64), np.log(np.fmax(b, 1e-300)))

    def opt_fun(self, x):
        weights = self.softmax(x[:-1])
        subst_rate = x[-1]
        result = self.target_fun(weights, subst_rate)
        return result

