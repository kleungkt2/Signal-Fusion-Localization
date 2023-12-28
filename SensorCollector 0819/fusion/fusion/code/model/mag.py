from util.general_util import *
from fastdtw import fastdtw
from collections import OrderedDict


class MagDTWMatching:
    def __init__(self, mag_fp: OrderedDict):
        self.mag_fp = mag_fp

    def predict(self, seq, history_pos=None, const_d=100, sigma=1e4, d_range=120, reverse=True):
        likelihoods = np.zeros(len(self.mag_fp))
        for i, (rp_loc, fps) in enumerate(self.mag_fp.items()):
            dist = distance(history_pos, rp_loc) if history_pos is not None else 0
            if dist <= d_range:
                min_d = 1e99
                for fp in fps:
                    dist1, _ = fastdtw(seq, fp, dist=lambda x, y: np.abs(x - y))
                    dist2, _ = fastdtw(seq, fp[::-1], dist=lambda x, y: np.abs(x - y)) if reverse else (dist1, -1)
                    min_d = min(min_d, dist1, dist2)
                likelihoods[i] = np.exp(-1*min_d*min_d/(2*sigma*sigma))

        return likelihoods
