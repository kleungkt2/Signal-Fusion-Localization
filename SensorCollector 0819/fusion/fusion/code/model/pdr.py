import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


class PDR:
    def __init__(self, map_scale=1, map_rot=0):
        self.map_scale = map_scale
        self.theta = map_rot - np.pi/2
        self.heading = 0
        self.acc_seq = np.empty(0)
        self.total_step = 0
        self.last_ts = None
        self.last_v = np.zeros(2)

    def get_acc(self, acc, mag, gravity):
        _, _, _, rot_mat = self.get_orientation(np.mean(mag, 0), np.mean(gravity, 0))
        acc = rot_mat.dot(np.mean(acc, 0))
        return np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]]) \
            .dot(acc[:2] * [-1, 1])

    def get_speed(self, acc, mag, gravity, ts, last_v=None):
        if last_v is not None:
            self.last_v = last_v

        if self.last_ts is None:
            delta_v = np.zeros(2)
        else:
            _, _, _, rot_mat = self.get_orientation(np.mean(mag, 0), np.mean(gravity, 0))
            delta_t = (ts - self.last_ts) / 1000 / len(acc)
            delta_v = np.zeros(2)
            for a in acc:
                acc_w = rot_mat.dot(a)
                delta_v += acc_w[:2] * delta_t
            delta_v = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])\
                .dot(delta_v * [-1, 1])
        self.last_v += delta_v
        self.last_ts = ts
        return self.last_v, delta_v

    def get_travel_distance(self, acc):
        self.acc_seq = np.concatenate((self.acc_seq, np.linalg.norm(acc, axis=1)))
        step, last_peak = self._count_step(self.acc_seq, H=9.5, W=5, D=20)
        if step > 0:
            stride_length = self._stride_length_estimation()
            # print(stride_length)
            self.acc_seq = self.acc_seq[last_peak:]
            self.total_step += step
            return step * stride_length * self.map_scale
        else:
            return 0

    def get_orientation(self, mag, gravity):
        G = gravity / np.linalg.norm(gravity)
        B = mag
        H = np.cross(B, G)
        H /= np.linalg.norm(H)

        B /= np.linalg.norm(mag)
        M = np.cross(G, H)

        azimuth = np.arctan2(H[1], M[1])
        pitch = np.arcsin(-G[1])
        roll = np.arctan2(-G[0], G[2])
        rot_mat = np.array([H, M, G])

        return azimuth, pitch, roll, rot_mat

    def get_heading_change(self, heading):
        h = np.mean(heading, 0)[0]
        dh = (h - self.heading) * np.pi / 180
        self.heading = h
        return dh

    def _count_step(self, values, H=9.5, L=0, W=5, D=20):
        peaks, properties = find_peaks(values, height=H, width=W, distance=D)
        return len(peaks), None if len(peaks) == 0 else peaks[-1]

    def _stride_length_estimation(self):
        return 0.4 * np.power(np.max(self.acc_seq,0) - np.min(self.acc_seq, 0), 0.25)
