from matplotlib import path
import numpy as np


class ParticleFilter:
    def __init__(self, n, rp_locs, map_constraints, std=(0.5, 0.1)):
        self.n = n
        self.rp_locs = rp_locs
        self.in_path = path.Path([(x1, y1) for x1, y1, _, _ in map_constraints[0]])
        self.out_paths = [path.Path([(x1, y1) for x1, y1, _, _ in out_constraint]) for out_constraint in map_constraints[1]]
        self.std = std
        self.weights = np.repeat(1 / n, n)
        self._init_uniform_particles()

    def _init_uniform_particles(self):
        self.particles = np.empty((self.n, 4))
        indices = np.random.choice(np.arange(len(self.rp_locs)), self.n)
        self.particles[:, :2] = np.take(self.rp_locs, indices, axis=0)
        self.particles[:, 2] = np.random.uniform(0, 2 * np.pi, self.n)
        self.particles[:, 3] = indices

    def is_inside_constraints(self, locs):
        is_in = self.in_path.contains_points(locs)
        is_out = np.any([p.contains_points(locs) for p in self.out_paths], axis=0)
        return np.logical_and(is_in, ~is_out)

    def predict(self, heading, dl):
        self.particles[:, 2] = heading + np.random.normal(0, self.std[1], self.n)

        l = dl + np.random.normal(0, self.std[0], self.n)
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * l
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * l

        # is_inside = self.is_inside_constraints(self.particles[:,:2])
        # n = (~is_inside).sum()
        # indices = np.random.choice(np.arange(len(self.rp_locs)), n)
        # self.particles[np.where(~is_inside),:2] = np.take(self.rp_locs, indices, axis=0)
        # self.particles[np.where(~is_inside),2] = np.random.uniform(0, 2*np.pi, n)

        if np.random.rand() > 0.5:
            n = int(self.n * 0.1)
            indices = np.random.choice(self.n, replace=False, size=n)
            self.particles[indices, :2] = np.take(self.rp_locs, np.random.choice(np.arange(len(self.rp_locs)), n),
                                                  axis=0)
            self.particles[indices, 2] = np.random.uniform(0, 2 * np.pi, n)
            self.weights[indices] = np.mean(self.weights)

        closest_grid_indices = np.sum((self.particles[:, :2][:, np.newaxis] - np.array(self.rp_locs)) ** 2,
                                      axis=2).argmin(1)
        #         self.particles[:, :2] = np.take(self.rp_locs, closest_grid_indices, axis=0)
        self.particles[:, 3] = closest_grid_indices

    def update(self, prob):
        w = prob[self.particles[:, 3].astype(int)]

        self.weights *= w
        is_inside = self.is_inside_constraints(self.particles[:, :2])
        self.weights[np.where(~is_inside)] = 0
        self.weights += 1e-20
        self.weights /= sum(self.weights)

    def estimate(self):
        # top_k = np.argpartition(self.weights, -self.n//10)[-self.n//10:]
        # mean = np.average(self.particles[top_k, 0:2], weights=self.weights[top_k], axis=0)
        mean = np.average(self.particles[:, 0:2], weights=self.weights, axis=0)
        is_inside = self.is_inside_constraints([mean])
        if not is_inside:
            closest_grid_index = np.sum((mean.reshape(1, -1) - np.array(self.rp_locs)) ** 2, axis=1).argmin()
            mean = np.take(self.rp_locs, closest_grid_index, axis=0)
        return mean

    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        positions = (np.arange(self.n) + np.random.random()) / self.n
        indices = np.searchsorted(cumulative_sum, positions)
        self.particles[:] = self.particles[indices]
        self.particles[:, 2] += np.random.normal(0, self.std[1], self.n)
        self.particles[:, :2] += np.random.normal(0, self.std[0], (self.n, 2))
        self.weights.fill(1.0 / self.n)

    def neff(self):
        return 1 / np.sum(self.weights ** 2)


class KalmanFilter:
    def __init__(self, *, x, P, H):
        self.x = x
        self.P = P
        self.H = H

    def step(self, *, F, B, Q, u, R, z):
        self._predict(F=F, B=B, u=u, Q=Q)
        self._update(z=z, R=R)
        return self.x

    def _predict(self, *, F, B, u, Q):
        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + Q

    def _update(self, *, z, R):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(K.shape[0]) - K @ self.H) @ self.P

