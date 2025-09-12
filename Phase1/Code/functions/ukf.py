from typing import Tuple
import numpy as np
from scipy.linalg import cholesky
import scipy


class UKF:
    """
    Scaled Unscented Transform (SUT) sigma-point weights.

    Parameters
    ----------
    n : int
        Dimensionality of the state. 2n+1 weights will be generated.
    alpha : float
        Determines the spread of the sigma points around the mean (e.g., 1e-3).
    beta : float
        Incorporates prior knowledge about the distribution; 2.0 is optimal for Gaussian.
    kappa : float
        Secondary scaling parameter; often 0.0 or 3 - n.

    Attributes
    ----------
    Wm : np.ndarray
        Weights for each sigma point for the mean (shape: 2n+1,).
    Wc : np.ndarray
        Weights for each sigma point for the covariance (shape: 2n+1,).
    """

    def __init__(self, n: int, alpha: float, beta: float, kappa: float = 0.0) -> None:
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.Wm: np.ndarray
        self.Wc: np.ndarray
        self.sqrt = cholesky
        self._compute_weights()

    @property
    def num_sigma_points(self) -> int:
        """Return the number of sigma points (2n + 1)."""
        return 2 * self.n + 1

    @property
    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return both mean and covariance weights as a tuple (Wm, Wc)."""
        return self.Wm, self.Wc

    def _compute_weights(self) -> None:
        """Compute mean and covariance weights for the SUT."""
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n
        denom = n + lam

        # Common factor for non-central points
        factor = 1.0 / (2.0 * denom)

        Wm = np.full(2 * n + 1, factor, dtype=float)
        Wc = np.full(2 * n + 1, factor, dtype=float)

        Wm[0] = lam / denom
        Wc[0] = lam / denom + (1.0 - self.alpha**2 + self.beta)

        self.Wm = Wm
        self.Wc = Wc
        return None

    def _sigma_points(self, mean_x, P):
        # Dimention tanget space
        n = self.n

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = cholesky((lambda_ + n)*P, lower=False)
        U_aux = scipy.linalg.sqrtm((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0, :] = mean_x

        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1, :]   = mean_x + U[:, k]
            sigmas[n+k+1, :] = mean_x - U[:, k]
        return sigmas

    def _exp(self, w):        
        wx = w[0]
        wy = w[1]
        wz = w[2]
        alpha_w = np.sqrt(wx**2 + wy**2 + wz**2) + 1.0e-12
        s = np.cos(alpha_w/2.0)
        v_1 = np.sin(alpha_w/2.0) * (wx / alpha_w)
        v_2 = np.sin(alpha_w/2.0) * (wy / alpha_w)
        v_3 = np.sin(alpha_w/2.0) * (wz / alpha_w)
        return np.array([s, v_1, v_2, v_3])

    def quatdot_multiply(self, q1, q2):
        q1 = np.asarray(q1).reshape(4)
        w, x, y, z = q1
        ww, wx, wy, wz = np.asarray(q2).reshape(4)

        H_r_plus = np.array([
            [ w, -x, -y, -z],
            [ x,  w, -z,  y],
            [ y,  z,  w, -x],
            [ z, -y,  x,  w],
        ])

        omega_quat = np.array([ww, wx, wy, wz])
        product = (H_r_plus @ omega_quat)
        return product

    def _sigma_points_manifold(self, estimated, sigma_points):
        q_estimated = estimated[0:4]
        w_estimate = estimated[4:7]
        print(w_estimate.shape)

        # Empy vector sigma points
        sigma_points_manifold = np.zeros((sigma_points.shape[0], 7))
        for k in range(sigma_points.shape[0]):
            # Projection of the sigma points to the quaternio
            print(sigma_points[k, 0:3])
            quaternion_pojection = self.quatdot_multiply(q_estimated, self._exp(sigma_points[k, 0:3]))
            omega_projection = w_estimate + sigma_points[k, 3:6]

            # Update sigma points manifold
            sigma_points_manifold[k, 0:4] = quaternion_pojection
            sigma_points_manifold[k, 4:7] = omega_projection
        return sigma_points_manifold