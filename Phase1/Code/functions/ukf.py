from typing import Tuple
import numpy as np
from scipy.linalg import cholesky
import casadi as ca


class UKF:
    """
    UKF for attitude (quaternion) + angular rate + gyro bias.
    State: x = [q (4, scalar-first), w (3), b_g (3)]
    Tangent: xi = [dtheta (3), dw (3), db (3)]  -> n = 9

    Use:
      ukf = UKF(n=9, alpha=1e-3, beta=2.0, kappa=0.0)
      q,w,b,P = q0,w0,b0,P0
      q,w,b,P,Xsig = ukf.predict(q,w,b,P,Q,dt)
      q,w,b,P      = ukf.update_gyro(q,w,b,P,Xsig,z_g,Rg)
      q,w,b,P      = ukf.update_accel(q,w,b,P,Xsig,z_a,Ra)  # z_a normalized accel dir
    """

    def __init__(self, n: int, alpha: float, beta: float, kappa: float = 0.0) -> None:
        # n should be 9 for [dtheta, dw, db]
        self.n = int(n)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.Wm: np.ndarray
        self.Wc: np.ndarray
        self.integral = self._build_quat_rk4()   # CasADi RK4 integrator for qdot
        self.sqrt = cholesky
        self._compute_weights()

    # ---------------- Weights ----------------
    @property
    def num_sigma_points(self) -> int:
        return 2 * self.n + 1

    @property
    def weights(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.Wm, self.Wc

    def _compute_weights(self) -> None:
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n
        denom = n + lam
        factor = 1.0 / (2.0 * denom)

        Wm = np.full(2 * n + 1, factor, dtype=float)
        Wc = np.full(2 * n + 1, factor, dtype=float)
        Wm[0] = lam / denom
        Wc[0] = lam / denom + (1.0 - self.alpha**2 + self.beta)

        self.Wm = Wm
        self.Wc = Wc

    # ------------- Quaternion utilities -------------
    @staticmethod
    def _q_normalize(q):
        q = np.asarray(q, dtype=float).reshape(4)
        n = np.linalg.norm(q) + 1e-15
        return q / n

    @staticmethod
    def _q_mul(q1, q2):  # scalar-first Hamilton product
        w, x, y, z = q1
        ww, wx, wy, wz = q2
        return np.array([
            w*ww - x*wx - y*wy - z*wz,
            w*wx + x*ww + y*wz - z*wy,
            w*wy - x*wz + y*ww + z*wx,
            w*wz + x*wy - y*wx + z*ww
        ])

    @staticmethod
    def _q_exp(d):  # d in R^3
        d = np.asarray(d, dtype=float).reshape(3)
        th = np.linalg.norm(d)
        if th < 1e-12:
            return np.array([1.0, 0.5*d[0], 0.5*d[1], 0.5*d[2]])
        s = np.sin(0.5*th)/th
        return np.array([np.cos(0.5*th), s*d[0], s*d[1], s*d[2]])

    def _q_log(self, q):
        q = self._q_normalize(q)
        s, v = q[0], q[1:]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return np.zeros(3)
        ang = np.arctan2(nv, s)
        return (2.0*ang/nv) * v

    def _boxplus(self, q, d):
        return self._q_normalize(self._q_mul(q, self._q_exp(d)))

    def _boxminus(self, q1, q2):
        q2inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]])
        return self._q_log(self._q_mul(q2inv, q1))

    @staticmethod
    def _q_align(q, q_ref):
        return q if float(np.dot(q, q_ref)) >= 0.0 else -q

    # ------------- Sigma points in tangent -------------
    def _sigma_points(self, P):
        """Return deltas Xi (2n+1, n) in tangent/Euclid; Xi[0]=0."""
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n
        U = cholesky((n + lam) * P, lower=False)  # no +Q here
        Xi = np.zeros((2*n + 1, n))
        for k in range(n):
            Xi[k+1,   :] =  U[:, k]
            Xi[n+k+1, :] = -U[:, k]
        return Xi

    # ------------- Retraction -------------
    def _retract_state(self, q_bar, w_bar, b_bar, delta):
        dth = delta[0:3]
        dw  = delta[3:6]
        db  = delta[6:9]
        q_i = self._boxplus(q_bar, dth)
        w_i = w_bar + dw
        b_i = b_bar + db
        return q_i, w_i, b_i

    # ------------- Process propagation -------------
    def _build_quat_rk4(self):
        q = ca.MX.sym('q', 4)
        omega = ca.MX.sym('omega', 3)
        K = ca.MX.sym('K', 1)
        ts = ca.MX.sym('ts', 1)

        wx, wy, wz = omega[0], omega[1], omega[2]
        ww = 0.0

        # small norm correction if desired
        quat_err = 1.0 - ca.dot(q, q)

        H_r_plus = ca.vertcat(
            ca.horzcat(ww, -wx, -wy, -wz),
            ca.horzcat(wx,  ww,  wz, -wy),
            ca.horzcat(wy, -wz,  ww,  wx),
            ca.horzcat(wz,  wy, -wx,  ww),
        )
        qdot = 0.5 * ca.mtimes(H_r_plus, q) + K * quat_err * q
        f = ca.Function('quatdot', [q, omega, K], [qdot])

        k1 = f(q,               omega, K)
        k2 = f(q + 0.5*ts*k1,   omega, K)
        k3 = f(q + 0.5*ts*k2,   omega, K)
        k4 = f(q + ts*k3,       omega, K)
        qn = q + (ts/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return ca.Function('rk4_q', [q, omega, K, ts], [qn])

    def _propagate_sigma(self, q, w, b, dt):
        # Design B: integrate with state ω, not the gyro measurement
        q_next = np.array(self.integral(q, w, 0.0, dt)).reshape(4)
        q_next = self._q_normalize(q_next)
        # Constant models for w and b; process noise comes via Q
        return q_next, w.copy(), b.copy()

    # ------------- Quaternion weighted mean -------------
    def _quat_weighted_mean(self, Qs, Wm, eps=1e-9, iters=20):
        q_bar = self._q_normalize(Qs[int(np.argmax(Wm))])
        for _ in range(iters):
            Qs_align = np.array([self._q_align(self._q_normalize(q), q_bar) for q in Qs])
            E = np.array([self._boxminus(qi, q_bar) for qi in Qs_align])  # (2n+1,3)
            Delta = (Wm[:, None] * E).sum(axis=0)
            if np.linalg.norm(Delta) < eps:
                break
            q_bar = self._boxplus(q_bar, Delta)
        return q_bar

    # ------------- Prediction step -------------
    def predict(self, q_bar, w_bar, b_bar, P, Q, dt):
        """
        Inputs:
          q_bar (4,), w_bar (3,), b_bar (3,), P (9x9), Q (9x9), dt
        Returns:
          q_pred, w_pred, b_pred, P_pred, Xsig (2n+1, 10)
        """
        Xi = self._sigma_points(P)  # (2n+1, 9)
        # retract + propagate each sigma
        Xsig = []
        for i in range(Xi.shape[0]):
            qi, wi, bi = self._retract_state(q_bar, w_bar, b_bar, Xi[i])
            qi, wi, bi = self._propagate_sigma(qi, wi, bi, dt)
            Xsig.append(np.hstack([qi, wi, bi]))
        Xsig = np.asarray(Xsig)  # (2n+1, 10)

        # Predicted mean
        Qs = Xsig[:, 0:4]
        Ws = Xsig[:, 4:7]
        Bs = Xsig[:, 7:10]

        q_pred = self._quat_weighted_mean(Qs, self.Wm)
        w_pred = (self.Wm[:, None] * Ws).sum(axis=0)
        b_pred = (self.Wm[:, None] * Bs).sum(axis=0)

        # Residuals in tangent ⊕ Euclid
        E_q = np.array([self._boxminus(Qs[i], q_pred) for i in range(Qs.shape[0])])  # (2n+1,3)
        E_w = Ws - w_pred
        E_b = Bs - b_pred
        Xi_res = np.hstack([E_q, E_w, E_b])  # (2n+1, 9)

        P_pred = np.zeros_like(P)
        for i in range(Xi_res.shape[0]):
            P_pred += self.Wc[i] * np.outer(Xi_res[i], Xi_res[i])
        P_pred += Q
        return q_pred, w_pred, b_pred, P_pred, Xsig

    # ------------- Gyro update: z_g = w + b + n_g -------------
    def update_gyro(self, q_bar, w_bar, b_bar, P, Xsig, z_g, Rg):
        """
        z_g: gyro measurement (rad/s), shape (3,)
        Rg: 3x3 gyro noise covariance
        """
        # Predict measurement for each sigma
        Z = Xsig[:, 4:7] + Xsig[:, 7:10]      # (2n+1,3)
        z_hat = (self.Wm[:, None] * Z).sum(axis=0)

        dZ = Z - z_hat
        # State residuals around the current mean (tangent ⊕ Euclid)
        E_q = np.array([self._boxminus(Xsig[i, 0:4], q_bar) for i in range(Xsig.shape[0])])
        E_w = Xsig[:, 4:7] - w_bar
        E_b = Xsig[:, 7:10] - b_bar
        Xi_res = np.hstack([E_q, E_w, E_b])  # (2n+1,9)

        S = Rg.copy()
        Pxz = np.zeros((P.shape[0], 3))
        for i in range(Z.shape[0]):
            S   += self.Wc[i] * np.outer(dZ[i], dZ[i])
            Pxz += self.Wc[i] * np.outer(Xi_res[i], dZ[i])

        K = Pxz @ np.linalg.inv(S)
        nu = z_g - z_hat

        dx = K @ nu                      # (9,)
        dth, dw, db = dx[0:3], dx[3:6], dx[6:9]

        q_upd = self._boxplus(q_bar, dth)
        w_upd = w_bar + dw
        b_upd = b_bar + db
        P_upd = P - K @ S @ K.T
        return q_upd, w_upd, b_upd, P_upd

    # ------------- Accel update: gravity direction -------------
    @staticmethod
    def _R_of_q(q):
        q = q / (np.linalg.norm(q) + 1e-15)
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
        ])

    def update_accel(self, q_bar, w_bar, b_bar, P, Xsig, z_a, Ra,
                     g=np.array([0, 0, 9.80665])):
        """
        z_a: accelerometer measurement, should be normalized to a unit vector (3,)
        Ra: 3x3 accel direction noise covariance (tune small, since normalized)
        """
        gdir = g / (np.linalg.norm(g) + 1e-15)
        Z = []
        for i in range(Xsig.shape[0]):
            Ri = self._R_of_q(Xsig[i, 0:4])
            zi = Ri.T @ gdir            # body-frame gravity direction
            zi = zi / (np.linalg.norm(zi) + 1e-15)
            Z.append(zi)
        Z = np.asarray(Z)  # (2n+1,3)

        z_hat = (self.Wm[:, None] * Z).sum(axis=0)
        dZ = Z - z_hat

        E_q = np.array([self._boxminus(Xsig[i, 0:4], q_bar) for i in range(Xsig.shape[0])])
        E_w = Xsig[:, 4:7] - w_bar
        E_b = Xsig[:, 7:10] - b_bar
        Xi_res = np.hstack([E_q, E_w, E_b])

        S = Ra.copy()
        Pxz = np.zeros((P.shape[0], 3))
        for i in range(Z.shape[0]):
            S   += self.Wc[i] * np.outer(dZ[i], dZ[i])
            Pxz += self.Wc[i] * np.outer(Xi_res[i], dZ[i])

        K = Pxz @ np.linalg.inv(S)
        nu = z_a - z_hat

        dx = K @ nu
        dth, dw, db = dx[0:3], dx[3:6], dx[6:9]
        q_upd = self._boxplus(q_bar, dth)
        w_upd = w_bar + dw
        b_upd = b_bar + db
        P_upd = P - K @ S @ K.T
        return q_upd, w_upd, b_upd, P_upd