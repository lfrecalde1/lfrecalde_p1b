import argparse
from pathlib import Path
import numpy as np
from scipy import io
from functions.plots_p0 import plot_samples, plot_acc, plot_gyro, plot_angles, plot_all_methods, plot_quaternions
from functions.plots_p0 import plot_all_methods_new
from functions.video import make_orientation_video
from functions.ukf import UKF
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp
import casadi as ca

def _overlap_window(t1, t2):
    t0 = max(t1.min(), t2.min())
    t1_ = min(t1.max(), t2.max())
    if t1_ <= t0:
        raise ValueError("No time overlap between the two sequences.")
    return t0, t1_

def _median_dt(t):
    return float(np.median(np.diff(t)))

def _choose_target_time(t_imu, t_vicon, prefer="denser"):
    t0, t1 = _overlap_window(t_imu, t_vicon)

    imu_in = (t_imu >= t0) & (t_imu <= t1)
    vic_in = (t_vicon >= t0) & (t_vicon <= t1)

    if prefer == "imu":
        return t_imu[imu_in]
    if prefer == "vicon":
        return t_vicon[vic_in]

    # prefer == "denser"
    imu_dt = _median_dt(t_imu[imu_in])
    vic_dt = _median_dt(t_vicon[vic_in])
    # smaller dt = denser
    if imu_dt <= vic_dt:
        print("Imu TIME")
        print(imu_dt)
        return t_imu[imu_in]
    else:
        print("Vicon TIME")
        print(vic_dt)
        return t_vicon[vic_in]
def interp_linear_timeseries(t_src, X_src, t_tgt):
    t_src = np.asarray(t_src).ravel()
    t_tgt = np.asarray(t_tgt).ravel()

    if X_src.ndim == 1:
        return np.interp(t_tgt, t_src, X_src)

    D, N = X_src.shape
    X_out = np.empty((D, t_tgt.size), dtype=float)
    for i in range(D):
        X_out[i, :] = np.interp(t_tgt, t_src, X_src[i, :])
    return X_out

def slerp_rotmats(t_src, R_src_3x3xN, t_tgt):
    N = R_src_3x3xN.shape[2]
    # Build Rotation sequence
    Rs = R.from_matrix(np.transpose(R_src_3x3xN, (2, 0, 1)))  # (N,3,3)
    slerp = Slerp(np.asarray(t_src).ravel(), Rs)
    Rtgt = slerp(np.asarray(t_tgt).ravel())  # Rotation object at target times
    return Rtgt

def angles_from_rotation_obj_xyz(Robj):
    eul_zyx = Robj.as_euler('xyz', degrees=False)
    rpy = np.vstack([eul_zyx[:, 2], eul_zyx[:, 1], eul_zyx[:, 0]])
    return rpy

def quat_from_matrix(Rm):
    Rm = Rm.as_matrix()
    q = np.zeros((4, Rm.shape[0]))
    for k in range(0, Rm.shape[0]):
        r = R.from_matrix(Rm[k, :, :])
        x, y, z, w = r.as_quat()   # scipy returns (x, y, z, w)
        q[:, k] = np.array([w, x, y, z], dtype=float)
        # ensure unit norm (guards against tiny numerical drift)
        q[:, k] /= np.linalg.norm(q[:, k])

    # --- Fix double-cover discontinuities (sign flips) ---
    for k in range(1, Rm.shape[0]):
        if np.dot(q[:, k-1], q[:, k]) < 0.0:
            q[:, k] *= -1.0
    return q

def scale_measurements(imu, parameters):
    scales = parameters[0, :]
    biases = parameters[1, :]

    # Acc
    imu_filtered_empty = np.zeros_like(imu[0:3, :], dtype=float)

    # Gyro
    gyro_filtered_empy = np.zeros_like(imu[3:6, :], dtype=float)
    gyro_mean = np.mean(imu[3:6, 0:200], axis=1)
    
    # Filter Data For Loop
    for k in range(0, imu_filtered_empty.shape[1]):
        # Acc
        imu_filtered_empty[0, k] = ((imu[0, k]*scales[0]) + biases[0])*9.8
        imu_filtered_empty[1, k] = ((imu[1, k]*scales[1]) + biases[1])*9.8
        imu_filtered_empty[2, k] = ((imu[2, k]*scales[2]) + biases[2])*9.8

        # Gyro
        gyro_filtered_empy[:, k] = (3300.0/1023.0)*(np.pi/180.0)*(0.3)*(imu[3:6, k] - gyro_mean)

    return imu_filtered_empty, gyro_filtered_empy

def angles_from_acc(acc):
    rpy = np.zeros_like(acc)

    for k in range(0, rpy.shape[1]):
        rpy[2, k] = np.arctan((np.sqrt(acc[0, k]**2 + acc[1, k]**2))/acc[2, k])
        rpy[0, k] = np.arctan2(acc[1, k], acc[2, k])
        rpy[1, k] = -np.arctan2(acc[0, k], np.sqrt(acc[1, k]**2 + acc[2, k]**2))

    return rpy


def angles_from_rot(rot):
    M = rot.shape[2]
    rpy = np.zeros((3, M))
    for k in range(M):
        eul_zyx = R.from_matrix(rot[:, :, k]).as_euler('xyz', degrees=False)
        rpy[:, k] = np.array([eul_zyx[2], eul_zyx[1], eul_zyx[0]])
    return rpy

def euler_dot(euler, omega_imu):
    roll, pitch, yaw = euler.flatten()
    p_imu, q_imu, r_imu = omega_imu.flatten()

    # Map IMU gyro to body/Vicon frame
    p = p_imu
    q = q_imu
    r = r_imu

    sphi, cphi = np.sin(roll), np.cos(roll)
    tth, cth = np.tan(pitch), np.cos(pitch)

    # Transformation matrix from body rates -> Euler rates
    T = np.array([
        [1.0, sphi*tth, cphi*tth],
        [0.0, cphi,    -sphi],
        [0.0, sphi/cth, cphi/cth]
    ])

    omega_b = np.array([[p], [q], [r]])
    return T @ omega_b  # (3,1)

def integrate_gyro_euler(gyro, rpy0, t):
    M = gyro.shape[1]
    rpy = np.zeros((3, M))
    rpy[:, 0] = rpy0
    for k in range(M-1):
        dt = (t[k+1] - t[k])
        x  = rpy[:, k].reshape(3, 1)
        u = np.array([gyro[0, k], -gyro[1, k], -gyro[2, k]])
        u  = u.reshape(3, 1)

        k1 = euler_dot(x,            u)
        k2 = euler_dot(x + 0.5*dt*k1, u)
        k3 = euler_dot(x + 0.5*dt*k2, u)
        k4 = euler_dot(x + dt*k3,    u)

        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        rpy[:, k+1] = x_next[:, 0]
    return rpy


def low_passs_filter(signal, parameter):
    
    filter = np.zeros_like(signal)
    gain = np.array([[1.0 - parameter, 0.0, 0.0], [0.0, 1.0-parameter, 0.0],
                     [0.0, 0.0, 1.0-parameter]])
    I = np.array([[parameter, 0.0, 0.0], [0.0, parameter, 0.0],
                     [0.0, 0.0, parameter]])
    # Init Values
    filter[:, 0] = signal[:, 0]

    for k in range (0, signal.shape[1]-1):
        filter[:, k + 1] = gain@signal[:, k + 1] + I@filter[:, k]
    return filter

def high_pass_filter(x, fc, Ts):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    tau = 1.0 / (2.0 * np.pi * fc)
    alpha = tau / (tau + Ts)
    y[:, 0] = 0.0
    for k in range(1, x.shape[1]):
        y[:, k] = alpha * (y[:, k-1] + x[:, k] - x[:, k-1])
    return y

def quat_to_euler_xyz(q):
    q_xyzw = np.column_stack([q[1, :], q[2, :], q[3, :], q[0, :]])
    r = R.from_quat(q_xyzw)
    eulers = r.as_euler('xyz', degrees=False)
    return eulers.T

def quatdot_np(q, omega, K_quat=0.0):
    q = np.asarray(q, dtype=float).reshape(4)
    w, x, y, z = q
    wx, wy, wz = np.asarray(omega, dtype=float).reshape(3)

    quat_err = 1.0 - (w*w + x*x + y*y + z*z)

    H_r_plus = np.array([
        [ w, -x, -y, -z],
        [ x,  w, -z,  y],
        [ y,  z,  w, -x],
        [ z, -y,  x,  w],
    ], dtype=float)

    omega_quat = np.array([0.0, wx, wy, wz], dtype=float)
    qdot = 0.5 * (H_r_plus @ omega_quat) + K_quat * quat_err * q
    return qdot  # (4,)

def quatdot_aux_casadi(q, omega, K_quat=0.0):
    q     = ca.reshape(ca.MX(q),     4, 1)
    omega = ca.reshape(ca.MX(omega), 3, 1)

    wx, wy, wz   = omega[0], omega[1], omega[2]
    ww           = 0.0

    quat_err = 1.0 - ca.dot(q, q)

    H_r_plus = ca.vertcat(
        ca.horzcat(ww, -wx, -wy, -wz),
        ca.horzcat(wx,  ww,  wz, -wy),
        ca.horzcat(wy, -wz,  ww,  wx),
        ca.horzcat(wz,  wy, -wx,  ww),
    )

    qdot = 0.5 * ca.mtimes(H_r_plus, q) + K_quat * quat_err * q
    return qdot

def quatdot_function():
    q_sym     = ca.MX.sym('q', 4)
    omega_sym = ca.MX.sym('omega', 3)
    K_sym     = ca.MX.sym('K', 1)
    ts = ca.MX.sym('ts', 1)

    qdot_sym  = quatdot_aux_casadi(q_sym, omega_sym, K_sym)
    dynamics_f = ca.Function('quatdot', [q_sym, omega_sym, K_sym], [qdot_sym])

    ## Integration method
    k1 = dynamics_f(q_sym, omega_sym, K_sym)
    k2 = dynamics_f(q_sym + (1/2)*ts*k1, omega_sym, K_sym)
    k3 = dynamics_f(q_sym + (1/2)*ts*k2, omega_sym, K_sym)
    k4 = dynamics_f(q_sym + ts*k3, omega_sym, K_sym)

    # Compute forward Euler method
    xk = q_sym + (1/6)*ts*(k1 + 2*k2 + 2*k3 + k4)
    casadi_kutta = ca.Function('casadi_kutta',[q_sym, omega_sym, K_sym, ts], [xk])

    ## Calculate jacobian and gradient
    dfdx_f = ca.jacobian(xk, q_sym) 
    dfdu_f = ca.jacobian(xk, omega_sym)

    df_dx = ca.Function('df_dx', [q_sym, omega_sym, K_sym, ts], [dfdx_f])
    df_du = ca.Function('df_du', [q_sym, omega_sym, K_sym, ts], [dfdu_f])
    
    return casadi_kutta, df_dx, df_du, dynamics_f

def integrate_gyro_quaternion(gyro, q0, t, dynamics, K_quat=10.0, renorm=True):
    gyro = np.asarray(gyro, float)
    t = np.asarray(t, float).reshape(-1)
    N = gyro.shape[1]
    assert gyro.shape == (3, N)
    assert t.shape[0] == N

    Q = np.zeros((4, N), dtype=float)
    q = np.asarray(q0, float).reshape(4)
    q /= np.linalg.norm(q)
    Q[:, 0] = q

    for k in range(N-1):
        dt = float(t[k+1] - t[k])
        omega = np.array([gyro[0, k], -gyro[1, k], -gyro[2, k]])
        aux = dynamics(q, omega, K_quat, dt)
        q = np.array(aux).reshape((4, ))
        if renorm:  # keep unit length
            q /= np.linalg.norm(q)

        Q[:, k+1] = q

    return Q

def simple_kalman_filter(gyro, q0, t, dynamics, A_d, B_d, rpy_acc, gain_q,
                         gain_r, K_quat=10.0):
    t = np.asarray(t, float).reshape(-1)
    H = np.eye(4, 4)

    Q = gain_q*np.eye(4, 4)
    R_m = gain_r*np.eye(4, 4)

    x = np.zeros((4, gyro.shape[1]))
    x[:, 0] = q0
    P = 1*np.eye(4, 4)

    for k in range(0, gyro.shape[1]-1):
        dt = float(t[k+1] - t[k])
        rot = R.from_euler('xyz', [rpy_acc[0, k], rpy_acc[1, k], rpy_acc[2, k]], degrees=False)
        q_xyzw = rot.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        z = q_wxyz.reshape((4,))
        
        A = A_d(x[:, k], gyro[:, k], 10, dt)
        xp = A@x[:, k]
        Pp = A@P@A.T + Q

        K = Pp@H.T@np.linalg.inv(H@Pp@H.T + R_m)
        
        aux = xp + K@(z - H@xp)
        x[:, k+1] = np.array(aux).reshape((4, ))
        P = Pp - K@H@Pp
    return x

def scale_measurements_normalized(imu, parameters):
    scales = parameters[0, :]
    biases = parameters[1, :]

    # Acc
    imu_filtered_empty = np.zeros_like(imu[0:3, :], dtype=float)

    # Gyro
    gyro_filtered_empy = np.zeros_like(imu[3:6, :], dtype=float)
    gyro_mean = np.mean(imu[3:6, 0:200], axis=1)
    
    # Filter Data For Loop
    for k in range(0, imu_filtered_empty.shape[1]):
        # Acc
        imu_filtered_empty[0, k] = ((imu[0, k]*scales[0]) + biases[0])*9.8
        imu_filtered_empty[1, k] = ((imu[1, k]*scales[1]) + biases[1])*9.8
        imu_filtered_empty[2, k] = ((imu[2, k]*scales[2]) + biases[2])*9.8

        # Gyro
        gyro_filtered_empy[:, k] = (3300.0/1023.0)*(np.pi/180.0)*(0.3)*(imu[3:6, k] - gyro_mean)

    return imu_filtered_empty, gyro_filtered_empy

def madwick_filter(gyro, q0, t, dynamics, acc, flow, K_quat=10.0, renorm=True):
    gyro = np.asarray(gyro, float)
    t = np.asarray(t, float).reshape(-1)
    N = gyro.shape[1]
    Q = np.zeros((4, N), dtype=float)
    q = np.asarray(q0, float).reshape(4)
    Q[:, 0] = q

    for k in range(N-1):
        # Get sample time
        dt = float(t[k+1] - t[k])

        # Rotate angular velocity
        omega = np.array([gyro[0, k], -gyro[1, k], -gyro[2, k]])
        acc_normalized = acc[:, k]/np.linalg.norm(acc[:, k])

        # Integration quaternion Runge Kutta 4
        q_dot_w = flow(q, omega, K_quat)
        q_dot_w = np.array(q_dot_w).reshape((4, 1))
        #u_t = alpha * np.linalg.norm(q_dot)*dt

        # Cost Madgwick
        f = cost_madgwick(q, acc_normalized)
        J = jacobian_madgwick(q)
        gradient = J.T@f
        q_dot_a = gradient/(np.linalg.norm(gradient))
        q_dot = q_dot_w - 1.1*q_dot_a
        q_dot = np.array(q_dot).reshape((4,))
    
        # Integration method
        q = q + q_dot*dt
        q = q/np.linalg.norm(q)
        q = np.array(q).reshape((4, ))
        Q[:, k+1] = q

    return Q

def cost_madgwick(q, a):
    # Split Quaternions
    q_w = q[0]
    q_x = q[1]
    q_y = q[2]
    q_z = q[3]

    # split acceleration
    a_x = a[0]
    a_y = a[1]
    a_z = a[2]

    f11 = 2*q_x*q_z - 2*q_w*q_y - a_x
    f21 = 2*q_w*q_x - a_y + 2*q_y*q_z
    f31 = q_w**2 - q_x**2 - q_y**2 + q_z**2 - a_z

    f = np.array([[f11], [f21], [f31]])

    return f

def jacobian_madgwick(q):
    # Split Quaternions
    q_w = q[0]
    q_x = q[1]
    q_y = q[2]
    q_z = q[3]

    j11 = -2*q_y
    j12 = 2*q_z
    j13 = -2*q_w
    j14 = 2*q_x

    j21 =  2*q_x
    j22 = 2*q_w
    j23 = 2*q_z
    j24 = 2*q_y

    j31 = 2*q_w
    j32 = -2*q_x
    j33 = -2*q_y
    j34 = 2*q_z

    J = np.array([[j11, j12, j13, j14], [j21, j22, j23, j24], [j31, j32, j33,
                                                               j34]])
    return J

def loadmat_optional(path: Path):
    """Load a .mat only if it exists; else return None."""
    if path is None:
        return None
    if path.exists():
        return io.loadmat(path)
    return None

def filters_estimation(imu_data, imu_params, imu_ts, vicon_data, vicon_ts,
                       flag):
    if flag:
        # Scaled IMU Data
        acc_data_filtered, gyro_data_filtered = scale_measurements(imu_data, imu_params)

        # Check the common time grid 
        t_sync = _choose_target_time(imu_ts[0, :].ravel(), vicon_ts[0, :].ravel(), prefer="denser")

        # IMU (acc & gyro) are linear-interpolated per axis
        acc_sync  = interp_linear_timeseries(imu_ts[0, :],  acc_data_filtered,  t_sync)   
        gyro_sync = interp_linear_timeseries(imu_ts[0, :],  gyro_data_filtered, t_sync)   

        # Interpolation of the rotation matrices 
        R_sync = slerp_rotmats(vicon_ts[0, :], vicon_data, t_sync)

        # Compute euler angles from vicon
        quaternion_vicon_sync = quat_from_matrix(R_sync)
        rpy_vicon_sync = quat_to_euler_xyz(quaternion_vicon_sync)

        # Angles from acc
        rpy_acc_sync = angles_from_acc(acc_sync)

        # Dystem dynamics and A matrix
        dynamics, df_dx, df_du, flow = quatdot_function()

        # Integral gyro using quaternion dot
        q0 = quaternion_vicon_sync[:, 0]
        q_gyro_sync = integrate_gyro_quaternion(gyro_sync, q0, t_sync, dynamics)
        rpy_gyro_quat = quat_to_euler_xyz(q_gyro_sync)

        q_complementary = simple_kalman_filter(gyro_sync, q0, t_sync, dynamics, df_dx, df_du,
                             rpy_acc_sync, 0.000001, 1000000,  K_quat=10.0)
        rpy_complementary = quat_to_euler_xyz(q_complementary)

        ## ---------------------------- P1 -----------------------------------
        acc_data_normalized, gyro_data_filtered = scale_measurements_normalized(imu_data, imu_params)

        t_sync = _choose_target_time(imu_ts[0, :].ravel(), vicon_ts[0, :].ravel(), prefer="denser")

        # IMU (acc & gyro) are linear-interpolated per axis
        acc_sync  = interp_linear_timeseries(imu_ts[0, :],  acc_data_normalized,  t_sync)   
        gyro_sync = interp_linear_timeseries(imu_ts[0, :],  gyro_data_filtered, t_sync)   

        q0 = quaternion_vicon_sync[:, 0]
        q_madgwick = madwick_filter(gyro_sync, q0, t_sync, dynamics, acc_sync, flow, K_quat=10.0, renorm=True)
        rpy_madgwick = quat_to_euler_xyz(q_madgwick)
        return rpy_acc_sync, rpy_gyro_quat, rpy_complementary, rpy_madgwick, rpy_vicon_sync, t_sync
    else:
        # Scaled IMU Data
        acc_data_filtered, gyro_data_filtered = scale_measurements(imu_data, imu_params)
        
        t_sync = imu_ts[0, :]

        # IMU (acc & gyro) are linear-interpolated per axis
        acc_sync  = acc_data_filtered
        gyro_sync = gyro_data_filtered

        # Angles from acc
        rpy_acc_sync = angles_from_acc(acc_sync)

        # Dystem dynamics and A matrix
        dynamics, df_dx, df_du, flow = quatdot_function()

        # Integral gyro using quaternion dot
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q_gyro_sync = integrate_gyro_quaternion(gyro_sync, q0, t_sync, dynamics)
        rpy_gyro_quat = quat_to_euler_xyz(q_gyro_sync)

        q_complementary = simple_kalman_filter(gyro_sync, q0, t_sync, dynamics, df_dx, df_du,
                             rpy_acc_sync, 0.000001, 1000000,  K_quat=10.0)
        rpy_complementary = quat_to_euler_xyz(q_complementary)

        ## ---------------------------- P1 -----------------------------------
        acc_data_normalized, gyro_data_filtered = scale_measurements_normalized(imu_data, imu_params)

        q0 = q0
        q_madgwick = madwick_filter(gyro_sync, q0, t_sync, dynamics, acc_sync, flow, K_quat=10.0, renorm=True)
        rpy_madgwick = quat_to_euler_xyz(q_madgwick)
        return rpy_acc_sync, rpy_gyro_quat, rpy_complementary, rpy_madgwick, t_sync
def deg2rad(d): return d * np.pi / 180.0

def q_normalize(q):
    q = np.asarray(q).reshape(4)
    return q / (np.linalg.norm(q) + 1e-15)

def q_mul(q1, q2):  # scalar-first
    w, x, y, z = q1
    ww, wx, wy, wz = q2
    return np.array([
        w*ww - x*wx - y*wy - z*wz,
        w*wx + x*ww + y*wz - z*wy,
        w*wy - x*wz + y*ww + z*wx,
        w*wz + x*wy - y*wx + z*ww
    ])

def q_exp(d):
    th = np.linalg.norm(d)
    if th < 1e-12:
        return np.array([1.0, 0.5*d[0], 0.5*d[1], 0.5*d[2]])
    s = np.sin(0.5*th)/th
    return np.array([np.cos(0.5*th), s*d[0], s*d[1], s*d[2]])

def q_log(q):
    q = q_normalize(q)
    s, v = q[0], q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros(3)
    ang = np.arctan2(nv, s)
    return (2.0*ang/nv) * v

def boxplus(q, d):
    return q_normalize(q_mul(q, q_exp(d)))

def boxminus(q1, q2):
    q2inv = np.array([q2[0], -q2[1], -q2[2], -q2[3]])
    return q_log(q_mul(q2inv, q1))

def R_of_q(q):
    q = q_normalize(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def quat_weighted_mean(Qs, Wm, eps=1e-9, iters=20):
    q_bar = q_normalize(Qs[int(np.argmax(Wm))])
    for _ in range(iters):
        Qs_align = np.array([q if np.dot(q, q_bar) >= 0.0 else -q for q in Qs])
        E = np.array([boxminus(qi, q_bar) for qi in Qs_align])  # (2n+1,3)
        Delta = (Wm[:, None] * E).sum(axis=0)
        if np.linalg.norm(Delta) < eps:
            break
        q_bar = boxplus(q_bar, Delta)
    return q_bar

def filters_ukf(imu_data, imu_params, imu_ts, vicon_data, vicon_ts,
                       flag):
    if flag:
        
        acc_data_normalized, gyro_data_filtered = scale_measurements_normalized(imu_data, imu_params)
        t_sync = _choose_target_time(imu_ts[0, :].ravel(), vicon_ts[0, :].ravel(), prefer="denser")
    else:
        
        acc_data_normalized, gyro_data_filtered = scale_measurements_normalized(imu_data, imu_params)
        t_sync = imu_ts[0, :]

    acc_sync  = interp_linear_timeseries(imu_ts[0, :], acc_data_normalized,  t_sync)
    gyro_sync = interp_linear_timeseries(imu_ts[0, :], gyro_data_filtered,  t_sync)

    # ---------- State & covariances ----------
    n_tan = 6

    # Initial mean
    x_hat = np.zeros((7,))
    x_hat[0] = 1.0            # unit quaternion

    # initial gyro bias can be 0 or from a short stationary calibration
    b_sigma0 = deg2rad(0.1)/np.sqrt(3.0)

    # Initial covariance (tangent)
    sigma_rp  = deg2rad(7.0)
    sigma_yaw = deg2rad(10.0)

    P_theta0  = np.diag([sigma_rp**2, sigma_rp**2, sigma_yaw**2])
    P_b0      = (b_sigma0**2)*np.eye(3)
    P = np.block([[P_theta0,            np.zeros((3,3))],
                  [np.zeros((3,3)),     P_b0]])

    # Process noise (tangent): [attitude random walk proxy, bias RW]
    q_proc   = (deg2rad(0.2))**2        
    b_rw     = (deg2rad(0.02))**2        
    Q = np.diag([q_proc, q_proc, q_proc, b_rw, b_rw, b_rw])

    # Accelerometer direction noise (unit vector)
    Ra = (0.01**2)*np.eye(3)

    # ---------- UKF weights ----------
    W = UKF(n=n_tan, alpha=0.1, beta=2.0, kappa=0.0)
    Wm, Wc = W.weights

    q_hist = np.zeros((4, t_sync.shape[0]))
    b_hist = np.zeros((3, t_sync.shape[0]))
    P_hist = np.zeros((6, 6, t_sync.shape[0]))

    q_hist[:, 0] = x_hat[0:4]
    b_hist[:, 0] = x_hat[4:7]
    P_hist[:, :, 0] = P

    g = np.array([0.0, 0.0, 9.80665])

    for k in range(t_sync.shape[0]-1):
        dt = float(t_sync[k+1] - t_sync[k])

        q_bar = x_hat[0:4].copy()
        b_bar = x_hat[4:7].copy()
        omega_m = gyro_sync[:, k]

        # --- Sigma generation in tangent (no Q here) ---
        Xi = W._sigma_points(P)

        # --- Retract to state space ---
        Xsig_state = np.zeros((Xi.shape[0], 7))
        for i in range(Xi.shape[0]):
            dth = Xi[i, 0:3]
            db  = Xi[i, 3:6]
            qi = boxplus(q_bar, dth)
            bi = b_bar + db
            Xsig_state[i, 0:4] = qi
            Xsig_state[i, 4:7] = bi

        # --- PROPAGATION USING GYRO MEASUREMENT (Design A) ---
        Xsig_prop = np.zeros_like(Xsig_state)

        for i in range(Xsig_state.shape[0]):
            qi = Xsig_state[i, 0:4]
            bi = Xsig_state[i, 4:7]

            # integrate with (omega_m - bias_i)
            # your W.integral signature: integral(q, omega, K, ts)
            qi_next = np.array(W.integral(qi, omega_m - bi, 0.0, dt)).reshape(4)
            qi_next = q_normalize(qi_next)

            # bias random walk (mean stays same; noise is accounted by Q)
            bi_next = bi

            Xsig_prop[i, 0:4] = qi_next
            Xsig_prop[i, 4:7] = bi_next

        # --- Predicted mean ---
        Qs = Xsig_prop[:, 0:4]
        Bs = Xsig_prop[:, 4:7]
        q_pred = quat_weighted_mean(Qs, Wm)
        b_pred = (Wm[:, None] * Bs).sum(axis=0)

        E_q = np.array([boxminus(Qs[i], q_pred) for i in range(Qs.shape[0])]) 
        E_b = Bs - b_pred                                                     
        Xi_res = np.hstack([E_q, E_b])                                       
        P_pred = np.zeros_like(P)
        for i in range(Xi_res.shape[0]):
            P_pred += Wc[i] * np.outer(Xi_res[i], Xi_res[i])
        P_pred += Q
        
        # Accelerometer update 
        z_a = acc_sync[:, k]
        z_a = z_a / (np.linalg.norm(z_a) + 1e-15)

        # Predict accel direction from each propagated sigma
        Z = []
        for i in range(Xsig_prop.shape[0]):
            Ri = R_of_q(Xsig_prop[i, 0:4])
            zi = Ri.T @ (g / np.linalg.norm(g))
            zi = zi / (np.linalg.norm(zi) + 1e-15)
            Z.append(zi)
        Z = np.asarray(Z)                      
        z_hat = (Wm[:, None] * Z).sum(axis=0)
        dZ = Z - z_hat

        # residuals around (q_pred, b_pred)
        E_q2 = np.array([boxminus(Xsig_prop[i, 0:4], q_pred) for i in range(Xsig_prop.shape[0])])
        E_b2 = Xsig_prop[:, 4:7] - b_pred
        Xi_res2 = np.hstack([E_q2, E_b2])     

        S = Ra.copy()
        Pxz = np.zeros((6, 3))
        for i in range(Z.shape[0]):
            S   += Wc[i] * np.outer(dZ[i], dZ[i])
            Pxz += Wc[i] * np.outer(Xi_res2[i], dZ[i])

        K = Pxz @ np.linalg.inv(S)
        nu = z_a - z_hat
        dx = K @ nu

        dth = dx[0:3]; db = dx[3:6]
        q_post = boxplus(q_pred, dth)
        b_post = b_pred + db
        P_post = P_pred - K @ S @ K.T

        # commit
        x_hat[0:4] = q_post
        x_hat[4:7] = b_post
        P = P_post

        # logs
        q_hist[:, k+1] = x_hat[0:4]
        b_hist[:, k+1] = x_hat[4:7]
        P_hist[:, :, k+1] = P
    rpy_ukf = quat_to_euler_xyz(q_hist)
    return rpy_ukf

def main():

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../Data", help="Root folder containing Train/Test splits")
    parser.add_argument("--split", choices=["Train", "Test"], default="Test",
                        help="Dataset split to use (Train or Test)")
    parser.add_argument("--imu_params", default="../IMUParams.mat", help="Path to IMU parameters file")
    
    # You can still override directories explicitly if you want
    parser.add_argument("--imu_dir", default=None, help="(Optional) Override IMU directory")
    parser.add_argument("--vicon_dir", default=None, help="(Optional) Override Vicon directory")
    
    # Experiment number
    parser.add_argument("--exp_num", type=int, choices=range(1, 7), default=10,
                        help="Experiment number (1–6).")
    
    args = parser.parse_args()
    
    # Resolve directories: use overrides if provided, otherwise build from root/split
    imu_dir = Path(args.imu_dir) if args.imu_dir else Path(args.data_root) / args.split / "IMU"
    vicon_dir = Path(args.vicon_dir) if args.vicon_dir else Path(args.data_root) / args.split / "Vicon"
    
    imu_file = f"imuRaw{args.exp_num}"
    vicon_file = f"viconRot{args.exp_num}"
    
    # Full paths
    imu_path    = imu_dir / f"{imu_file}.mat"
    vicon_path  = vicon_dir / f"{vicon_file}.mat"
    params_path = Path(args.imu_params)

    
    # --- Load data ---
    imu = io.loadmat(imu_path)  # IMU must exist
    params = io.loadmat(params_path)
    
    # Vicon is optional: only load if file actually exists
    vicon = loadmat_optional(vicon_path)
    
    # --- Extract matrices ---
    imu_data = imu["vals"]
    imu_ts   = imu["ts"]

    # IMU params
    imu_params = params["IMUParams"]
    
    # Vicon (optional)
    if vicon is not None and "rots" in vicon and "ts" in vicon:
        vicon_data = vicon["rots"]
        vicon_ts   = vicon["ts"]
        flag = True
    else:
        vicon_data = None
        vicon_ts   = None
        flag = False
    rpy_ukf = filters_ukf(imu_data, imu_params, imu_ts, vicon_data, vicon_ts, flag) 

    if flag:
    ## returns: acc, gyro, complementary, madgwick, vicon, t
        rpy_ukf = filters_ukf(imu_data, imu_params, imu_ts, vicon_data, vicon_ts, flag) 
        rpy_acc_sync, rpy_gyro_quat, rpy_complementary, rpy_madgwick, rpy_vicon_sync, t_sync = filters_estimation(imu_data, imu_params, imu_ts, vicon_data, vicon_ts, flag)
        plot_all_methods_new(t_sync, rpy_acc_sync,                  # acc
            time_rot=t_sync, rpy_rot=rpy_vicon_sync,           # vicon (present)
            time_gyro=t_sync, rpy_gyro=rpy_gyro_quat,          # gyro
            time_complement=t_sync, rpy_complement=rpy_complementary,  # complementary
            time_madgwick=t_sync, rpy_madgwick=rpy_madgwick,
            time_ukf=t_sync, rpy_ukf=rpy_ukf,
            name=f"Results{args.exp_num}")
    else:

        # returns: acc, gyro, complementary, madgwick, t
        rpy_ukf = filters_ukf(imu_data, imu_params, imu_ts, vicon_data, vicon_ts, flag) 
        rpy_acc_sync, rpy_gyro_quat, rpy_complementary, rpy_madgwick, t_sync = filters_estimation(imu_data, imu_params, imu_ts, vicon_data, vicon_ts, flag)

        plot_all_methods_new(t_sync, rpy_acc_sync,                  # acc
            time_rot=None, rpy_rot=None,           # no vicon
            time_gyro=t_sync, rpy_gyro=rpy_gyro_quat,
            time_complement=t_sync, rpy_complement=rpy_complementary,
            time_madgwick=t_sync, rpy_madgwick=rpy_madgwick,
            time_ukf=t_sync, rpy_ukf=rpy_ukf,
            name=f"Results{args.exp_num}")
    
    # Video
    #make_orientation_video(t_sync, rpy_vicon_sync, rpy_acc_sync, rpy_gyro_quat,
                           #rpy_complementary, out_path="orientations.mp4", euler_order="xyz", degrees=False, fps=None)


if __name__ == "__main__":
    main()
