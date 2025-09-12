import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
# rotplot must be in scope exactly as you provided it
# If your rotplot uses `a3`, ensure this import exists *somewhere* before calling it:
import mpl_toolkits.mplot3d as a3
from scipy.spatial.transform import Rotation as R

def rotplot(R, currentAxes=None):
    # This is a simple function to plot the orientation
    # of a 3x3 rotation matrix R in 3-D
    # You should modify it as you wish for the project.

    lx = 3.0
    ly = 1.5
    lz = 1.0

    x = .5 * np.array([[+lx, -lx, +lx, -lx, +lx, -lx, +lx, -lx],
                       [+ly, +ly, -ly, -ly, +ly, +ly, -ly, -ly],
                       [+lz, +lz, +lz, +lz, -lz, -lz, -lz, -lz]])

    xp = np.dot(R, x);
    ifront = np.array([0, 2, 6, 4, 0])
    iback = np.array([1, 3, 7, 5, 1])
    itop = np.array([0, 1, 3, 2, 0])
    ibottom = np.array([4, 5, 7, 6, 4])

    if currentAxes:
        ax = currentAxes;
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    ax.plot(xp[0, itop], xp[1, itop], xp[2, itop], 'k-')
    ax.plot(xp[0, ibottom], xp[1, ibottom], xp[2, ibottom], 'k-')

    rectangleFront = a3.art3d.Poly3DCollection([list(zip(xp[0, ifront], xp[1, ifront], xp[2, ifront]))])
    rectangleFront.set_facecolor('r')
    ax.add_collection(rectangleFront)

    rectangleBack = a3.art3d.Poly3DCollection([list(zip(xp[0, iback], xp[1, iback], xp[2, iback]))])
    rectangleBack.set_facecolor('b')
    ax.add_collection(rectangleBack)

    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)

    return ax


def euler_to_R_series(rpy, order="xyz", degrees=False):
    rot = R.from_euler(order, rpy.T, degrees=degrees)
    return rot.as_matrix()

def make_orientation_video(t, rpy_vicon, rpy_acc, rpy_gyro, rpy_comp, out_path="orientations.mp4", euler_order="xyz", degrees=False, fps=None):
    R_vicon = euler_to_R_series(rpy_vicon, order=euler_order, degrees=degrees)
    R_acc   = euler_to_R_series(rpy_acc,   order=euler_order, degrees=degrees)
    R_gyro  = euler_to_R_series(rpy_gyro,  order=euler_order, degrees=degrees)
    R_comp  = euler_to_R_series(rpy_comp,  order=euler_order, degrees=degrees)

    T = len(t)
    if fps is None:
        dt = float(np.median(np.diff(t))) if T > 1 else 0.02
        fps = int(np.clip(round(1.0 / max(dt, 1e-6)), 5, 60))

    fig = plt.figure(figsize=(10, 8))
    axs = [
        fig.add_subplot(2, 2, 1, projection='3d'),
        fig.add_subplot(2, 2, 2, projection='3d'),
        fig.add_subplot(2, 2, 3, projection='3d'),
        fig.add_subplot(2, 2, 4, projection='3d'),
    ]
    titles = ["Vicon (GT)", "Accelerometer", "Gyroscope", "Complementary"]
    fps = 15
    writer = FFMpegWriter(fps=fps, metadata=dict(artist=""), bitrate=2400)
    print(f"[make_orientation_video_simple] Writing {out_path} at {fps} FPS, frames={T}")

    with writer.saving(fig, out_path, dpi=200):
        for k in range(T):
            # clear subplots, keep it simple; rotplot will set limits/aspect
            for ax, title in zip(axs, titles):
                ax.cla()
                ax.set_title(f"{title} | t={t[k]:.2f}s" if title == "Vicon (GT)" else title)

            # draw each orientation using your original rotplot (unchanged)
            rotplot(R_vicon[k], currentAxes=axs[0])
            rotplot(R_acc[k],   currentAxes=axs[1])
            rotplot(R_gyro[k],  currentAxes=axs[2])
            rotplot(R_comp[k],  currentAxes=axs[3])

            writer.grab_frame()

    plt.close(fig)
    print("[make_orientation_video_simple] Done.")
