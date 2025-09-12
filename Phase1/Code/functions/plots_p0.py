import matplotlib.pyplot as plt
import scienceplots

def plot_samples(imu_ts):
    with plt.style.context(["science", "no-latex"]): 
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(imu_ts[0, :], label=f"ts")
        ax.set_xlabel("Samples [k]")
        ax.set_ylabel("Time [s]")
        ax.autoscale(tight=True)
        fig.savefig("sample_imu.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_acc(time_s, imu_data_filtered):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["x", "y", "z"]
        for i in range(imu_data_filtered.shape[0]):
            ax.plot(time_s, imu_data_filtered[i, :], label=f"a{labels[i]}")
        ax.legend(loc = "upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Acceleration [m/s^2]")
        ax.autoscale(tight=True)
        fig.savefig("acc.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_gyro(time_s, imu_data_filtered):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["x", "y", "z"]
        for i in range(imu_data_filtered.shape[0]):
            ax.plot(time_s, imu_data_filtered[i, :], label=f"w{labels[i]}")
        ax.legend(loc = "upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angular Velocity [rad/s]")
        ax.autoscale(tight=True)
        fig.savefig("gyro.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_angles(time_s, imu_data_filtered, name):

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["roll", "pitch", "yaw"]
        colors = ["red", "green", "blue"]  # x=roll → red, y=pitch → green, z=yaw → blue
        for i in range(imu_data_filtered.shape[0]):
            ax.plot(time_s, imu_data_filtered[i, :],
                    label=f"{labels[i]}",
                    color=colors[i])
        ax.legend(loc="upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [rad]")
        ax.autoscale(tight=True)

        filename = f"angles_{name}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return None

def plot_quaternions(time_s, quat_data, name):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["w", "x", "y", "z"]
        colors = ["black", "red", "green", "blue"]  # distinct colors

        for i in range(quat_data.shape[0]):
            ax.plot(time_s, quat_data[i, :],
                    label=f"{labels[i]}",
                    color=colors[i])

        ax.legend(loc="upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Quaternion value")
        ax.autoscale(tight=True)

        filename = f"quaternions_{name}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")


def plot_all_methods(time_acc, rpy_acc,
                     time_rot, rpy_rot,
                     time_gyro, rpy_gyro,
                     time_complement, rpy_complement,
                     name="rpy_axis"):
    import matplotlib.pyplot as plt

    with plt.style.context(["science", "no-latex"]):
        labels = ["roll", "pitch", "yaw"]

        method_colors = {
            "vicon":        "black",
            "acc":          "blue",
            "gyro":         "red",
            "complementary":"green",
        }
        linestyles = {
            "vicon":        "solid",
            "acc":          "dashed",
            "gyro":         "dotted",
            "complementary":"dashdot",
        }

        methods = [
            (time_acc,        rpy_acc,        "acc"),
            (time_rot,        rpy_rot,        "vicon"),
            (time_gyro,       rpy_gyro,       "gyro"),
            (time_complement, rpy_complement, "complementary"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=False, sharey=False)

        handles_for_legend = None
        for i, label in enumerate(labels):
            ax = axes[i]
            for time_s, data, method in methods:
                ax.plot(
                    time_s, data[i, :],
                    label=method,
                    color=method_colors[method],
                    linestyle=linestyles[method],
                    linewidth=1.6 if method in ("vicon", "complementary") else 1.2,
                    zorder=3 if method in ("vicon", "complementary") else 2,
                )
            ax.set_title(label.capitalize())
            ax.set_xlabel("Time [s]")

            # Only add ylabel for the first subplot (Roll)
            if i == 0:
                ax.set_ylabel("Angle [rad]")

            ax.autoscale(tight=True)

            if i == len(labels) - 1:
                handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

        # Single shared legend above subplots
        if handles_for_legend:
            fig.legend(handles_for_legend, labels_for_legend,
                       loc="upper center", ncol=4, frameon=False)

        fig.tight_layout(rect=(0, 0, 1, 0.90))  # leave room for legend on top

        filename = f"{name}_rpy.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

def plot_all_methods_new(time_acc, rpy_acc,
                         time_rot=None, rpy_rot=None,
                         time_gyro=None, rpy_gyro=None,
                         time_complement=None, rpy_complement=None,
                         time_madgwick=None, rpy_madgwick=None,
                         name="rpy_axis"):
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.style.context(["science", "no-latex"]):
        labels = ["roll", "pitch", "yaw"]

        method_colors = {
            "vicon":         "black",
            "acc":           "blue",
            "gyro":          "red",
            "complementary": "green",
            "madgwick":      "orange",
        }
        linestyles = {
            "vicon":         "solid",
            "acc":           "dashed",
            "gyro":          "dotted",
            "complementary": "dashdot",
            "madgwick":      (0, (3, 1, 1, 1)),  # dash-dot-dot
        }

        # Always include accelerometer
        methods = [(time_acc, rpy_acc, "acc")]

        # Conditionally add the others
        if time_rot is not None and rpy_rot is not None and len(time_rot) > 0 and rpy_rot.size > 0:
            methods.append((time_rot, rpy_rot, "vicon"))
        if time_gyro is not None and rpy_gyro is not None and len(time_gyro) > 0 and rpy_gyro.size > 0:
            methods.append((time_gyro, rpy_gyro, "gyro"))
        if time_complement is not None and rpy_complement is not None and len(time_complement) > 0 and rpy_complement.size > 0:
            methods.append((time_complement, rpy_complement, "complementary"))
        if time_madgwick is not None and rpy_madgwick is not None and len(time_madgwick) > 0 and rpy_madgwick.size > 0:
            methods.append((time_madgwick, rpy_madgwick, "madgwick"))

        fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=False, sharey=False)

        handles_for_legend = None
        labels_for_legend  = None

        for i, label in enumerate(labels):
            ax = axes[i]
            for time_s, data, method in methods:
                ax.plot(
                    time_s, data[i, :],
                    label=method,
                    color=method_colors[method],
                    linestyle=linestyles[method],
                    linewidth=1.6 if method in ("vicon", "complementary", "madgwick") else 1.2,
                    zorder=3 if method in ("vicon", "complementary", "madgwick") else 2,
                )
            ax.set_title(label.capitalize())
            ax.set_xlabel("Time [s]")
            if i == 0:
                ax.set_ylabel("Angle [rad]")
            ax.autoscale(tight=True)

            if i == len(labels) - 1:
                handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

        # Single shared legend above subplots
        if handles_for_legend:
            fig.legend(handles_for_legend, labels_for_legend,
                       loc="upper center", ncol=len(methods), frameon=False)

        fig.tight_layout(rect=(0, 0, 1, 0.90))  # leave room for legend on top

        filename = f"{name}_rpy.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return filename
