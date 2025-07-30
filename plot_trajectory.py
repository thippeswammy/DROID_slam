import numpy as np
import matplotlib.pyplot as plt
import csv

def load_trajectory(csv_path):
    trajectory = []
    with open(csv_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            trajectory.append((x, y, z))
    return np.array(trajectory)

def plot_trajectory(trajectory, title="Camera Trajectory"):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o', label="Trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()

    # Optional 2D top-down view
    plt.figure(figsize=(6, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='blue')
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Top-Down X-Y View")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    trajectory = load_trajectory("trajectory_output.csv")
    plot_trajectory(trajectory)
