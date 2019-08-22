from .PoseMapper import PoseMapper
from .PoseFormatEnum import PoseFormatEnum
from .Sequence import Sequence
from . import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tslearn.metrics as ts
import seaborn as sns
import numpy as np


sns.set()
sns.set_style(style='whitegrid')


def plot_ball_joint_angle(joint_aligned_positions: list, ball_joint_idx: int, angle_ref_joint_idx: int):
    zero_x, zero_y, zero_z = joint_aligned_positions[ball_joint_idx]
    ax, ay, az = joint_aligned_positions[angle_ref_joint_idx]
    vx, vy, vz = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    ax3d.set_xlim3d(-1, 1)
    ax3d.set_ylim3d(-1, 1)
    ax3d.set_zlim3d(-1, 1)
    for i, p in enumerate(joint_aligned_positions):
        if i == ball_joint_idx or i == angle_ref_joint_idx:
            ax3d.scatter(p[0], p[1], p[2], c="red")
        else:
            ax3d.scatter(p[0], p[1], p[2], c="blue")
    ax3d.plot([zero_x, ax],
            [zero_y, ay],
            [zero_z, az],
            color="pink", linewidth=1)

    ax3d.plot([zero_x, vx[0]],
            [zero_y, vx[1]],
            [zero_z, vx[2]],
            color="pink", linewidth=1)
    ax3d.plot([zero_x, vy[0]],
            [zero_y, vy[1]],
            [zero_z, vy[2]],
            color="maroon", linewidth=1)
    ax3d.plot([zero_x, vz[0]],
            [zero_y, vz[1]],
            [zero_z, vz[2]],
            color="red", linewidth=1)
    plt.show()
