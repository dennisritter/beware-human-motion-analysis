import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tslearn.metrics as ts
import seaborn as sns
from PoseMapper import PoseMapper
from PoseFormatEnum import PoseFormatEnum
import numpy as np
import Sequence
import distance


sns.set()
sns.set_style(style='whitegrid')


def vis_angle_for_frame(seq: Sequence, angle: dict, frame: int):
    # Inverse bodyparts map to use bodypart index
    inv_body_parts = {v: k for k, v in seq.body_parts.items()}
    angle_vertex = angle["angle_vertex"]
    angle_ray_a = angle["rays"][0]
    angle_ray_b = angle["rays"][1]

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.axes(projection='3d')
    # ax.set_aspect('equal')
    # x,y,z for angle vertices
    # angle_x = [seq.positions[frame][angle_vertex][0], seq.positions[frame][angle_ray_a][0], seq.positions[frame][angle_ray_b][0]]
    # ignore x values
    angle_x = [0, 0, 0]
    angle_y = [seq.positions[frame][angle_vertex][1], seq.positions[frame][angle_ray_a][1], seq.positions[frame][angle_ray_b][1]]
    angle_z = [seq.positions[frame][angle_vertex][2], seq.positions[frame][angle_ray_a][2], seq.positions[frame][angle_ray_b][2]]

    ax.scatter(angle_x, angle_y, angle_z, c='r')
    ax.plot([angle_x[0], angle_x[1]], [angle_y[0], angle_y[1]], [angle_z[0], angle_z[1]], color='r')
    ax.plot([angle_x[0], angle_x[2]], [angle_y[0], angle_y[2]], [angle_z[0], angle_z[2]], color='r')
    # ax.plot([seq.positions[frame][angle_vertex][0], seq.positions[frame][angle_ray_a][0]], [seq.positions[frame][angle_vertex][1], seq.positions[frame][angle_ray_a][1]], [seq.positions[frame][angle_vertex][2], seq.positions[frame][angle_ray_a][2]], color='r')
    # ax.plot([seq.positions[frame][angle_vertex][0], seq.positions[frame][angle_ray_b][0]], [seq.positions[frame][angle_vertex][1], seq.positions[frame][angle_ray_b][1]], [seq.positions[frame][angle_vertex][2], seq.positions[frame][angle_ray_b][2]], color='r')
    ax.text(angle_x[0], angle_y[0], angle_z[0], inv_body_parts[angle_vertex], color="r")
    ax.text(angle_x[1], angle_y[1], angle_z[1], inv_body_parts[angle_ray_a], color="r")
    ax.text(angle_x[2], angle_y[2], angle_z[2], inv_body_parts[angle_ray_b], color="r")

    noangle_x = []
    noangle_y = []
    noangle_z = []
    for i in range(len(seq.positions[frame])):
        if (i == angle_vertex or i == angle_ray_a or i == angle_ray_b):
            continue
        noangle_x.append(0)
        # noangle_x.append(seq.positions[frame][i][0])
        noangle_y.append(seq.positions[frame][i][1])
        noangle_z.append(seq.positions[frame][i][2])
        ax.text(seq.positions[frame][i][0], seq.positions[frame][i][1], seq.positions[frame][i][2], inv_body_parts[i], color="blue")

    ax.scatter(noangle_x, noangle_y, noangle_z, color="blue")
    plt.show()

    print(f"Position angle_vertex: {angle_x[0]},{angle_y[0]},{angle_z[0]}")
    print(f"Position ray_a: {angle_x[1]},{angle_y[1]},{angle_z[1]}")
    print(f"Position ray_b: {angle_x[2]},{angle_y[2]},{angle_z[2]}")
