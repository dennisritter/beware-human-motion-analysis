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


def vis_angle_for_frame_x(seq: Sequence, angle: dict, frame: int):
    # Inverse bodyparts map to use bodypart index
    inv_body_parts = {v: k for k, v in seq.body_parts.items()}
    angle_vertex = angle["angle_vertex"]
    angle_ray_a = angle["rays"][0]
    angle_ray_b = angle["rays"][1]

    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax = plt.axes()
    ax.set_aspect('equal')
    # x,y,z for angle vertices
    # angle_x = [seq.positions[frame][angle_vertex][0], seq.positions[frame][angle_ray_a][0], seq.positions[frame][angle_ray_b][0]]
    # ignore x values
    angle_x = [0, 0, 0]
    angle_y = [seq.positions[frame][angle_vertex][1], seq.positions[frame][angle_ray_a][1], seq.positions[frame][angle_ray_b][1]]
    angle_z = [seq.positions[frame][angle_vertex][2], seq.positions[frame][angle_ray_a][2], seq.positions[frame][angle_ray_b][2]]

    # 3d
    # ax.scatter(angle_x, angle_y, angle_z, c='r')
    # ax.plot([angle_x[0], angle_x[1]], [angle_y[0], angle_y[1]], [angle_z[0], angle_z[1]], color='r')
    # ax.plot([angle_x[0], angle_x[2]], [angle_y[0], angle_y[2]], [angle_z[0], angle_z[2]], color='r')

    # ax.text(angle_x[0], angle_y[0], angle_z[0], inv_body_parts[angle_vertex], color="r")
    # ax.text(angle_x[1], angle_y[1], angle_z[1], inv_body_parts[angle_ray_a], color="r")
    # ax.text(angle_x[2], angle_y[2], angle_z[2], inv_body_parts[angle_ray_b], color="r")
    # 2d
    ax.scatter(angle_z, angle_y, c='r')
    ax.plot([angle_z[0], angle_z[1]], [angle_y[0], angle_y[1]], color='r')
    ax.plot([angle_z[0], angle_z[2]], [angle_y[0], angle_y[2]], color='r')

    ax.text(angle_z[0], angle_y[0], inv_body_parts[angle_vertex], color="r")
    ax.text(angle_z[1], angle_y[1], inv_body_parts[angle_ray_a], color="r")
    ax.text(angle_z[2], angle_y[2], inv_body_parts[angle_ray_b], color="r")

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
        # ax.text(seq.positions[frame][i][0], seq.positions[frame][i][1], seq.positions[frame][i][2], inv_body_parts[i], color="blue")
        ax.text(seq.positions[frame][i][2], seq.positions[frame][i][1], inv_body_parts[i], color="blue")

    # 2d
    ax.scatter(noangle_z, noangle_y, color="blue")
    # 3d
    # ax.scatter(noangle_x, noangle_y, noangle_z, color="blue")
    plt.show()

    print(f"Position angle_vertex: {angle_x[0]},{angle_y[0]},{angle_z[0]}")
    print(f"Position ray_a: {angle_x[1]},{angle_y[1]},{angle_z[1]}")
    print(f"Position ray_b: {angle_x[2]},{angle_y[2]},{angle_z[2]}")


def vis_angle_for_frame(seq: Sequence, angle: dict, frame: int):
    # Inverse bodyparts map to use bodypart index
    inv_body_parts = {v: k for k, v in seq.body_parts.items()}
    angle_vertex = angle["angle_vertex"]
    angle_ray_a = angle["rays"][0]
    angle_ray_b = angle["rays"][1]

    angle_x = [seq.positions[frame][angle_vertex][0], seq.positions[frame][angle_ray_a][0], seq.positions[frame][angle_ray_b][0]]
    angle_y = [seq.positions[frame][angle_vertex][1], seq.positions[frame][angle_ray_a][1], seq.positions[frame][angle_ray_b][1]]
    angle_z = [seq.positions[frame][angle_vertex][2], seq.positions[frame][angle_ray_a][2], seq.positions[frame][angle_ray_b][2]]

    noangle_x = []
    noangle_y = []
    noangle_z = []
    for i in range(len(seq.positions[frame])):
        if (i == angle_vertex or i == angle_ray_a or i == angle_ray_b):
            continue
        noangle_x.append(seq.positions[frame][i][0])
        noangle_y.append(seq.positions[frame][i][1])
        noangle_z.append(seq.positions[frame][i][2])

    # Plotting
    fig = plt.figure(figsize=plt.figaspect(1)*2)

    # 3d
    ax = fig.add_subplot(2, 2, 1)
    ax.set_aspect('equal')
    ax.scatter(angle_z, angle_y, c='r')
    ax.plot([angle_z[0], angle_z[1]], [angle_y[0], angle_y[1]], color='r')
    ax.plot([angle_z[0], angle_z[2]], [angle_y[0], angle_y[2]], color='r')
    ax.text(angle_z[0], angle_y[0], inv_body_parts[angle_vertex], color="r")
    ax.text(angle_z[1], angle_y[1], inv_body_parts[angle_ray_a], color="r")
    ax.text(angle_z[2], angle_y[2], inv_body_parts[angle_ray_b], color="r")
    ax.scatter(noangle_z, noangle_y, color="blue")

    # x = 0
    ax = fig.add_subplot(2, 2, 2)
    ax.set_aspect('equal')
    plt.xlabel('z_pos')
    plt.ylabel('y_pos')
    ax.scatter(angle_z, angle_y, c='r')
    ax.plot([angle_z[0], angle_z[1]], [angle_y[0], angle_y[1]], color='r')
    ax.plot([angle_z[0], angle_z[2]], [angle_y[0], angle_y[2]], color='r')
    ax.text(angle_z[0], angle_y[0], inv_body_parts[angle_vertex], color="r")
    ax.text(angle_z[1], angle_y[1], inv_body_parts[angle_ray_a], color="r")
    ax.text(angle_z[2], angle_y[2], inv_body_parts[angle_ray_b], color="r")
    ax.scatter(noangle_z, noangle_y, color="blue")
    for i in range(len(seq.positions[frame])):
        if (i == angle_vertex or i == angle_ray_a or i == angle_ray_b):
            continue
        ax.text(seq.positions[frame][i][2], seq.positions[frame][i][1], inv_body_parts[i], color="blue", size="small", alpha=0.3)
    # y = 0
    ax = fig.add_subplot(2, 2, 3)
    ax.set_aspect('equal')
    plt.xlabel('x_pos')
    plt.ylabel('z_pos')
    ax.scatter(angle_x, angle_z, c='r')
    ax.plot([angle_x[0], angle_x[1]], [angle_z[0], angle_z[1]], color='r')
    ax.plot([angle_x[0], angle_x[2]], [angle_z[0], angle_z[2]], color='r')
    ax.text(angle_x[0], angle_z[0], inv_body_parts[angle_vertex], color="r")
    ax.text(angle_x[1], angle_z[1], inv_body_parts[angle_ray_a], color="r")
    ax.text(angle_x[2], angle_z[2], inv_body_parts[angle_ray_b], color="r")
    ax.scatter(noangle_x, noangle_z, color="blue")
    for i in range(len(seq.positions[frame])):
        if (i == angle_vertex or i == angle_ray_a or i == angle_ray_b):
            continue
        ax.text(seq.positions[frame][i][0], seq.positions[frame][i][2], inv_body_parts[i], color="blue", size="small", alpha=0.3)
    # z = 0
    ax = fig.add_subplot(2, 2, 4)
    ax.set_aspect('equal')
    plt.xlabel('x_pos')
    plt.ylabel('y_pos')
    ax.scatter(angle_x, angle_y, c='r')
    ax.plot([angle_x[0], angle_x[1]], [angle_y[0], angle_y[1]], color='r')
    ax.plot([angle_x[0], angle_x[2]], [angle_y[0], angle_y[2]], color='r')
    ax.text(angle_x[0], angle_y[0], inv_body_parts[angle_vertex], color="r")
    ax.text(angle_x[1], angle_y[1], inv_body_parts[angle_ray_a], color="r")
    ax.text(angle_x[2], angle_y[2], inv_body_parts[angle_ray_b], color="r")
    ax.scatter(noangle_x, noangle_y, color="blue")
    for i in range(len(seq.positions[frame])):
        if (i == angle_vertex or i == angle_ray_a or i == angle_ray_b):
            continue
        ax.text(seq.positions[frame][i][0], seq.positions[frame][i][1], inv_body_parts[i], color="blue", size="small", alpha=0.3)

    plt.show()

    print(f"Position angle_vertex: {angle_x[0]},{angle_y[0]},{angle_z[0]}")
    print(f"Position ray_a: {angle_x[1]},{angle_y[1]},{angle_z[1]}")
    print(f"Position ray_b: {angle_x[2]},{angle_y[2]},{angle_z[2]}")
