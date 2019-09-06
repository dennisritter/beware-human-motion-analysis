from .Sequence import Sequence
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math


def get_angle(v1, v2):
    return np.arccos(np.dot(norm(v1), norm(v2)))


def get_perpendicular_vector(v1, v2):

    v1 = norm(v1)
    v2 = norm(v2)

    # If theta 180° (dot product = -1)
    if (np.dot(v1, v2) == -1):
        # TODO: Arbitrary Vector.. Find method to ensure its not parallel to vx_new
        return np.cross(np.array([3, 2, 1]), v2)
    else:
        return norm(np.cross(v1, v2))


def norm(v):
    if math.sqrt(np.dot(v, v)) == 0:
        return np.zeros(3)
    else:
        return v / math.sqrt(np.dot(v, v))


def rotation_matrix_4x4(axis, theta):
    # Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta in radians as 4x4 Transformation Matrix
    """
    axis = np.asarray(axis)
    axis = norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def translation_matrix_4x4(v):
    T = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
    ])
    T[:3, 3] = v
    return T


def align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, seq: Sequence, frame: int):
    """
    Aligns the coordinate system to the given origin point.
    The X-Axis will be in direction of x_direction-origin.
    The Y-Axis will be in direction of y_direction-origin, without crossing the y_direction point but perpendicular to the new X-Axis.
    The Z-Axis will be perpendicular to the XY-Plane.
    Parameters
    ----------
    origin_bp_idx: int
    x_direction_bp_idx: int
    y_direction_bp_idx: int
    seq: Sequence
    frame: int
    """

    # We want to move new_origin position to 0,0,0
    zero_position = np.array([0, 0, 0])
    # We want the alignment of x,y,z direction
    vx = norm(np.array([1, 0, 0]))
    vy = norm(np.array([0, 1, 0]))
    vz = norm(np.array([0, 0, 1]))

    # TODO: Ensure correct directions of perpendicular vectors: Z-Axis to front or back? Y-Axis up or down?
    # Positions of given orientation joints in GCS
    origin = seq.positions[frame][origin_bp_idx]
    x_direction_bp_gpos = seq.positions[frame][x_direction_bp_idx]
    y_direction_bp_gpos = seq.positions[frame][y_direction_bp_idx]
    # Y-direction should point upwards. So turn y_direction if it lies under origin
    # New X-Axis from origin to x_direction
    vx_new = x_direction_bp_gpos - origin
    if vx_new[0] < 0:
        vx_new = -vx_new

    # New Z-Axis is perpendicular to the origin to x_direction and origin to y_direction vectors
    # if origin[1] > y_direction_bp_gpos[1]:
    #     vz_new = get_perpendicular_vector((y_direction_bp_gpos - origin)*-1, vx_new)
    # else:
    vz_new = get_perpendicular_vector((y_direction_bp_gpos - origin), vx_new)
    # Ensure Z points away from camera position
    if vz_new[2] < 0:
        vz_new = -vz_new

    # New Y-Axis is perpendicular to new X-Axis and Z-Axis
    vy_new = get_perpendicular_vector(vx_new, vz_new)
    # Ensure that vy points up
    if vy_new[1] < 0:
        vy_new = -vy_new

        # Construct translation Matrix to move given origin to zero-position
    T = translation_matrix_4x4(zero_position-origin)
    # Construct rotation matrix for X-Alignment to rotate about x_rot_axis for the angle theta
    x_rot_axis = get_perpendicular_vector(vx_new, vx)
    theta_x = get_angle(vx_new, vx)
    Rx = rotation_matrix_4x4(x_rot_axis, theta_x)

    # Use new X-Axis axis for y rotation and Rotate Y-direction vector to get rotation angle for Y-Alignment
    y_rot_axis = vx_new
    vy_new_rx = np.matmul(Rx, np.append(vy_new, 1))[:3]
    theta_y = get_angle(vy_new_rx, vy)

    Ry_plus = rotation_matrix_4x4(norm(y_rot_axis), theta_y)
    Ry_minus = rotation_matrix_4x4(norm(y_rot_axis), -theta_y)
    # Check Y Rotation direction
    # NOTE: Did not find a way to determine Y-Rotation direction (theta_y || -theta_y) consistently
    #       without checking y_direction_bp_idx Z-Position after the transformation. (should be ~0)
    trans_y_bp_pos = y_direction_bp_gpos
    trans_y_bp_pos = np.matmul(T, np.append(trans_y_bp_pos, 1))[:3]
    trans_y_bp_pos_plus = np.matmul(Ry_plus, np.append(trans_y_bp_pos, 1))[:3]
    trans_y_bp_pos_minus = np.matmul(Ry_minus, np.append(trans_y_bp_pos, 1))[:3]
    trans_y_bp_pos_plus = np.matmul(Rx, np.append(trans_y_bp_pos_plus, 1))[:3]
    trans_y_bp_pos_minus = np.matmul(Rx, np.append(trans_y_bp_pos_minus, 1))[:3]
    # The smaller absolute Z-coord deviation to 0 after all transformations must be the correct rotation.
    if(abs(trans_y_bp_pos_plus[2]) <= abs(trans_y_bp_pos_minus[2])):
        Ry = Ry_plus
    else:
        Ry = Ry_minus

    # Actually transform all keypoints of the given frame
    transformed_positions = []
    for pos in seq.positions[frame]:
        # TODO: Construct one Transformation Matrix M from T-Ry-Rx
        pos = np.matmul(T, np.append(pos, 1))[:3]
        pos = np.matmul(Ry, np.append(pos, 1))[:3]
        pos = np.matmul(Rx, np.append(pos, 1))[:3]
        transformed_positions.append(pos)

    ################### PLOTTING #####################

    # fig = plt.figure(figsize=plt.figaspect(1)*2)
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.set_xlim3d(-0.5, 0.5)
    # ax.set_ylim3d(-0.5, 0.5)
    # ax.set_zlim3d(-0.5, 0.5)
    # for i, p in enumerate(transformed_positions):
    #     ax.scatter(p[0], p[1], p[2], c="blue")
    # # ax.plot([zero_position[0], -0.1], [zero_position[1], 0.05], [zero_position[2], -0.1], color="pink", linewidth=1)
    # # for j in range(len(seq.positions[frame])):
    # #     ax.scatter(seq.positions[frame][j][0], seq.positions[frame][j][1], seq.positions[frame][j][2], c="red", alpha=0.5)
    # #     ax.text(seq.positions[frame][j][0], seq.positions[frame][j][1], seq.positions[frame][j][2], j)
    # # ax.annotate(f"{j}", (seq.positions[frame][j][0], seq.positions[frame][j][1]))
    # ax.plot([zero_position[0], vx[0]/2], [zero_position[1], vx[1]], [zero_position[2], vx[2]], color="pink", linewidth=1)
    # ax.plot([zero_position[0], vy[0]], [zero_position[1], vy[1]/2], [zero_position[2], vy[2]], color="maroon", linewidth=1)
    # ax.plot([zero_position[0], vz[0]], [zero_position[1], vz[1]], [zero_position[2], vz[2]/2], color="red", linewidth=1)
    # # ax.plot([zero_position[0], vy_new[0]], [zero_position[1], vy_new[1]], [zero_position[2], vy_new[2]], color="green", linewidth=1)
    # # ax.plot([zero_position[0], vx[0]], [zero_position[1], vx[1]], [zero_position[2], vx[2]], color="pink", linewidth=1)
    # # ax.plot([zero_position[0], vy[0]], [zero_position[1], vy[1]], [zero_position[2], vy[2]], color="maroon", linewidth=1)
    # # ax.plot([zero_position[0], vz[0]], [zero_position[1], vz[1]], [zero_position[2], vz[2]], color="red", linewidth=1)
    # plt.show()

    return transformed_positions
