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
        # TODO: Arbitrary Vector.. Find method to ensure its not parallel to vx
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


def align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, positions: np.ndarray):
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
    positions: np.ndarray
    """

    # Positions of given orientation joints in GCS
    origin = positions[origin_bp_idx]
    x_direction_bp_pos = positions[x_direction_bp_idx]
    y_direction_bp_pos = positions[y_direction_bp_idx]

    # New X-Axis from origin to x_direction
    vx = x_direction_bp_pos - origin
    if vx[0] < 0:
        vx = -vx
    # New Z-Axis is perpendicular to the origin-y_direction vector and vx
    vz = get_perpendicular_vector((y_direction_bp_pos - origin), vx)
    if vz[2] < 0:
        vz = -vz
    # New Y-Axis is perpendicular to new X-Axis and Z-Axis
    vy = get_perpendicular_vector(vx, vz)
    if vy[1] < 0:
        vy = -vy

    # Construct translation Matrix to move given origin to zero-position
    T = translation_matrix_4x4(np.array([0, 0, 0])-origin)
    # Construct rotation matrix for X-Alignment to rotate about x_rot_axis for the angle theta
    x_rot_axis = get_perpendicular_vector(vx, np.array([1, 0, 0]))
    theta_x = get_angle(vx, np.array([1, 0, 0]))
    Rx = rotation_matrix_4x4(x_rot_axis, theta_x)
    # Use new X-Axis axis for y rotation and Rotate Y-direction vector to get rotation angle for Y-Alignment
    y_rot_axis = vx
    vy_rx = np.matmul(Rx, np.append(vy, 1))[:3]
    theta_y = get_angle(vy_rx, np.array([0, 1, 0]))
    Ry = rotation_matrix_4x4(norm(y_rot_axis), theta_y)
    # Transform all positions
    transformed_positions = []
    M = np.matmul(T, Rx, Ry)
    for pos in positions:
        pos = np.matmul(M, np.append(pos, 1))[:3]
        transformed_positions.append(pos)

    return transformed_positions
