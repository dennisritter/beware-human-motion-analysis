import math
import numpy as np
from scipy.spatial.transform import Rotation
import sklearn.preprocessing as preprocessing


def get_angle(v1, v2):
    return np.arccos(np.dot(norm(v1), norm(v2)))


def get_rotation(v1, v2):
    """Returns a homogenious 4x4 transformation matrix without translation vector that describes the rotational transformation from v1 to v2"""
    v1 = norm(v1)
    v2 = norm(v2)
    theta = get_angle(v1, v2)
    rotation_axis = get_perpendicular_vector(v1, v2)
    R = rotation_matrix_4x4(rotation_axis, theta)
    return R


def get_perpendicular_vector(v1, v2):
    """Returns a vector that is perpendicular to v1 and v2

    Args:
        v1 (np.ndarray): Vector one, which is perpendicular to the returned vector.
        v2 (np.ndarray): Vector two, which is perpendicular to the returned vector.
    """
    v1 = norm(v1)
    v2 = norm(v2)

    # If theta 180° (dot product = -1)
    # print(np.dot(v1, v2))
    v1_dot_v2 = np.dot(v1, v2)
    if v1_dot_v2 == -1 or v1_dot_v2 == 1:
        # Whenever v1 and v2 are parallel to each other, we can use an arbitrary vector that is NOT parallel to v1 and v2
        # So call this function recursively until a non-parallel vector has been found
        return get_perpendicular_vector(np.random.rand(3), v2)
    else:
        return norm(np.cross(v1, v2))


def get_perpendicular_vector_batch(v_arr1, v_arr2):
    """Returns an array of vectors that are perpendicular to the vectors at corresponding indices. 

    Args:
        v1 (np.ndarray): The first array of vectors that are perpendicular to the respective vector of the returned array.
        v2 (np.ndarray): The second array of vectors that are perpendicular to the respective vector of the returned array.
    """
    v_arr1 = norm_batch(v_arr1)
    v_arr2 = norm_batch(v_arr2)

    # Matrix multiplication to get dot products of all rows of A and columns of B
    # The diagonal values of the resulting product matrix represent the dot product of A[n] and B[n] in the original arrays
    dot_products = dot_batch(v_arr1, v_arr2)
    # Get indices of vector pairs that are parallel theta 0°/180° (dot product = 1 or -1)
    parallel_vector_indices = np.where((dot_products == -1) | (dot_products == 1))[0]
    for idx in parallel_vector_indices:
        # Replace all vectors of v_arr1 that are parallel to the respective vector in v_arr2 by random unparallel vectors
        v_arr1[idx] = random_unparallel_vector(v_arr2[idx])

    return norm_batch(np.cross(v_arr1, v_arr2))


def random_unparallel_vector(v1):
    """Returns an unparallel vector to v1 with same dimension. """
    v2 = np.random.rand(v1.shape[0])
    v1_dot_v2 = np.dot(v1, v2)
    if v1_dot_v2 == -1 or v1_dot_v2 == 1:
        return random_unparallel_vector(v1)
    else:
        return v2


def dot_batch(va1, va2) -> np.ndarray:
    return (va1 @ va2.transpose()).diagonal()


def norm(v) -> np.ndarray:
    """Normalises the given vector v and returns it afterwards.

    Args:
        v (np.ndarray): The vector to normalise.
    """
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return np.zeros(3)
    return v / v_norm


def norm_batch(v_arr):
    """Normalises the given vectors of v_arr and returns them in an array afterwards.

    Args:
        v_arr (np.ndarray): A numpy array of vectors to normalise.
    """

    return preprocessing.normalize(v_arr, norm='l2')


def mat_vec_mul_batch(a, b):
    if a.ndim == 3 and b.ndim == 2:
        return np.einsum('ijk, ik -> ij', a, b)
    elif a.ndim != 3:
        raise ValueError('The first parameter a should be a matrix of matrices (a.ndim == 3)')
    elif b.ndim != 2:
        raise ValueError('The second parameter b should be a matrix of vectors (b.ndim == 2)')


def mat_mat_mul_batch(a, b):
    if a.ndim == 3 and b.ndim == 3:
        return np.einsum('ijk, ikl -> ijl', a, b)
    elif a.ndim != 3:
        raise ValueError('The first parameter a should be a matrix of matrices (a.ndim == 3)')
    elif b.ndim != 3:
        raise ValueError('The second parameter b should be a matrix of matrices (b.ndim == 3)')


def rotation_matrix_4x4(axis, theta) -> np.ndarray:
    # Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta in radians as 4x4 Transformation Matrix

    Args:
        axis(np.array): The vector to rotate about.
        theta(float): The degrees to rotate about the given axis.

    Returns:
        (np.ndarray) 4x4 rotation matrix representing a rotation about the given axis
    """
    axis = np.asarray(axis)
    axis = norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
        [0, 0, 0, 1]
    ])  # yapf: disable


def translation_matrix_4x4(v) -> np.ndarray:
    """Returns a 4x4 Matrix representing a translation.

    Args:
        v (np.ndarray): A vector defining the translation.

    Returns:
        (np.ndarray) 4x4 transformation matrix representing a translation as defined by argument v.
    """
    T = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
    ])  # yapf: disable
    T[:3, 3] = v
    return T


def translation_matrix_4x4_batch(v_arr) -> np.ndarray:
    """Returns an array of 4x4 Matrices, each representing a translation.

    Args:
        v_arr (np.ndarray): An array of vectors defining translations.

    Returns:
        (np.ndarray) Array of 4x4 transformation matrices, each representing a translation as defined by respective vectors in v_arr.
    """
    M = np.empty([len(v_arr), 4, 4])
    I = [[1.0, 0, 0, 0],
         [0, 1.0, 0, 0],
         [0, 0, 1.0, 0],
         [0, 0, 0, 1.0]]  # yapf: disable
    M[:] = I
    M[:, :3, 3] = v_arr[:, :]
    return M


def get_local_coordinate_system_direction_vectors(origin, x_direction_bp_pos, y_direction_bp_pos):
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

    return np.array([norm(vx), norm(vy), norm(vz)])


# TODO: Decide where is the best place for this function. Maybe a new module makes sense so transformations.py holds general functions only
#       and another module/class implements more specific functions like get_pelvis_coordinate_system.
#           Transformation module (keep here)?
#           Sequence class?
#           New module?
def get_pelvis_coordinate_system(pelvis: np.ndarray, torso: np.ndarray, hip_l: np.ndarray, hip_r: np.ndarray):
    """Returns a pelvis coordinate system defined as a tuple containing an origin point and a list of three normalised direction vectors.

    Constructs direction vectors that define the axes directions of the pelvis coordinate system.
    X-Axis-Direction:   Normalised vector whose direction points from hip_l to hip_r. Afterwards, it is translated so that it starts at the pelvis.
    Y-Axis-Direction:   Normalised vector whose direction is determined so that it is perpendicular to the hip_l-hip_r vector and points to the torso.
                        Afterwards, it is translated so that it starts at the pelvis.
    Z-Axis-Direction:   The normalised cross product vector between X-Axis and Y-Axis that starts at the pelvis and results in a right handed Coordinate System.

    Args:
        pelvis (np.ndarray): The X, Y and Z coordinates of the pelvis body part.
        torso (np.ndarray): The X, Y and Z coordinates of the torso body part.
        hip_r (np.ndarray): The X, Y and Z coordinates of the hip_l body part.
        hip_l (np.ndarray): The X, Y and Z coordinates of the hip_r body part.
    """

    # Direction of hip_l -> hip_r is the direction of the X-Axis
    hip_l_hip_r = hip_r - hip_l

    # Orthogonal Projection to determine Y-Axis direction
    a = torso - hip_l
    b = hip_r - hip_l

    scalar = np.dot(a, b) / np.dot(b, b)
    a_on_b = (scalar * b) + hip_l
    v = torso - a_on_b

    origin = pelvis
    vx = norm(hip_l_hip_r)
    vz = norm(v)
    vy = get_perpendicular_vector(vz, vx)

    return [(origin, [vx, vy, vz])]


def get_cs_projection_transformation(from_cs: np.ndarray, target_cs: np.ndarray):
    """Returns a 4x4 transformation to project positions from the from_cs coordinate system to the to_cs coordinate system.

    Args:
        from_cs (np.ndarray): The current coordinate system
            example: [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
        target_cs (np.ndarray): The target coordinate system
    """
    from_cs_origin, from_cs_x, from_cs_y, from_cs_z = from_cs
    target_cs_origin, target_cs_x, target_cs_y, target_cs_z = target_cs

    # Get Translation
    T = translation_matrix_4x4(from_cs_origin - target_cs_origin)
    # Construct rotation matrix for X-Alignment to rotate about x_rot_axis for the angle theta
    x_rot_axis = get_perpendicular_vector(target_cs_x, from_cs_x)
    theta_x = get_angle(target_cs_x, from_cs_x)
    Rx = rotation_matrix_4x4(x_rot_axis, theta_x)

    # Use target x-axis direction vector as rotation axis as it must be perpendicular to the y-axis
    y_rot_axis = target_cs_x
    target_cs_y_rx = (Rx @ np.append(target_cs_y, 1))[:3]
    theta_y = get_angle(target_cs_y_rx, from_cs_y)
    Ry = rotation_matrix_4x4(norm(y_rot_axis), theta_y)

    # Determine complete transformation matrix
    M = Rx @ Ry @ T
    return M


def align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, z_direction_bp_idx: int, positions: np.ndarray):
    """
    Aligns the coordinate system to the given origin point.
    The X-Axis will be in direction of x_direction-origin.
    The Y-Axis will be in direction of y_direction-origin, without crossing the y_direction point but perpendicular to the new X-Axis.
    The Z-Axis will be perpendicular to the XY-Plane.

    Args:
        origin_bp_idx (int): The body part index whose position represents the origin of the coordinate system.
        x_direction_bp_idx (int): The body part index whose position denotes the direction of the x-axis.
        y_direction_bp_idx (int): The body part index whose position denotes the direction of the y-axis.
        positions (np.ndarray): The tracked positions of all body parts for one frame of a motion sequence.
    """

    # Positions of given orientation joints in GCS
    origin = positions[origin_bp_idx]
    x_direction_bp_pos = positions[x_direction_bp_idx]
    z_direction_bp_pos = positions[z_direction_bp_idx]

    # New X-Axis from origin to x_direction
    vx = x_direction_bp_pos - origin
    if vx[0] < 0:
        vx = -vx
    # New Z-Axis is perpendicular to the origin-y_direction vector and vx
    vy = get_perpendicular_vector((z_direction_bp_pos - origin), vx)
    if vy[1] < 0:
        vy = -vy

    # New Y-Axis is perpendicular to new X-Axis and Z-Axis
    vz = get_perpendicular_vector(vx, vy)
    if vz[2] < 0:
        vz = -vz

    # Construct translation Matrix to move given origin to zero-position
    T = translation_matrix_4x4(np.array([0, 0, 0]) - origin)
    # Construct rotation matrix for X-Alignment to rotate about x_rot_axis for the angle theta
    x_rot_axis = get_perpendicular_vector(vx, np.array([1, 0, 0]))
    theta_x = get_angle(vx, np.array([1, 0, 0]))
    Rx = rotation_matrix_4x4(x_rot_axis, theta_x)
    # Use new X-Axis axis for y rotation and Rotate Y-direction vector to get rotation angle for Y-Alignment
    z_rot_axis = vx
    vz_rx = np.matmul(Rx, np.append(vz, 1))[:3]
    theta_z = get_angle(vz_rx, np.array([0, 1, 0]))
    Rz = rotation_matrix_4x4(norm(z_rot_axis), theta_z)
    # Transform all positions
    transformed_positions = []
    M = np.matmul(T, Rx, Rz)
    for pos in positions:
        pos = np.matmul(M, np.append(pos, 1))[:3]
        transformed_positions.append(pos)

    return transformed_positions
