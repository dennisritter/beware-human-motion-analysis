import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PoseMapper import PoseMapper
from PoseFormatEnum import PoseFormatEnum
import numpy as np
import Sequence
import distance
import tslearn.metrics as ts

sns.set()
sns.set_style(style='whitegrid')


def plot_dtw_sin_cos():
    # Create sin coords
    sin_x = np.arange(0, 10, 0.1)
    sin_y = np.sin(sin_x)
    sin_z = sin_y*np.sin(sin_x)
    # create cos coords
    cos_x = np.arange(0, 10, 0.1)
    cos_y = np.cos(cos_x)
    cos_z = cos_y*np.cos(cos_x)
    # Merge coords into a 2d Array
    cos = np.zeros((len(cos_x), 3))
    for i in range(len(cos)):
        cos[i] = [cos_x[i], cos_y[i], cos_z[i]]
    sin = np.zeros((len(sin_x), 3))
    for i in range(len(sin)):
        sin[i] = [sin_x[i], sin_y[i], sin_z[i]]

    # Calc dtw distance and path
    dtw_path, dtw_distance = ts.dtw_path(cos, sin)
    print(f"dtw_distance: {dtw_distance}")

    ### Plotting ###
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    fig.suptitle(f"DTW Distance: {dtw_distance}")
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    path_x = np.zeros(len(dtw_path))
    path_y = np.zeros(len(dtw_path))
    for i in range(len(dtw_path[0])):
        path_x[i] = dtw_path[i][0]
        path_y[i] = dtw_path[i][1]

    ax.scatter(sin_x, sin_y, sin_z, c='red')
    ax.scatter(cos_x, cos_y, cos_z, c='blue')
    for i in range(len(dtw_path)):
        ax.plot([cos[dtw_path[i][0]][0], sin[dtw_path[i][1]][0]], [cos[dtw_path[i][0]][1],
                                                                   sin[dtw_path[i][1]][1]], [cos[dtw_path[i][0]][2], sin[dtw_path[i][1]][2]], c="green")
    # Create Visualization of dtw_path matrix
    matrix_path = np.zeros((len(cos), len(sin)), dtype=np.int)
    for i, j in dtw_path:
        matrix_path[i, j] = 1

    plt.subplot(2, 2, 2)
    plt.imshow(matrix_path, cmap="gray_r")
    plt.tight_layout()
    plt.show()


def plot_pcas_dtw(seq1: Sequence, seq2: Sequence, dtw_path: list, dtw_distance: float, path_matrix=True):
    # Plotting Sequences
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    fig.suptitle(f"DTW Distance: {dtw_distance}")
    if path_matrix:
        ax = fig.add_subplot(2, 2, 1, projection='3d')
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    # Plot Sequence Principal components as 3d graph
    seq1_pc = seq1.get_pcs(num_components=3)
    seq2_pc = seq2.get_pcs(num_components=3)
    # Visualize PCs
    ax.scatter(seq1_pc[:, 0], seq1_pc[:, 1],
               seq1_pc[:, 2], marker='.', c="red")
    ax.scatter(seq2_pc[:, 0], seq2_pc[:, 1],
               seq2_pc[:, 2], marker='.', c="blue")
    # Connect points from PCs to 3D-Line
    ax.plot(seq1_pc[:, 0], seq1_pc[:, 1], seq1_pc[:, 2], label=seq1.name)
    ax.plot(seq2_pc[:, 0], seq2_pc[:, 1], seq2_pc[:, 2], label=seq2.name)

    for i in range(len(dtw_path)):
        ax.plot([seq1_pc[dtw_path[i][0]][0], seq2_pc[dtw_path[i][1]][0]], [seq1_pc[dtw_path[i][0]][1],
                                                                           seq2_pc[dtw_path[i][1]][1]], [seq1_pc[dtw_path[i][0]][2], seq2_pc[dtw_path[i][1]][2]], c="green")
    ax.legend()

    if path_matrix:
        # Create Visualization of dtw_path matrix
        matrix_path = np.zeros((len(seq1_pc), len(seq2_pc)), dtype=np.int)
        for i, j in dtw_path:
            matrix_path[i, j] = 1
        plt.subplot(2, 2, 2)
        plt.imshow(matrix_path, cmap="gray_r")

    plt.tight_layout()
    plt.show()


def plot_pcas(seqs: list):
    """
    Plots all sequences of the given list.
    Parameters
    ----------
    seqs : list<Sequence>
        A list of sequences to plot
    """
    # Plotting Sequences
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = plt.axes(projection='3d')
    for sequence in seqs:
        seq_pc = sequence.get_pcs(num_components=3)
        # Visualize PCs
        ax.scatter(seq_pc[:, 0], seq_pc[:, 1],
                   seq_pc[:, 2], marker='.', c="red")
        # Connect points from PCs to 3D-Line
        ax.plot(seq_pc[:, 0], seq_pc[:, 1], seq_pc[:, 2], label=sequence.name)
        # ax.axis('square')
        ax.legend()

    plt.show()


# Load Sequence json files
with open('data/sequences/squat_1/complete-session.json', 'r') as myfile:
    mocap_seq1 = myfile.read()
with open('data/sequences/squat_3/complete-session.json', 'r') as myfile:
    mocap_seq2 = myfile.read()
# Get PoseMapper instance for MOCAP sequences in json
mocap_opmpi_mapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq1 = mocap_opmpi_mapper.map(mocap_seq1, 'squat_1')
seq2 = mocap_opmpi_mapper.map(mocap_seq2, 'squat_2')

# Calculate Hausdorff distance between two sequences' principal component graphs
seq1_pc = seq1.get_pcs()
seq2_pc = seq2.get_pcs()
print(f"Hausdorff distance: {distance.hausdorff(seq1_pc, seq2_pc)[0]}")
# Calculate Dynamic Time Warping path and distance
dtw_path, dtw_distance = ts.dtw_path(seq1_pc, seq2_pc)
print(f"DTW Distance: {dtw_distance}")

plot_pcas_dtw(seq1, seq2, dtw_path, dtw_distance, path_matrix=False)
# plot_dtw_sin_cos()
# plot_pcas([seq1, seq2])
