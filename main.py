import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PoseMapper import PoseMapper
from PoseMapper import PoseMappingEnum
import numpy as np
import Sequence
import distance

# C:/Users/dennis/Anaconda3/Scripts/activate.bat
sns.set()


def plot_example_mocap_sequence():
    MOCAP_SEQUENCE = ""
    with open('data/example.json', 'r') as myfile:
        MOCAP_SEQUENCE = myfile.read()

    mocap_opmpi_mapper = PoseMapper(PoseMappingEnum.MOCAP)
    # Convert mocap json string Positions to dictionary with openpose MPI postions
    sequence = mocap_opmpi_mapper.map(MOCAP_SEQUENCE)
    # print(sequence.positions)
    # print(sequence.timestamps)
    # print(sequence.body_parts)

    # TODO: Find method to plot one graph visualizing a motion of multiple keypoints
    # Plotting Lines of the motions --> Use PCA?
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # Plot movement of all parts stored in array
    for i in range(len(sequence.body_parts)):
        px, py, pz = [], [], []
        px = sequence.positions[i][:, 0]
        py = sequence.positions[i][:, 1]
        pz = sequence.positions[i][:, 2]
        ax.plot(px, py, pz, label=list(
            sequence.body_parts)[i])
    ax.legend()
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
        xPCA = sequence.get_pcs(num_components=3)
        # Visualize PCs
        ax.scatter(xPCA[:, 0], xPCA[:, 1], xPCA[:, 2], marker='.', c="red")
        # Connect points from PCs to 3D-Line
        ax.plot(xPCA[:, 0], xPCA[:, 1], xPCA[:, 2], label=sequence.name)
        # ax.axis('square')
        ax.legend()

    plt.show()


with open('data/squat_1/complete-session.json', 'r') as myfile:
    mocap_seq1 = myfile.read()
with open('data/squat_2/complete-session.json', 'r') as myfile:
    mocap_seq2 = myfile.read()
with open('data/squat_false/complete-session.json', 'r') as myfile:
    mocap_seq3 = myfile.read()
with open('data/no_squat/complete-session.json', 'r') as myfile:
    mocap_seq4 = myfile.read()
with open('data/example/complete-session.json', 'r') as myfile:
    example = myfile.read()

# Get PoseMapper instance for MOCAP sequences in json
mocap_opmpi_mapper = PoseMapper(PoseMappingEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq1 = mocap_opmpi_mapper.map(mocap_seq1, 'squat_1')
seq2 = mocap_opmpi_mapper.map(mocap_seq2, 'squat_2')
seq3 = mocap_opmpi_mapper.map(mocap_seq3, 'squat_false')
seq4 = mocap_opmpi_mapper.map(mocap_seq4, 'no_squat')
example_seq = mocap_opmpi_mapper.map(example, 'example')

# Calculate Hausdorff distance between two sequences' principal component graphs
u = seq1.get_pcs()
v = seq2.get_pcs()
# Cut sequence to same length -> 150 Keypoints in this case
print(f"Hausdorff distance: {distance.hausdorff(u, v)[0]}")
# TODO: Check how to prepare parametersfor dtw function
# TODO: Consider usage of another dtw function module
dtw_distance, dtw_path = distance.fast_dtw(np.ndarray.flatten(
    seq1.get_pcs()), np.ndarray.flatten(seq2.get_pcs()))
print(f"Dynamic Time Warping distance: {dtw_distance}")

plot_pcas([seq1, seq2])