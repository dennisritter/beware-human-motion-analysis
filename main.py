import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PoseMapper import PoseMapper
from PoseMapper import PoseMappingEnum
import PCA

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


def plot_PCA():
    MOCAP_SEQUENCE = ""
    with open('data/example.json', 'r') as myfile:
        MOCAP_SEQUENCE = myfile.read()

    mocap_opmpi_mapper = PoseMapper(PoseMappingEnum.MOCAP)
    # Convert mocap json string Positions to dictionary with openpose MPI postions
    sequence = mocap_opmpi_mapper.map(MOCAP_SEQUENCE)

    xPCA = PCA.calc_pc(sequence)
    # Plotting Sequence
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Plot original sequence (joint positions for all bodyparts and all keypoints)
    # ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], marker=",", c="blue")
    # Visualize PCs
    ax.scatter(xPCA[:, 0], xPCA[:, 1], xPCA[:, 2], marker='.', c="red")
    # Connect points from PCs to 3D-Line
    ax.plot(xPCA[:, 0], xPCA[:, 1], xPCA[:, 2], c="blue")
    # Plot PCs inverse (reconstruct original sequence from PCs)
    # ax.scatter(xPCA_inverse[:, 0], xPCA_inverse[:, 1], xPCA_inverse[:, 2])
    ax.axis('square')
    plt.show()


plot_PCA()
