import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from PoseMapper import PoseMapper
from PoseMapper import PoseMappingEnum
import numpy as np
import Sequence
import distance
import tslearn.metrics as ts

# C:/Users/dennis/Anaconda3/Scripts/activate.bat
sns.set()


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


"""
with open('data/squat_1/complete-session.json', 'r') as myfile:
    mocap_seq1 = myfile.read()
with open('data/squat_4/complete-session.json', 'r') as myfile:
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
u = seq2.get_pcs()
v = seq3.get_pcs()
# Cut sequence to same length -> 150 Keypoints in this case
print(f"Hausdorff distance: {distance.hausdorff(u, v)[0]}")
# TODO: Check how to prepare parametersfor dtw function
# TODO: Consider usage of another dtw function module
# dtw_distance, dtw_path = distance.fast_dtw(np.ndarray.flatten(
    # seq1.get_pcs()), np.ndarray.flatten(seq2.get_pcs()))
# print(f"Dynamic Time Warping distance: {dtw_distance}")

plot_pcas([seq1, seq2])
"""

sin_x = np.arange(0, 10, 0.25)
sin_y = np.sin(sin_x)
sin_z = sin_y*np.sin(sin_x)
cos_x = np.arange(0, 10, 0.1)
cos_y = np.cos(cos_x)
cos_z = cos_y*np.cos(cos_x)
cos = np.zeros((len(cos_x), 3))
sin = np.zeros((len(sin_x), 3))

for i in range(len(cos)):
    cos[i] = [cos_x[i], cos_y[i], cos_z[i]]
for i in range(len(sin)):
    sin[i] = [sin_x[i], sin_y[i], sin_z[i]]

# dtw_distance, dtw_path = distance.fast_dtw(sin.any(), cos.any())
dtw_distance = ts.dtw(cos, sin)
dtw_path = ts.dtw_path(cos, sin)
print(dtw_distance)
print(dtw_path)

fig = plt.figure()
ax = plt.axes(projection='3d')

path_x = np.zeros(len(dtw_path[0]))
path_y = np.zeros(len(dtw_path[0]))
for i in range(len(dtw_path[0])):
    path_x[i] = dtw_path[0][i][0]
    path_y[i] = dtw_path[0][i][1]
ax.scatter(sin_x, sin_y, sin_z, c='red')
ax.scatter(cos_x, cos_y, cos_z, c='blue')

for i in range(len(dtw_path[0])):
    ax.plot([cos[dtw_path[0][i][0]][0], sin[dtw_path[0][i][1]][0]], [cos[dtw_path[0][i][0]][1],
                                                                     sin[dtw_path[0][i][1]][1]], [cos[dtw_path[0][i][0]][2], sin[dtw_path[0][i][1]][2]], c="green")
plt.show()
