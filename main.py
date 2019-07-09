import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PoseMapper import PoseMapper
from PoseMapper import PoseMappingEnum

MOCAP_SEQUENCE = ""
with open('data/example.json', 'r') as myfile:
    MOCAP_SEQUENCE = myfile.read()

mocap_opmpi_mapper = PoseMapper(PoseMappingEnum.MOCAP)
# Convert mocap json string Positions to dictionary with openpose MPI postions
sequence = mocap_opmpi_mapper.map(MOCAP_SEQUENCE)
print(sequence.positions)
print(sequence.timestamps)
print(sequence.body_parts)

# TODO: Find method to plot one graph visualizing a motion of multiple keypoints
# Plotting Lines of the motions
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
