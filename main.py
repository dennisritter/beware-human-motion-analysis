import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PoseMapper import PoseMapper
from PoseMapper import PoseMappingEnum

MOCAP_SEQUENCE = ""
with open('example.json', 'r') as myfile:
    MOCAP_SEQUENCE = myfile.read()

mocap_opmpi_mapper = PoseMapper(PoseMappingEnum.MOCAP_TO_OPENPOSE_MPI)
# Convert mocap json string Positions to dictionary with openpose MPI postions
sequence = mocap_opmpi_mapper.map(MOCAP_SEQUENCE)

# TODO: Adjust Mapping output to be more like this array in order to parse it much easier
# Create 3d Array storing keypoints for each part by their Mapped index
# Visualization of the arrays structure:
# [
#   [[part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z]],
#   [[part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z]],
#   ...
# ]
sequence_arr = []
for ele in range(len(PoseMapper.OPENPOSE_MPI_BODY_PARTS) - 1):
    sequence_arr.append([])

for pose in sequence:
    for timestamp in pose:
        for part in pose[timestamp]:
            sequence_arr[PoseMapper.OPENPOSE_MPI_BODY_PARTS[part]].append(
                [pose[timestamp][part]["x"], pose[timestamp][part]["y"], pose[timestamp][part]["z"]])
sequence_arr = np.array(sequence_arr)

# TODO: Find method to plot one graph visualizing a motion of multiple keypoints
# Plotting Lines of the motions
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# Plot movement of all parts stored in array
for i in range(len(sequence_arr)):
    px, py, pz = [], [], []
    px = sequence_arr[i][:, 0]
    py = sequence_arr[i][:, 1]
    pz = sequence_arr[i][:, 2]
    ax.plot(px, py, pz, label=list(
        PoseMapper.OPENPOSE_MPI_BODY_PARTS.keys())[i])
ax.legend()
plt.show()
