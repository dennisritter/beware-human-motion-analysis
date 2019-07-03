import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PoseMapper import PoseMapper
from PoseMapper import PoseMappingEnum

MOCAP_TESTPOSE = """{
    "RightWrist": "-0.1325387, -0.1372004, 0.4613429",
    "RightElbow": "-0.2033342, -0.1619639, 0.6714674",
    "RightShoulder": "-0.1296259, 0.05647476, 0.7525782",
    "Neck": "0.01361471, 0.1407253, 0.786413",
    "Torso": "-0.0003008375, -0.1353417, 0.8338204",
    "Waist": "0.00202914, -0.3101477, 0.8007931",
    "RightAnkle": "-0.03736664, -1.110017, 0.818454",
    "RightKnee": "-0.03736664, -0.7665025, 0.818454",
    "RightHip": "-0.03736664, -0.4013138, 0.818454",
    "LeftAnkle": "0.07614213, -1.086898, 0.8233797",
    "LeftKnee": "0.07614213, -0.7433834, 0.8233797",
    "LeftHip": "0.07614213, -0.3781947, 0.8233797",
    "LeftWrist": "0.2697008, -0.1348271, 0.4850084",
    "LeftElbow": "0.2739598, -0.1159825, 0.7072791",
    "LeftShoulder": "0.1544218, 0.05747554, 0.8311753",
    "Head": "0.02427456, 0.2437453, 0.7632803"
}"""
MOCAP_SEQUENCE = ""

with open('example.json', 'r') as myfile:
    MOCAP_SEQUENCE = myfile.read()

mocap_opmpi_mapper = PoseMapper(PoseMappingEnum.MOCAP_TO_OPENPOSE_MPI)
# Convert mocap json string Positions to dictionary with openpose MPI postions
sequence = mocap_opmpi_mapper.map(MOCAP_SEQUENCE)
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
# print(np.shape(sequence_arr))
print(sequence_arr)
"""
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for key in pose:
    ax.scatter3D(pose[key]["x"], pose[key]["y"], pose[key]["z"])

plt.show()
"""
