import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

mocap_opmpi_mapper = PoseMapper(PoseMappingEnum.MOCAP_TO_OPENPOSE_MPI)
# Convert mocap json string Positions to dictionary with openpose MPI postions
pose = mocap_opmpi_mapper.map(MOCAP_TESTPOSE)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
for key in pose:
    ax.scatter3D(pose[key]["x"], pose[key]["y"], pose[key]["z"])

plt.show()
