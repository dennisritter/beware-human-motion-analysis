import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.sequence import Sequence
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
seq = mocap_poseprocessor.load('data/sequences/squat-dennis-multi-1/complete-session.json', 'squat-dennis-multi-1')
frame = np.array([i for i in range(len(seq))]).flatten()
# px = seq.positions[:, 1, 0].flatten()
# py = seq.positions[:, 1, 1].flatten()
# pz = seq.positions[:, 1, 2].flatten()
px = np.array([i for i in range(len(seq))]).flatten()
py = np.array([i for i in range(len(seq))]).flatten()
pz = np.array([i for i in range(len(seq))]).flatten()

df = pd.DataFrame({"frame": frame, "x": px, "y": py, "z": pz})


def update_graph(num):
    data = df[df['frame'] == num]
    print("###########################################")
    print(data.x)
    print(data.y)
    print(data.z)
    print("###########################################")
    graph._offsets3d = (data.x, data.y, data.z)
    title.set_text('3D Test, frame={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

data = df[df['frame'] == 0]
graph = ax.scatter(data.x, data.y, data.z)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(seq),
                                         interval=30, blit=False)

plt.show()
