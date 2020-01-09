import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
sequence = mocap_poseprocessor.load(filename)


p = sequence.positions
frame = 0

scene = dict(
    xaxis=dict(
        range=[-1000, 1000],
    ),
    yaxis=dict(
        range=[-1000, 1000],
    ),
    zaxis=dict(
        range=[-1000, 1000],
    ),
)
layout = go.Layout(
    scene = scene,
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None])]
    )]
)

trace_joints = go.Scatter3d(
    x=p[frame,:, 0],
    y=p[frame,:, 1],
    z=p[frame,:, 2],
    mode="markers",
    marker=dict(color="royalblue"))
trace_lelbow_lwrist = go.Scatter3d(
    x=[p[frame, 1, 0], p[frame, 0, 0]],
    y=[p[frame, 1, 1], p[frame, 0, 1]],
    z=[p[frame, 1, 2], p[frame, 0, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_lshoulder_lelbow = go.Scatter3d(
    x=[p[frame, 2, 0], p[frame, 1, 0]],
    y=[p[frame, 2, 1], p[frame, 1, 1]],
    z=[p[frame, 2, 2], p[frame, 1, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_neck_lshoulder = go.Scatter3d(
    x=[p[frame, 3, 0], p[frame, 2, 0]],
    y=[p[frame, 3, 1], p[frame, 2, 1]],
    z=[p[frame, 3, 2], p[frame, 2, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_neck_rshoulder = go.Scatter3d(
    x=[p[frame, 3, 0], p[frame, 14, 0]],
    y=[p[frame, 3, 1], p[frame, 14, 1]],
    z=[p[frame, 3, 2], p[frame, 14, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_rshoulder_relbow = go.Scatter3d(
    x=[p[frame, 14, 0], p[frame, 13, 0]],
    y=[p[frame, 14, 1], p[frame, 13, 1]],
    z=[p[frame, 14, 2], p[frame, 13, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_relbow_rwrist = go.Scatter3d(
    x=[p[frame, 13, 0], p[frame, 12, 0]],
    y=[p[frame, 13, 1], p[frame, 12, 1]],
    z=[p[frame, 13, 2], p[frame, 12, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_head_neck = go.Scatter3d(
    x=[p[frame, 15, 0], p[frame, 3, 0]],
    y=[p[frame, 15, 1], p[frame, 3, 1]],
    z=[p[frame, 15, 2], p[frame, 3, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_neck_torso = go.Scatter3d(
    x=[p[frame, 3, 0], p[frame, 4, 0]],
    y=[p[frame, 3, 1], p[frame, 4, 1]],
    z=[p[frame, 3, 2], p[frame, 4, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_torso_waist = go.Scatter3d(
    x=[p[frame, 4, 0], p[frame, 5, 0]],
    y=[p[frame, 4, 1], p[frame, 5, 1]],
    z=[p[frame, 4, 2], p[frame, 5, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_waist_lhip = go.Scatter3d(
    x=[p[frame, 5, 0], p[frame, 8, 0]],
    y=[p[frame, 5, 1], p[frame, 8, 1]],
    z=[p[frame, 5, 2], p[frame, 8, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_waist_rhip = go.Scatter3d(
    x=[p[frame, 5, 0], p[frame, 11, 0]],
    y=[p[frame, 5, 1], p[frame, 11, 1]],
    z=[p[frame, 5, 2], p[frame, 11, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)

trace_rhip_rknee = go.Scatter3d(
    x=[p[frame, 11, 0], p[frame, 10, 0]],
    y=[p[frame, 11, 1], p[frame, 10, 1]],
    z=[p[frame, 11, 2], p[frame, 10, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_rknee_rankle = go.Scatter3d(
    x=[p[frame, 10, 0], p[frame, 9, 0]],
    y=[p[frame, 10, 1], p[frame, 9, 1]],
    z=[p[frame, 10, 2], p[frame, 9, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_lhip_lknee = go.Scatter3d(
    x=[p[frame, 8, 0], p[frame, 7, 0]],
    y=[p[frame, 8, 1], p[frame, 7, 1]],
    z=[p[frame, 8, 2], p[frame, 7, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)
trace_lknee_lankle = go.Scatter3d(
    x=[p[frame, 7, 0], p[frame, 6, 0]],
    y=[p[frame, 7, 1], p[frame, 6, 1]],
    z=[p[frame, 7, 2], p[frame, 6, 2]],
    mode="lines",
    line=dict(color="firebrick", width=5)
)

traces = [
    trace_joints,
    trace_head_neck,
    trace_neck_lshoulder,
    trace_lshoulder_lelbow,
    trace_lelbow_lwrist,
    trace_neck_rshoulder,
    trace_rshoulder_relbow,
    trace_relbow_rwrist,
    trace_neck_torso,
    trace_torso_waist,
    trace_waist_lhip,
    trace_lhip_lknee,
    trace_lknee_lankle,
    trace_waist_rhip,
    trace_rhip_rknee,
    trace_rknee_rankle
]


fig = go.Figure(
    data=traces,
    layout=layout
)
fig.show()
# fig.show()

# sliders_dict = {
#     "active": 0,
#     "yanchor": "top",
#     "xanchor": "left",
#     "currentvalue": {
#         "font": {"size": 20},
#         "prefix": "Year:",
#         "visible": True,
#         "xanchor": "right"
#     },
#     "transition": {"duration": 300, "easing": "cubic-in-out"},
#     "pad": {"b": 10, "t": 50},
#     "len": 0.9,
#     "x": 0.1,
#     "y": 0,
#     "steps": []
# }
# frames = []
# for frame in range(1, len(sequence)):
#     frames.append(go.Frame(data=[go.Scatter3d(x=p[frame,:, 0],
#                                         y=p[frame, :, 1],
#                                         z=p[frame,:, 2], mode="markers")]))

#     slider_step = {"args": [
#         [frame],
#         {"frame": {"duration": 100, "redraw": False},
#         "mode": "immediate",
#         "transition": {"duration": 100}}
#     ],
#         "label": frame,
#         "method": "animate"}
#     sliders_dict["steps"].append(slider_step)

# fig = go.Figure(
#     data=[go.Scatter3d(x=p[0, :, 0], y=p[0, :, 1],
#                        z=p[0,:, 2], mode="markers")],
#     layout=layout,
#     frames=frames
# )


# fig = px.scatter_3d(data_frame=positions, x=positions[:, 0], y=positions[:, 1], z=positions[:, 2])
# fig.show()
