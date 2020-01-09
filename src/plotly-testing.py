import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis import transformations

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
    camera=dict(
        up=dict(x=0, y=1, z=0),
        eye=dict(x=-1.5, y=1.5, z=-1.5)
    )
)

layout = go.Layout(
    scene=scene,
)

def get_lcs_trace(origin, x_direction_pos, y_direction_pos):
    """ Returns a list that contains a plotly trace object the X, Y and Z axes of the local joint coordinate system calculated from and origin, a X-axis-direction and a Y-axis-direction.
    """

    # Get Local Coordinate System vectors
    lcs = transformations.get_local_coordinate_system_direction_vectors(
        origin,
        x_direction_pos,
        y_direction_pos
    )

    # Set Local Coordinate System vectors' length to 100 and move relative to local origin
    lcs[0] = lcs[0] * 100 + origin
    lcs[1] = lcs[1] * 100 + origin
    lcs[2] = lcs[2] * 100 + origin
    trace_x = go.Scatter3d(
        x=[origin[0], lcs[0, 0]],
        y=[origin[1], lcs[0, 1]],
        z=[origin[2], lcs[0, 2]],
        mode="lines",
        marker=dict(color="red")
    )
    trace_y = go.Scatter3d(
        x=[origin[0], lcs[1, 0]],
        y=[origin[1], lcs[1, 1]],
        z=[origin[2], lcs[1, 2]],
        mode="lines",
        marker=dict(color="green")
    )
    trace_z = go.Scatter3d(
        x=[origin[0], lcs[2, 0]],
        y=[origin[1], lcs[2, 1]],
        z=[origin[2], lcs[2, 2]],
        mode="lines",
        marker=dict(color="blue")
    )
    return [trace_x, trace_y, trace_z]


def get_traces():
    trace_joints = go.Scatter3d(
        x=p[frame, :, 0],
        y=p[frame, :, 1],
        z=p[frame, :, 2],
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

    # Get Local Joint Coordinate System Axis traces 
    bps = sequence.body_parts
    ls_lcs_traces = get_lcs_trace(p[frame, bps["LeftShoulder"]], p[frame, bps["RightShoulder"]], p[frame, bps["Torso"]])
    rs_lcs_traces = get_lcs_trace(p[frame, bps["RightShoulder"]], p[frame, bps["LeftShoulder"]], p[frame, bps["Torso"]])
    lh_lcs_traces = get_lcs_trace(p[frame, bps["LeftHip"]], p[frame, bps["RightHip"]], p[frame, bps["Torso"]])
    rh_lcs_traces = get_lcs_trace(p[frame, bps["RightHip"]], p[frame, bps["LeftHip"]], p[frame, bps["Torso"]])
    
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
        trace_rknee_rankle,
        ls_lcs_traces[0],
        ls_lcs_traces[1],
        ls_lcs_traces[2],
        rs_lcs_traces[0],
        rs_lcs_traces[1],
        rs_lcs_traces[2],
        lh_lcs_traces[0],
        lh_lcs_traces[1],
        lh_lcs_traces[2],
        rh_lcs_traces[0],
        rh_lcs_traces[1],
        rh_lcs_traces[2]
    ]

    return traces

fig = go.Figure(
    data=get_traces(),
    layout=layout
)
fig.show()
