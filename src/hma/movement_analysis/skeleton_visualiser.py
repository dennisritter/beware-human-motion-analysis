import plotly.graph_objects as go
from hma.movement_analysis import transformations


class SkeletonVisualiser:
    """Visualises a human pose skeleton as an animated 3D Scatter Plot.

    Attributes:
        sequence (Sequence): The motion sequence to visualise the skeleton from.
    """
    def __init__(
            self,
            sequence: 'Sequence',
    ):

        self.sequence = sequence[:]

    def show(self):
        """Visualises the human pose skeleton as an animated 3D Scatter
        Plot."""
        traces = self._get_traces(0)
        layout = self._get_layout()
        frames = self._get_frames()

        fig = go.Figure(data=traces, layout=layout, frames=frames)
        fig.show()

    def _get_layout(self):
        """Returns a Plotly layout."""
        updatemenus = []
        sliders = []
        if len(self.sequence) > 1:
            updatemenus = self._make_buttons()
            sliders = self._make_sliders()

        scene = dict(
            xaxis=dict(range=[-1500, 1500], ),
            yaxis=dict(range=[-1500, 1500], ),
            zaxis=dict(range=[-1500, 1500], ),
            camera=dict(up=dict(x=0, y=1.25, z=0), eye=dict(x=-1.2, y=1.2, z=-1.2)),
        )

        layout = go.Layout(
            scene=scene,
            scene_aspectmode="cube",
            updatemenus=updatemenus,
            sliders=sliders,
            showlegend=False,
        )
        return layout

    def _make_sliders(self):
        """Returns a list including one Plotly slider that allows users to
        controll the displayed frame."""
        p = self.sequence.positions
        # Frame Slider
        slider = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {
                    "size": 20
                },
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {
                "duration": 0,
                "easing": "linear"
            },
            "pad": {
                "b": 10,
                "t": 50
            },
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": []
        }
        for i in range(0, len(p)):
            # Create slider step for each frame
            slider_step = {
                "args": [[i], {
                    "frame": {
                        "duration": 33,
                        "redraw": True
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 0
                    }
                }],
                "label": i,
                "method": "animate"
            }
            slider["steps"].append(slider_step)

        return [slider]

    def _make_buttons(self):
        """Returns a list of Plotly buttons to start and stop the animation."""
        # Play / Pause Buttons
        buttons = [{
            "buttons": [{
                "label": "Play",
                "args": [None, {
                    "frame": {
                        "duration": 33,
                        "redraw": True
                    },
                    "fromcurrent": True,
                    "transition": {
                        "duration": 33,
                        "easing": "linear"
                    }
                }],
                "method": "animate"
            }, {
                "label": "Pause",
                "args": [[None], {
                    "frame": {
                        "duration": 33,
                        "redraw": False
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 33
                    }
                }],
                "method": "animate"
            }],
            "direction": "left",
            "pad": {
                "r": 10,
                "t": 87
            },
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]  # yapf: disable
        return buttons

    def _get_frames(self):
        """Returns a list of frames.

        Each frame represents a single scatter plot showing the
        skeleton.
        """
        # No animation frames needed when visualising only one frame
        if len(self.sequence) <= 1:
            return []

        p = self.sequence.positions
        frames = []
        for i in range(0, len(p)):
            frame = {"data": self._get_traces(i), "name": i}
            frames.append(frame)
        return frames

    def _make_joint_traces(self, frame):
        p = self.sequence.positions
        trace_joints = go.Scatter3d(x=p[frame, :, 0], y=p[frame, :, 1], z=p[frame, :, 2], mode="markers", marker=dict(color="royalblue", size=5))
        return [trace_joints]

    def _make_limb_traces(self, frame):
        p = self.sequence.positions
        bp = self.sequence.body_parts
        # Each element represents a pair of body part indices in sequence.positions that will be connected with a line
        limb_connections = [
            [bp["head"], bp["neck"]],
            [bp["neck"], bp["shoulder_l"]],
            [bp["neck"], bp["shoulder_r"]],
            [bp["shoulder_l"], bp["elbow_l"]],
            [bp["shoulder_r"], bp["elbow_r"]],
            [bp["elbow_l"], bp["wrist_l"]],
            [bp["elbow_r"], bp["wrist_r"]],
            [bp["neck"], bp["torso"]],
            [bp["torso"], bp["pelvis"]],
            [bp["pelvis"], bp["hip_l"]],
            [bp["pelvis"], bp["hip_r"]],
            [bp["hip_l"], bp["knee_l"]],
            [bp["hip_r"], bp["knee_r"]],
            [bp["knee_l"], bp["ankle_l"]],
            [bp["knee_r"], bp["ankle_r"]]
        ] # yapf: disable
        limb_traces = []
        for limb in limb_connections:
            limb_trace = go.Scatter3d(x=[p[frame, limb[0], 0], p[frame, limb[1], 0]],
                                      y=[p[frame, limb[0], 1], p[frame, limb[1], 1]],
                                      z=[p[frame, limb[0], 2], p[frame, limb[1], 2]],
                                      mode="lines",
                                      line=dict(color="firebrick", width=5))
            limb_traces.append(limb_trace)
        return limb_traces

    def _make_lcs_trace(self, origin, x_direction_pos, y_direction_pos):
        """Returns a list that contains a plotly trace object the X, Y and Z
        axes of the local joint coordinate system calculated from an origin, a
        X-axis-direction and a Y-axis-direction."""

        # Get Local Coordinate System vectors
        lcs = transformations.get_local_coordinate_system_direction_vectors(origin, x_direction_pos, y_direction_pos)

        # Set Local Coordinate System vectors' length to 100 and move relative to local origin.
        lcs[0] = lcs[0] * 100 + origin
        lcs[1] = lcs[1] * 100 + origin
        lcs[2] = lcs[2] * 100 + origin
        trace_x = go.Scatter3d(x=[origin[0], lcs[0, 0]], y=[origin[1], lcs[0, 1]], z=[origin[2], lcs[0, 2]], mode="lines", marker=dict(color="red"))
        trace_y = go.Scatter3d(x=[origin[0], lcs[1, 0]], y=[origin[1], lcs[1, 1]], z=[origin[2], lcs[1, 2]], mode="lines", marker=dict(color="green"))
        trace_z = go.Scatter3d(x=[origin[0], lcs[2, 0]], y=[origin[1], lcs[2, 1]], z=[origin[2], lcs[2, 2]], mode="lines", marker=dict(color="blue"))
        return [trace_x, trace_y, trace_z]

    def _make_pelvis_cs_trace(self, frame):
        # TODO: Refactor before develop merge
        bp = self.sequence.body_parts
        pcs = transformations.get_pelvis_coordinate_system(self.sequence.positions[frame][bp["pelvis"]], self.sequence.positions[frame][bp["torso"]],
                                                           self.sequence.positions[frame][bp["hip_l"]], self.sequence.positions[frame][bp["hip_r"]])
        p_origin = pcs[0][0]
        pcs[0][1][0] = pcs[0][1][0] * 100 + p_origin
        pcs[0][1][1] = pcs[0][1][1] * 100 + p_origin
        pcs[0][1][2] = pcs[0][1][2] * 100 + p_origin
        trace_x = go.Scatter3d(x=[p_origin[0], pcs[0][1][0][0]],
                               y=[p_origin[1], pcs[0][1][0][1]],
                               z=[p_origin[2], pcs[0][1][0][2]],
                               mode="lines",
                               marker=dict(color="red"))
        trace_y = go.Scatter3d(x=[p_origin[0], pcs[0][1][1][0]],
                               y=[p_origin[1], pcs[0][1][1][1]],
                               z=[p_origin[2], pcs[0][1][1][2]],
                               mode="lines",
                               marker=dict(color="green"))
        trace_z = go.Scatter3d(x=[p_origin[0], pcs[0][1][2][0]],
                               y=[p_origin[1], pcs[0][1][2][1]],
                               z=[p_origin[2], pcs[0][1][2][2]],
                               mode="lines",
                               marker=dict(color="blue"))
        return [trace_x, trace_y, trace_z]

    def _make_jcs_traces(self, frame):
        """Returns a list of Plotly  traces that display a Joint Coordinate
        System for each ball joint respectively."""
        # p = self.sequence.positions
        # bps = self.sequence.body_parts
        # ls_lcs_traces = self._make_lcs_trace(p[frame, bps["LeftShoulder"]], p[frame, bps["RightShoulder"]], p[frame, bps["Torso"]])
        # rs_lcs_traces = self._make_lcs_trace(p[frame, bps["RightShoulder"]], p[frame, bps["LeftShoulder"]], p[frame, bps["Torso"]])
        # lh_lcs_traces = self._make_lcs_trace(p[frame, bps["LeftHip"]], p[frame, bps["RightHip"]], p[frame, bps["Torso"]])
        # rh_lcs_traces = self._make_lcs_trace(p[frame, bps["RightHip"]], p[frame, bps["LeftHip"]], p[frame, bps["Torso"]])
        pelvis_cs_trace = self._make_pelvis_cs_trace(frame)

        # jcs_traces = ls_lcs_traces + rs_lcs_traces + lh_lcs_traces + rh_lcs_traces + pelvis_cs_trace
        jcs_traces = pelvis_cs_trace
        return jcs_traces

    def _get_traces(self, frame):
        """Returns joint, limb and JCS Plotly traces."""
        joint_traces = self._make_joint_traces(frame)
        limb_traces = self._make_limb_traces(frame)
        jcs_traces = self._make_jcs_traces(frame)

        return joint_traces + limb_traces + jcs_traces
