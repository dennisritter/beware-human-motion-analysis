import plotly.graph_objects as go
import numpy as np
from hma.movement_analysis.sequence import Sequence
from hma.movement_analysis import transformations


class SkeletonVisualiser:
    """Visualises a human pose skeleton as an animated 3D Scatter Plot.


    Attributes:
        sequence (Sequence): The motion sequence to visualise the skeleton from. 
    """
    def __init__(
            self,
            sequence: Sequence,
    ):

        self.sequence = sequence[:]

    def show(self):
        """ Visualises the human pose skeleton as an animated 3D Scatter Plot.
        """
        traces = self._get_traces(0)
        layout = self._get_layout()
        frames = self._get_frames()

        fig = go.Figure(data=traces, layout=layout, frames=frames)
        fig.show()

    def _get_layout(self):
        updatemenus = []
        sliders = []
        if len(self.sequence) > 1:
            updatemenus = self._make_buttons()
            sliders = self._make_sliders()

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
            ),
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
        p = self.sequence.positions
        # Frame Slider
        slider = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Frame: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0, "easing": "linear"},
            "pad": {"b": 10, "t": 50},
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
        # Play / Pause Buttons
        buttons = [{
            "buttons": [
                {
                    "label": "Play",
                    "args": [None, {"frame": {"duration": 33, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 33,
                                                                        "easing": "linear"}}],
                    "method": "animate"
                },
                {
                    "label": "Pause",
                    "args": [[None], {"frame": {"duration": 33, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 33}}],
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
        return buttons


    def _get_frames(self):
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
        trace_joints = go.Scatter3d(x=p[frame, :, 0], y=p[frame, :, 1], z=p[frame, :, 2], mode="markers", marker=dict(color="royalblue"))
        return [trace_joints]

    def _make_limb_traces(self, frame):
        p = self.sequence.positions
        # Each element represents a pair of body part indices in sequence.positions that will be connected with a line
        limb_connections = [[1, 0], [2, 1], [3, 2], [3, 14], [14, 13], [13, 12], [15, 3], [3, 4], [4, 5], [5, 8], [5, 11], [11, 10], [10, 9], [8, 7], [7, 6]]
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
        """ Returns a list that contains a plotly trace object the X, Y and Z axes of the local joint coordinate system
            calculated from an origin, a X-axis-direction and a Y-axis-direction.
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

    def _make_jcs_traces(self, frame):
        p = self.sequence.positions
        bps = self.sequence.body_parts
        ls_lcs_traces = self._make_lcs_trace(p[frame, bps["LeftShoulder"]], p[frame, bps["RightShoulder"]], p[frame, bps["Torso"]])
        rs_lcs_traces = self._make_lcs_trace(p[frame, bps["RightShoulder"]], p[frame, bps["LeftShoulder"]], p[frame, bps["Torso"]])
        lh_lcs_traces = self._make_lcs_trace(p[frame, bps["LeftHip"]], p[frame, bps["RightHip"]], p[frame, bps["Torso"]])
        rh_lcs_traces = self._make_lcs_trace(p[frame, bps["RightHip"]], p[frame, bps["LeftHip"]], p[frame, bps["Torso"]])

        jcs_traces = ls_lcs_traces + rs_lcs_traces + lh_lcs_traces + rh_lcs_traces
        return jcs_traces

    def _get_traces(self, frame):
        joint_traces = self._make_joint_traces(frame)
        limb_traces = self._make_limb_traces(frame)
        jcs_traces = self._make_jcs_traces(frame)

        return joint_traces + limb_traces + jcs_traces
