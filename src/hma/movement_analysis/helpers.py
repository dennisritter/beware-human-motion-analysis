import random
import networkx as nx
import numpy as np
from .enums.angle_types import AngleTypes
import matplotlib.pyplot as plt


def reformat_angles_dtw(seq):
    """Returns the sequences reformatted joint angles to use them for the
    tslearn.metrics.dtw_path function."""
    bp = seq.body_parts
    dtw_angles = []
    for frame in range(0, len(seq)):
        seq_frame_angles = []
        seq_frame_angles.append(seq.joint_angles[frame][bp["shoulder_l"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["shoulder_l"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["shoulder_r"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["shoulder_r"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["hip_l"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["hip_l"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["hip_r"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["hip_r"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["elbow_l"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["elbow_r"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["knee_l"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["knee_r"]][AngleTypes.FLEX_EX.value])
        dtw_angles.append(seq_frame_angles)
    return np.array(dtw_angles)


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike.

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        """see hierarchy_pos docstring for most arguments.

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def draw_scenegraph(G, root="pelvis"):
    """Draws a graph ordered by the nodes hierarchy.

    Root node will be on top.
    """

    pos = hierarchy_pos(G, root)
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()
