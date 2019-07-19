import sys
from Exercise import Exercise
sys.path.append('src/motion_sequences/')
from PoseFormatEnum import PoseFormatEnum  # noqa: E402


class JointsAngleMapper:
    """ Segments a set of joints of a pose of the specified poseformat into subsets of those joints.
    """

    def __init__(self, poseformat: PoseFormatEnum):
        """ JointsAngleMapper Constructor
        Parameters
        ----------
        poseformat : PoseFormatEnum
            A PoseFormatEnum member defining the skeleton format of the tracked motion sequence.

        Returns
        ----------
        JointsAngleMapper
            A JointsAngleMapper instance.
        """
        if(not isinstance(poseformat, PoseFormatEnum)):
            raise ValueError(
                "'poseformat' parameter must be a member of 'PoseFormatEnum'")
        self.poseformat = poseformat
        self.poseformatjoints = {}
        if (poseformat == PoseFormatEnum.MOCAP):
            self.poseformatjoints = {
                "RightWrist": 0,
                "RightElbow": 1,
                "RightShoulder": 2,
                "Neck": 3,
                "Torso": 4,
                "Waist": 5,
                "RightAnkle": 6,
                "RightKnee": 7,
                "RightHip": 8,
                "LeftAnkle": 9,
                "LeftKnee": 10,
                "LeftHip": 11,
                "LeftWrist": 12,
                "LeftElbow": 13,
                "LeftShoulder": 14,
                "Head": 15
            }

    def addJointsToAngles(self, exercise: Exercise) -> dict:
        """ Returns a Dictionary of mapped sets of three joints for bodypart/joint names and the possible movements.
        Return Format Example
        --------------
        {
            "Elbow_left": {
                "flexion_extension": { "angle_vertex": 13, "rays": [12, 14] }
            },
            "Elbow_right": {
                "flexion_extension": { "angle_vertex": 1, "rays": [0, 2] }
            }
        }
        Integer values represent the order_index of the poseformatjoints and describe where the position of that joint is stored
        in the second dimension list of Sequence.positions.
        Example usage: Sequence.positions[num_frame][joint_index] -> will get you the [x, y, z] position for a frame and a specific joint.
        """
        if (self.poseformat == PoseFormatEnum.MOCAP):
            return self.addJointstoAnglesMocap(exercise)

    def addJointstoAnglesMocap(self, exercise: Exercise):
        """ See description of 'getJointSets' method. Returns the jointsets for the MOCAP poseformat.
        """
        joints = self.poseformatjoints
        """ Defines which joints must be used for angle calculations
        NOTE:   The used rays for angle calculations might differ from the vectors used in medical angle definitions
                because in some cases the medical vectors are hard to identify for example.
        Find Medical definitions here:
            documents/Messblatt_Obere_Extremität.pdf
            documents/Messblatt_Untere_Extremität.pdf
            ROM.pdf
        """
        joints_angle_map = {
            "hip_left": {
                # NOTE: Using Hip-Torso vector for Hip angle calculation -> transfer result to medical definition
                "flexion_extension": {"angle_vertex": joints["LeftHip"], "rays": [joints["LeftKnee"], joints["Torso"]]},
                "innerrotation_outerrotation": {"angle_vertex": joints["LeftHip"], "rays": [joints["LeftKnee"], joints["LeftAnkle"]]},
                # NOTE: Using Hip-Torso vector for Hip angle calculation -> transfer result to medical definition
                "abduction_adduction": {"angle_vertex": joints["LeftHip"], "rays": [joints["LeftKnee"], joints["Torso"]]}
            },
            "hip_right": {
                # NOTE: Using Hip-Torso vector for Hip angle calculation -> transfer result to medical definition
                "flexion_extension": {"angle_vertex": joints["RightHip"], "rays": [joints["RightKnee"], joints["Torso"]]},
                "innerrotation_outerrotation": {"angle_vertex": joints["RightHip"], "rays": [joints["RightKnee"], joints["RightAnkle"]]},
                # NOTE: Using Hip-Torso vector for Hip angle calculation -> transfer result to medical definition
                "abduction_adduction": {"angle_vertex": joints["RightHip"], "rays": [joints["RightKnee"], joints["Torso"]]}
            },
            "knee_left": {
                "flexion_extension": {"angle_vertex": joints["LeftKnee"], "rays": [joints["LeftHip"], joints["LeftAnkle"]]},
                # NOTE: Is this even relevant? documents/Messblatt_Untere_Extremität.pdf does not mention this
                "innerrotation_outerrotation": {"angle_vertex": joints["LeftKnee"], "rays": [0, 0]}
            },
            "knee_right": {
                "flexion_extension": {"angle_vertex": joints["RightKnee"], "rays": [joints["RightHip"], joints["RightAnkle"]]},
                # NOTE: Is this even relevant? documents/Messblatt_Untere_Extremität.pdf does not mention this
                "innerrotation_outerrotation": {"angle_vertex": joints["RightKnee"], "rays": [0, 0]}
            },
            "shoulder_left": {
                # NOTE: Using Shoulder-Hip Vector as 0° reference
                "flexion_extension": {"angle_vertex": joints["LeftShoulder"], "rays": [joints["LeftElbow"], joints["LeftHip"]]},
                "innerrotation_outerrotation": {"angle_vertex": joints["LeftShoulder"], "rays": [joints["LeftElbow"], joints["LeftHip"]]},
                # NOTE: Using Shoulder-Hip Vector as 0° reference
                "abduction_adduction": {"angle_vertex": joints["LeftShoulder"], "rays": [joints["LeftElbow"], joints["LeftHip"]]}
            },
            "shoulder_right": {
                # NOTE: Using Shoulder-Hip Vector as 0° reference
                "flexion_extension": {"angle_vertex": joints["RightShoulder"], "rays": [joints["RightElbow"], joints["RightHip"]]},
                "innerrotation_outerrotation": {"angle_vertex": joints["RightShoulder"], "rays": [joints["RightElbow"], joints["RightHip"]]},
                "abduction_adduction": {"angle_vertex": joints["RightShoulder"], "rays": [joints["RightElbow"], joints["RightHip"]]}
            },
            "elbow_left": {
                "flexion_extension": {"angle_vertex": joints["LeftElbow"], "rays": [joints["LeftWrist"], joints["LeftShoulder"]]}
            },
            "elbow_right": {
                "flexion_extension": {"angle_vertex": joints["RightElbow"], "rays": [joints["RightWrist"], joints["RightShoulder"]]}
            }
        }
        # Add joints to all angles for all bodyparts
        for bodypart in joints_angle_map:
            for angle_name in joints_angle_map[bodypart]:
                exercise.addJointsToAngle(bodypart, angle_name, joints_angle_map[bodypart][angle_name])
