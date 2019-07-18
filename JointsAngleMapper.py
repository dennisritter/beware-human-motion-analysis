from PoseFormatEnum import PoseFormatEnum
from Exercise import Exercise 

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
                                    },
    
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
        exercise.addJointsToAngle("hip_left", "flexion_extension", { "angle_vertex": 0, "rays": [1, 2] })