from PoseFormatEnum import PoseFormatEnum

class ExercisePoseSegmentor:
    """ 
    """

    def __init__(self, poseformat: PoseFormatEnum):
        """ ExercisePoseSegmentor Constructor
        Parameters
        ----------
        poseformat : PoseFormatEnum
            A PoseFormatEnum member defining the skeleton format of the tracked motion sequence.

        Returns
        ----------
        ExercisePoseSegmentor
            A ExercisePoseSegmentor instance.
        """
        if(not isinstance(poseformat, PoseFormatEnum)):
            raise ValueError(
                "'poseformat' parameter must be a member of 'PoseFormatEnum'")
        self.poseformat = poseformat
