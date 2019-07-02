from enum import Enum


class PoseMappingEnum(Enum):
    MOCAP_TO_MPII = 1


class PoseMapper:

    def __init__(self, mapping: PoseMappingEnum):
        """ PoseMapper Constructor
        Parameters
        ----------
        mapping : PoseMappingEnum
            A PoseMapping Enumeration member defining input/output format of the mapper.

        Returns
        ----------
        PoseMapper
            A PoseMapper instance.
        """
        if(not isinstance(mapping, PoseMappingEnum)):
            raise ValueError(
                "'mapping' parameter must be a member of 'PoseMapping' enumeration.")
        self.mapping = mapping
        print(self.mapping)

    def map(self, input: str) -> str:
        """
        Parameters
        ----------
        input : str
            The Pose input string to convert into the constructor specified output.
        Returns
        ----------
        str   
           The converted output Pose string for the given input in the constructor specified output pose format.   
        """
        if (self.mapping == PoseMappingEnum.MOCAP_TO_MPII):
            return self.map_mocap_to_mpii(input)

    def map_mocap_to_mpii(self, input: str) -> str:
        """
        Parameters
        ----------
        input : str
            The MOCAP Pose input string to convert into the MPII representation.
        Returns
        ----------
        str
           A MPII pose string representation of the MOCAP input pose string.
        """
        return input[::-1]
