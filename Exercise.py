import warnings

class Exercise:

    def __init__(self, name: str, angles: dict, userId: int, description: str = "Sorry, there is no description"):
        # str - The name of this exercise
        self.name = name
        # dict - The angle restrictions for start/end state, for relevant bodyparts
        self.angles = angles
        # int - The number of sets for this exercise
        self.userId = userId
        self.description = description

    def addJointsToAngle(self, bodypart: str, anglename: str, joints: dict):
        """ Adds a 'joints' key:value pair to self.angles for the 
            specified bodypart key and inner anglename key
            
            Example for a joints key:value pair
            ----------
            "joints": { angle_vertex: 1, rays: [0, 2] }
                -> angle_vertex: The origin of both rays and joint to check the angle for
                -> rays: The rays that span the angle

        """
        # Try to add given joints dict to specified angle
        try:
            self.angles["start"][bodypart][anglename]["joints"] = joints
            self.angles["start"][bodypart][anglename]["joints"] = joints
        except KeyError as e:
            print(e)
