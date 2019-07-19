## Exercises

### Exercise representation

Exercises are defined by `.json` files and contain the target angle ranges for all relevant joints.
A `Exercise` class instance will be created by loading those `exercise.json` files into the software using the `exercise_loader.py` module.

#### Example file

```json
{
  // Name of the exercise
  "name": "Squat",
  // Id of the user/usergroup that performs that exercise
  // Can be used to specify personalized angles that might apply to a user with joint issues or alike
  "userId": 0,
  // A description for this exercise
  "description": "A Squat for healthy people.",
  // Defined medical start/end state angles to rate the performance for one iteration of the exercise
  "angles": {
    // Starting state
    "start": {
      // Bodypart: Left Hip
      "hip_left": {
        // Flexion and Extension angle
        "flexion_extension": {
          // The accepted angle range [from, to] (+- general tolerance)
          "angle": [0, 0],
          // Priority of that angle for this exercise
          "priority": 1.0
        },
        "innerrotation_outerrotation": {
          "angle": [30, 30],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "hip_right": {
        "flexion_extension": {
          "angle": [0, 0],
          "priority": 1.0
        },
        "innerrotation_outerrotation": {
          "angle": [30, 30],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "knee_left": {
        "flexion_extension": {
          "angle": [0, 0],
          "priority": 1.0
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "knee_right": {
        "flexion_extension": {
          "angle": [0, 0],
          "priority": 1.0
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "shoulder_left": {
        "flexion_extension": {
          "angle": [0, 0],
          "priority": 0.5
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "shoulder_right": {
        "flexion_extension": {
          "angle": [0, 0],
          "priority": 0.5
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "elbow_left": {
        "flexion_extension": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "elbow_right": {
        "flexion_extension": {
          "angle": [0, 0],
          "priority": 0.5
        }
      }
    },
    "end": {
      "hip_left": {
        "flexion_extension": {
          "angle": [45, 90],
          "priority": 0.5
        },
        "innerrotation_outerrotation": {
          "angle": [30, 30],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 10],
          "priority": 0.5
        }
      },
      "hip_right": {
        "flexion_extension": {
          "angle": [45, 90],
          "priority": 0.5
        },
        "innerrotation_outerrotation": {
          "angle": [30, 30],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 10],
          "priority": 0.5
        }
      },
      "knee_left": {
        "flexion_extension": {
          "angle": [45, 90],
          "priority": 1.0
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "knee_right": {
        "flexion_extension": {
          "angle": [45, 90],
          "priority": 1.0
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "shoulder_left": {
        "flexion_extension": {
          "angle": [40, 60],
          "priority": 0.5
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "shoulder_right": {
        "flexion_extension": {
          "angle": [45, 60],
          "priority": 0.5
        },
        "innerrotation_outerrotation": {
          "angle": [0, 0],
          "priority": 0.5
        },
        "abduction_adduction": {
          "angle": [0, 0],
          "priority": 0.5
        }
      },
      "elbow_left": {
        "flexion_extension": {
          "angle": [45, 90],
          "priority": 0.5
        }
      },
      "elbow_right": {
        "flexion_extension": {
          "angle": [45, 90],
          "priority": 0.5
        }
      }
    }
  }
}
```

### Assigning joints for angle calculation

The `JointAngleMapper` class is responsible for assigning the correct joints to all bodypart angles that might be checked.
Its constructor takes a pose format of the `PoseFormatEnum` as parameter. The defined pose format defines how many and which joints will be assigned to the angles of an `Exercise`. Which joints will be applied to the angles has to be defined manually in `JointsAngleMapper.py` for each pose format of `PoseFormatEnum`.

After creating a JointAngleMapper instance for a specific pose format the `JointAngleMapper.addJointstoAnglesMocap(self, exercise: Exercise)` method can be used to add the defined joints to the `angles` property of an `exercise`.

### How Joint angles are medically defined and how they are actually calculated

The software must track human poses and check whether an exercise has been done correctly. To perform this task, the correct start and target (medical) angles for human joints are defined in JSON files representing a single exercise.

In some cases, the medical angles are using angle vectors that can't be used for the corresponding softwares angle calculations since they hard to identify or not generalizable.

#### Hip left/right

##### Flexion/Extension

##### Abduction/Adduction

##### Inner rotation/Outer rotation

#### Knees left/right

##### Flexion/Extension

##### Inner rotation/Outer rotation

#### Shoulders left/right

##### Flexion/Extension

##### Abduction/Adduction

##### Inner rotation/Outer rotation

#### Elbow left/right

##### Flexion/Extension
