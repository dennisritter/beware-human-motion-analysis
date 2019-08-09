## Exercises

### Exercise representation

Exercises are defined by `.json` files and contain the target angle ranges for all relevant joints.
A `Exercise` class instance will be created by loading those `exercise.json` files into the software using the `exercise_loader.py` module.

#### Example file

```javascript
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

Each function that calculates angles for a particular joint expects an instance of a `Sequence` and indices of necessary joints for the computations.
The necessary joint positions will then be retrieved from the `Sequence.positions` attribute by selecting a timeframe and using the joint index. (e.g.: `my_sequence[<frame>][<joint_index>]`). All joint indices can be found in the `Sequence.body_parts` attribute, which is a dictionary that maps the joint names to it's index for the used tracking technique.

Possibly, the present angle calculation functions to not fit well to arbitrary tracking techniques, because of different numbers and kinds joints. So it might be necessary to implement new angle calculation funtions for different tracking techniques used.

### How Joint angles are medically defined and how they are actually calculated

The software must track human poses and check whether an exercise has been done correctly. To perform this task, the correct start and target (medical) angles for human joints are defined in JSON files representing a single exercise.

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
