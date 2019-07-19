## How Joint angles are medically defined and how they are actually calculated

The software must track human poses and check whether an exercise has been done correctly. To perform this task, the correct start and target (medical) angles for human joints are defined in JSON files representing a single exercise.

In some cases, the medical angles are using angle vectors that can't be used for the corresponding softwares angle calculations since they hard to identify or not generalizable.

### Exercise File Example

<details><summary>Click to expand</summary>
<p>
```json
{
  "name": "Squat",
  "userId": 0,
  "description": "A Squat for healthy people.",
  "angles": {
    "start": {
      "hip_left": {
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
</p>
</details>

### Hip left/right

### Knees left/right

### Shoulders left/right

### Elbow left/right
