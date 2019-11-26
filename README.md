# Human Motion Analysis - Sport Exercises

This software delivers functions to analyse human exercise motion sequences. The main functionalities are the identification of single iteration subsequences, the identification of the exercise performed in motion sequences and finally the rating of the trainees exercise execution performance'.

## Getting Started

1. Clone or download this repository
2. Download and install [Anaconda3 for python 3.x](https://www.anaconda.com/distribution/)
3. Load and activate the `./enviroment.yml` Anaconda3 enviroment
4. Run, edit, add scripts in `./src`

## Usage Scenario

This module delivers classes and functions to analyse human motion exercise sequences.
To get started, initialise a `PoseProcessor` with the desired `PoseFormat`. All currently available formats are listed in the `PoseFormatEnum`.
The `PoseProcessor` loads `JSON` sequence files and normalises positional body part information as well as the body part format in order to retrieve a uniform representation of a motion sequence as a `Sequence` object instance.<br/>
Whenever a `Sequence` object is initialised, joint angles are calculated from the body part positions and stored within the `joint_angles` attribute of the `Sequence` instance.<br/>
Next, we want to identify Subsequences of single exercise repetitions in the loaded `Sequence`. The `find_iteration_keypoints` method of the `ExerciseEvaluator` class performs this task. So first, we initialise an `Exercise` of the same exercise performed in the loaded `Sequence`. Then, we initialise an `ExerciseEvaluator` instance with the loaded `Exercise` and `Sequence` objects. After that, we can just call the `find_iteration_keypoints` method which returns the _start_, _turning_ and _end_ frame indices of all iterations within the loaded `Sequence`.<br/>
Now, as we identified single iteration subsequences within our `Sequence`, we are able to slice our subsequences out of that `Sequence` and evaluate the exercise execution performance by using the `ExerciseEvaluator` `evaluate` method. First, we have to change the `ExerciseEvaluator` `Sequence` to one of our single iteration ubsequences by calling the `set_sequence` method. Then, we call the `evaluate` method with a _turning_frame_ index parameter and receive the trainees exersice performance result, which most importantly includes `ResultStates` that tell us whether a predefined target angle of the `Exercise` has been reached and whether is was exceeded.<br/>
Another rating approach utilises the _Dynamic Time Walking_ algorithm and determines an _aligned distance_ between two motion sequences. This distance is interpreted as similarity between two motion sequences. To determine the distance between two sequences, load two sequences and reformat their `joint_angles` by calling the `helpers.reformat_angles_dtw` function with a sequence. Then just call the `tslearn.metrics.dtw_path` function with the reformatted angles of both sequences. The returned tuple includes the perfect DTW path and the resulting _DTW distance_. If we assume that one sequence represents the perfect execution of an exercise, and the other is a query sequence performed by a patient or trainee, the resulting distance indicates how well the trainee performed.<br/>
Another usecase for the _DTW distance_ is to identify which exercise has been performed in a motion sequence by performing the _DTW_ comparison with several ground-truth sequences, which represent a particular exercise. The _closest_ ground-truth sequence to our query sequence should then represent the exercise that is performed in the query sequence.<br/>
The next chapter presents example procedures and code examples to perform some of the tasks stated above.

## Usage Examples

The following examples shall clarify how to use the features of this module.
A list of example functions very similar to the following, can be found in the `src/hma/examples.py` script file.
<br/>
<br/>
Exercise and Sequence `JSON` files can be found in `data/exercises` and `data/sequences/191024_tracking` folders.

### Loading a Sequence from file

1. Init a `PoseProcessor`
2. Get sequence instance from sequence json file

```python
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename = "path/to/sequence.json"
    sequence = mocap_poseprocessor.load(filename)
```

### Loading an Exercise from file

1. Load an exercise by using the `exercise_loader.load` function

```python
    exercise = exercise_loader.load("path/to/exercise.json")
```

### Assume the type of an exercise

1. Load Query Sequence and Ground-Truth Sequences
2. Reformat angles to suit the tslearn.metrics.dtw_path function
3. Get DTW Path and DTW Distance between Query Sequence and Ground-Truth Sequences using the tslearn.metrics.dtw_path function
4. Determine Query Sequences Exercise Type checking which Ground-Truth Sequence has the lowest DTW Distance to the Query Sequence

```python
    # Load Query Sequence and Ground-Truth Sequences
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename_q = "path/to/query_sequence.json"
    filename_squat_gt = "path/to/squat_ground_truth.json"
    filename_biceps_curl_left_gt = "path/to/biceps_curl_left_ground_truth.json"
    sequence_q = mocap_poseprocessor.load(filename_q, "Query Sequence")
    sequence_squat_gt = mocap_poseprocessor.load(filename_squat_gt, "Squat Ground-Truth")
    sequence_biceps_curl_left_gt = mocap_poseprocessor.load(filename_biceps_curl_left_gt, "Biceps Curl Left Ground-Truth")
    # Reformat angles to suit the tslearn.metrics.dtw_path function
    angles_q = reformat_angles_dtw(sequence_q)
    angles_squat_gt = reformat_angles_dtw(sequence_squat_gt)
    angles_biceps_curl_left_gt = reformat_angles_dtw(sequence_biceps_curl_left_gt)
    # Get DTW Path and DTW Distance between Query Sequence and Ground-Truth Sequences
    path_squat_gt, dist_squat_gt = ts.dtw_path(angles_q, angles_squat_gt)
    path_biceps_curl_left_gt, dist_biceps_curl_left_gt = ts.dtw_path(angles_q, angles_biceps_curl_left_gt)
```

### Identify subsequences consisting of a single exercise repitition

1. Load Sequence
2. Load Exercise of that Sequence
3. Init ExerciseEvaluator with Exercise and Sequence
4. Identify single iteration Subsequences

```python
    # Load Sequence and Exercise
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename = "data/sequences/191024_tracking/multi/squat/user-3/191024__multi__squat__user-3__0.json"
    sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
    squat = exercise_loader.load("data/exercises/squat.json")
    # Init ExerciseEvaluiator with an Exercise and a Sequence
    EE = ExerciseEvaluator(squat, sequence)
    # Identify Single Iteration Subsequences and plot result graph
    iterations = EE.find_iteration_keypoints(plot=True)
    # iterations: [[10  109  168], [ 168  236  303], [ 303  363  431], ...]
```

### Rating trainees exercise execution performance - Angle Comparison Method

1. Load Sequence
2. Load Exercise of that Sequence
3. Init `ExerciseEvaluator` with Exercise and Sequence
4. Identify single iteration Subsequences _(even in single iteration sequences we have to identify thatr one iteration to know the index of the turning frame)_
5. Set `ExererciseEvaluator` Sequence to an identified subsequence using the `set_sequence` method
6. Get rating result from `ExerciseEvaluator` `evaluate` method.
7. Save result to `JSON` file.

```python
    # Load Sequence and Exercise
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
    sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
    squat = exercise_loader.load("data/exercises/squat.json")
    # Init ExerciseEvaluiator with an Exercise and a Sequence
    EE = ExerciseEvaluator(squat, sequence)
    # Identify Single Iteration Subsequences
    # NOTE: The Angle Comparison rating method needs the Turning Frame index as parameter,
    #       thus we first have to utilise the find_iteration_keypoints method of the ExerciseEvaluator.
    iterations = EE.find_iteration_keypoints(plot)
    if len(iterations) <= 0:
        print("No iterations identified. Unable to rate execution performance.")
    else:
        for i, it in enumerate(iterations):
            # Set Sequence to a identified Subsequence
            EE.set_sequence(EE.sequence[it[0]:it[2]])
            # The second element of an iteration is the turning frame.
            turning_frame = it[1]
            rating_result = EE.evaluate(turning_frame)
            # Create result JSON file
            with open(f"data/rating_result_{i}.json", 'w') as outfile:
                json.dump({"result": rating_result}, outfile)
```
