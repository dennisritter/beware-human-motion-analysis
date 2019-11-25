# A Tool to analyse human motion sequences of sport exercises.

## Getting Started

1. Clone or download this repository
2. Load and activate the `enviroment.yml` Anaconda3 enviroment
3. run, edit, add scripts in `./src`

## Usage

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
    exercise = exercise_loader.load("data/exercises/squat.json")
```

### Assume the type of an exercise

1. Load Query Sequence and Ground-Truth Sequences
2. Reformat angles to suit the tslearn.metrics.dtw_path function
3. Get DTW Path and DTW Distance between Query Sequence and Ground-Truth Sequences using the tslearn.metrics.dtw_path function
4. Determine Query Sequences Exercise Type by The lowest Ground-Truth distance

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

### Rating trainees exercise execution performance - Angle Comparison Method

### Rating trainees exercise execution performance - DTW Distance Method

### Contributing

Use [gitflow](https://de.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) as git workflow when implementing features, please.
