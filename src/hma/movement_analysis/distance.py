from scipy.spatial import distance
import fastdtw


def hausdorff(u, v):
    return distance.directed_hausdorff(u, v)


def euclidean(u, v):
    return distance.euclidean(u, v)


def dtw(u, v):
    return fastdtw.fastdtw(u, v)
