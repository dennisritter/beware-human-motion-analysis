from scipy.spatial.distance import directed_hausdorff
import fastdtw


def hausdorff(u, v):
    return directed_hausdorff(u, v)


def dtw(u, v):
    return fastdtw.fastdtw(u, v)
