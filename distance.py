from scipy.spatial.distance import directed_hausdorff
import fastdtw as dtw


def hausdorff(u, v):
    return directed_hausdorff(u, v)


def fastdtw(u, v):
    return dtw.fastdtw(u, v)
