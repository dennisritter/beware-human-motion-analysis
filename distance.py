from scipy.spatial.distance import directed_hausdorff


def hausdorff(u, v):
    return directed_hausdorff(u, v)
