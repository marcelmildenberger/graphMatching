def calc_condensed_index(i: int, j: int, m: int) -> int:
    """
    Computes the index of the pairwise distances between X[i] and X[j] in a scipy condensed distance matrix.
    Note: i<j<m
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    :param i: The index of the first element in the original matrix
    :param j: The index of the second element in the original matrix
    :param m: Number of observations in the original matrix
    :return: Index at which the distance between X[i] and X[j] can be found in the condensed distance matrix
    """
    return m * i + j - ((i + 2) * (i + 1)) // 2
