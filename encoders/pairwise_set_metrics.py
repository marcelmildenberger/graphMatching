import numpy as np


def make_inds(i_vals, numex, dtype=int):
    chunks = []
    for i in i_vals:
        pairs = []
        for j in range(i + 1, numex):
            pairs.append(np.array([i, j], dtype=dtype))
        if pairs:
            chunks.append(np.vstack(pairs))
    return np.vstack(chunks) if chunks else np.ndarray(shape=(0, 2), dtype=dtype)


def dice_similarity(set_a, set_b):
    denominator = len(set_a) + len(set_b)
    if denominator == 0:
        return 1.0
    return (2.0 * len(set_a & set_b)) / denominator


def jaccard_similarity(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 1.0
    return float(len(set_a & set_b) / len(union))


def compute_set_metrics(inds, cache, uids, metric, sim):
    metrics = np.zeros(len(inds), dtype=np.float32)
    pos = 0
    prev_i = None

    for i, j in inds:
        if i != prev_i:
            i_enc = cache[uids[i]]
            prev_i = i
        j_enc = cache[uids[j]]

        if metric == "jaccard":
            value = jaccard_similarity(i_enc, j_enc)
        else:
            value = dice_similarity(i_enc, j_enc)

        metrics[pos] = 1 - value if not sim else value
        pos += 1

    return metrics


def assemble_pairwise_results(inds_chunks, uids, flat_metrics):
    result = np.zeros((len(flat_metrics), 3), dtype=np.float32)
    result[:, 2] = flat_metrics

    start = 0
    for ind in inds_chunks:
        end = start + len(ind)
        ind[:, 0] = uids[ind[:, 0]]
        ind[:, 1] = uids[ind[:, 1]]
        result[start:end, 0:2] = ind
        start = end

    return result
