import pickle

import numpy as np
import os
import gc
from joblib import Parallel, delayed

def make_inds(i_vals, numex):
    tmp1 = []
    for i in i_vals:
        tmp2 = []
        for j in range(i + 1, numex):
            tmp2.append(np.array([i, j], dtype=np.uint32))
        if len(tmp2)>0:
            tmp1.append(np.vstack(tmp2))
    return np.vstack(tmp1, dtype=np.uint32) if len(tmp1) > 0 else np.ndarray(shape=(0,2), dtype=np.uint32)


def q_gram_dice_sim(q_gram_set1, q_gram_set2):
    """calculate the dice similarity between two given sets of q-grams.
       Dice sim(A,B)= 2 x number of common elements of A and B
                      ----------------------------------------
                      number of elements in A + number of elements in B
       returns a similarity value between 0 and 1.
    """

    num_common_q_gram = len(q_gram_set1 & q_gram_set2)

    q_gram_dice_sim = (2.0 * num_common_q_gram) / \
                      (len(q_gram_set1) + len(q_gram_set2))
    assert 0 <= q_gram_dice_sim and q_gram_dice_sim <= 1.0

    return q_gram_dice_sim



def q_gram_jacc_sim(q_gram_set1, q_gram_set2):
    """calculate the jaccard similarity between two given sets of q-grams.
       Jaccard sim(A,B) = |A intersection B|
                          ------------------
                              |A Union B|
       returns a value between 0 and 1.
    """

    q_gram_intersection_set = q_gram_set1 & q_gram_set2
    q_gram_union_set = q_gram_set1 | q_gram_set2

    q_gram_jacc_sim = float(len(q_gram_intersection_set) / len(q_gram_union_set))

    assert 0 <= q_gram_jacc_sim and q_gram_jacc_sim <= 1

    return q_gram_jacc_sim


def compute_metrics(inds, cache, uids, metric, sim):
    tmp = np.zeros(len(inds), dtype=np.float32)
    pos = 0
    prev_i = prev_j = None
    for i, j in inds:
        if i != prev_i:
            i_enc = cache[uids[i]]
            prev_i = i
        if j != prev_j:
            j_enc = cache[uids[j]]
            prev_j = j
        if metric == "jaccard":
            val = q_gram_jacc_sim(i_enc, j_enc)
        else:
            val = q_gram_dice_sim(i_enc, j_enc)

        if not sim:
            val = 1 - val
        tmp[pos] = val
        pos += 1
    return tmp

def calc_ngram(string, n):
    string = ["".join(string).replace(" ", "").lower()]
    return set([[b[i:i + n] for i in range(len(b) - n + 1)] for b in string][0])


class NonEncoder():

    def __init__(self, ngram_size: int, verbose: bool = False, workers=-1):
        """

        """
        self.ngram_size = ngram_size
        self.verbose = verbose
        self.workers = os.cpu_count() if workers == -1 else workers

    def encode_and_compare(self, data, uids, metric, sim=True, store_encs = False):
        available_metrics = ["jaccard", "dice"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        numex = len(uids)
        uids = np.array(uids, dtype=np.float32)

        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(delayed(calc_ngram)(d, 2) for d in data)
        cache = {}

        for i, enc in enumerate(output_generator):
            cache[uids[i]] = enc
        del output_generator, data
        gc.collect()

        if store_encs:
            with open("./data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

        output_generator = parallel(delayed(make_inds)(i, numex) for i in np.array_split(np.arange(numex), self.workers))

        inds = np.vstack(output_generator)
        numinds  = len(inds)
        inds = np.array_split(inds, self.workers)
        gc.collect()

        pw_metrics = parallel(delayed(compute_metrics)(i, cache, uids, metric, sim) for i in inds)
        del cache
        gc.collect()

        pw_metrics = np.concatenate(pw_metrics, axis=None)

        re = np.zeros((numinds, 3), dtype=np.float32)

        re[:,2] = pw_metrics
        del pw_metrics
        gc.collect()

        start = 0
        for i, ind in enumerate(inds):
            ind[:, 0] = uids[ind[:, 0]]
            ind[:, 1] = uids[ind[:, 1]]
            end = start + len(ind)
            re[start:end,0:2] = ind
            start += len(ind)
        del inds
        gc.collect()
        #...and add the metrics
        return re