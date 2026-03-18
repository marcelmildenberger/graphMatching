import pickle

import numpy as np
import os
import gc
from joblib import Parallel, delayed
from .encoder import Encoder, normalize_joined_record
from .pairwise_set_metrics import assemble_pairwise_results, compute_set_metrics, make_inds

def calc_ngram(string, n):
    normalized = normalize_joined_record(string)
    return {normalized[i:i + n] for i in range(len(normalized) - n + 1)}


class NonEncoder(Encoder):

    def __init__(self, ngram_size: int, verbose: bool = False, workers=-1):
        """

        """
        self.ngram_size = ngram_size
        self.verbose = verbose
        self.workers = os.cpu_count() if workers == -1 else workers

    def encode_and_compare_and_append(self, data, uids, metric, sim=True, store_encs = False):
        combined_data = np.column_stack((data, uids))
        available_metrics = ["jaccard", "dice"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        numex = len(uids)
        uids = np.array(uids, dtype=np.float32)

        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(delayed(calc_ngram)(d, self.ngram_size) for d in data)
        cache = {}

        for i, enc in enumerate(output_generator):
            cache[uids[i]] = enc
        del output_generator, data
        gc.collect()

        if store_encs:
            tmpdict = dict()

            for key, val in cache.items():
                tmpdict[str(int(key))] = val
            with open("./graphMatching/data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(tmpdict, f, pickle.HIGHEST_PROTOCOL)
            del tmpdict

        output_generator = parallel(delayed(make_inds)(i, numex, np.uint32) for i in np.array_split(np.arange(numex), self.workers))

        inds = np.vstack(output_generator)
        inds = np.array_split(inds, self.workers)
        gc.collect()

        pw_metrics = parallel(delayed(compute_set_metrics)(i, cache, uids, metric, sim) for i in inds)
        del cache
        gc.collect()

        pw_metrics = np.concatenate(pw_metrics, axis=None)
        re = assemble_pairwise_results(inds, uids, pw_metrics)
        del pw_metrics
        del inds
        gc.collect()
        #...and add the metrics
        return re, combined_data


    def get_encoding_dict(self, data, uids):
        uids = np.array(uids, dtype=np.float32)

        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(delayed(calc_ngram)(d, self.ngram_size) for d in data)
        cache = {}

        for i, enc in enumerate(output_generator):
            cache[uids[i]] = enc
        del output_generator, data
        gc.collect()

        tmpdict = dict()

        for key, val in cache.items():
            tmpdict[str(int(key))] = val

        return tmpdict
