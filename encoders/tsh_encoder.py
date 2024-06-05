import os
import gc
import random
import string
import numpy as np
from hashlib import sha256
from joblib import Parallel, delayed


# =============================================================================

def q_gram_dice_sim(q_gram_set1, q_gram_set2):
    """calculate the dice similarity between two given sets of q-grams.
       Dice sim(A,B)= 2 x number of common elements of A and B
                      ----------------------------------------
                      number of elements in A + number of elements in B
       returns a similarity value between 0 and 1.
    """

    num_common_q_gram = len(q_gram_set1.intersection(q_gram_set2))

    q_gram_dice = (2.0 * num_common_q_gram) / \
                  (len(q_gram_set1) + len(q_gram_set2))

    return q_gram_dice


# -----------------------------------------------------------------------------

def q_gram_jacc_sim(q_gram_set1, q_gram_set2):
    """calculate the jaccard similarity between two given sets of q-grams.
       Jaccard sim(A,B) = |A intersection B|
                          ------------------
                              |A Union B|
       returns a value between 0 and 1.
    """

    q_gram_intersection_set = q_gram_set1.intersection(q_gram_set2)
    q_gram_union_set = q_gram_set1.union(q_gram_set2)

    q_gram_jacc = float(len(q_gram_intersection_set) / len(q_gram_union_set))

    return q_gram_jacc


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


def make_inds(i_vals, numex):
    tmp1 = []
    for i in i_vals:
        tmp2 = []
        for j in range(i + 1, numex):
            tmp2.append(np.array([i, j], dtype=np.uint32))
        if len(tmp2) > 0:
            tmp1.append(np.vstack(tmp2))
    return np.vstack(tmp1, dtype=np.uint32)


class TSHEncoder():
    """A class that implements a column-based vector hashing approach for PPRL
       to encode strings into integer hash value sets.
    """

    def __init__(self, num_hash_funct, num_hash_col, ngram_size, rand_mode='PNG', secret=None, verbose=True, workers=-1,
                 seed=None):
        """To initialise the class we need to set the number of hash functions,
           the number of hash columns (bit arrays) to generate, the maximum
           integer for the random numbers to be generated, and a counter for
           which random value to be used in the final hash vector for each column.

           Input arguments:
             - num_hash_funct  The number of hash functions to be used to hash
                               each q-gram of an input q-grams set.
             - num_hash_col    The number of hash columns, i.e. bit arrays, to be
                               generated.
             - rand_mode       The random integer number generation mode, this can
                               either be generated using SHA-256 (SHA) or using a
                               pseudo random number generator (PNG).
           Output:
             - This method does not return anything.
        """

        assert num_hash_funct > 1, num_hash_funct
        assert num_hash_col > 1, num_hash_col
        assert rand_mode in ['PNG', 'SHA'], rand_mode

        self.num_hash_funct = num_hash_funct
        self.num_hash_col = num_hash_col
        self.rand_mode = rand_mode
        self.ngram_size = ngram_size
        self.range_p = 2 * ((2 ** num_hash_funct) - 1)
        if seed is not None:
            np.random.seed(seed=seed)
        self.hash_separators = np.random.randint(0, 2 ** 16, size=num_hash_funct).astype(str)
        self.salt = ''.join(random.choice(string.ascii_letters) for i in range(32)) if secret is None else secret
        self.verbose = verbose
        self.workers = os.cpu_count() if workers == -1 else workers

    # ---------------------------------------------------------------------------

    def __encode(self, data):
        """Apply column-based vector hashing on the given input q-gram set and
           generate a hash value set which is returned.

           Input arguments:
             - q_gram_set  The set of q-grams (strings) to be encoded.

           Output:
             - hash_set  A set of hash values representing the q-gram set.
        """

        concat_lower = "".join(data).replace(" ", "").lower()
        # Split each string in the data into a list of qgrams to process
        q_gram_set = [concat_lower[i:i + self.ngram_size] for i in range(len(concat_lower) - self.ngram_size + 1)]

        hash_bitarray = np.zeros((self.num_hash_funct, self.num_hash_col), np.uint8)

        for q in q_gram_set:
            for hash_ind, sep in enumerate(self.hash_separators):
                hash_str = q + sep
                hash_bitarray[hash_ind, int(sha256(hash_str.encode()).hexdigest(), 16) % self.num_hash_col] = 1

        hash_set = list()

        for col_index in range(self.num_hash_col):
            if sum(hash_bitarray[:, col_index]) == 0:
                continue
            hash_str = "".join(list(hash_bitarray[:, col_index].astype(str))) + str(col_index) + self.salt
            if self.rand_mode == "PNG":
                # Compute ranges for integers generated by PRNG. See P. 144 of "Secure and Accurate
                # Two-Step Hash Encoding for Privacy-Preserving Record Linkage"
                # for explanation:  https://doi.org/10.1007/978-3-030-47436-2_11
                rand_min = (col_index - 1) * self.range_p
                rand_max = col_index * self.range_p - 1

                # rand_min = col_index * (2 ** self.num_hash_funct)
                # rand_max = rand_min + (2 ** self.num_hash_funct)

                random.seed(hash_str)
                hash_val = random.randint(rand_min, rand_max)
            else:
                hash_val = int(sha256(hash_str.encode()).hexdigest(), 16)
            hash_set.append(hash_val)

        return set(hash_set)

    def encode(self, data):
        if type(data[0]) == list:
            return [self.__encode(d) for d in data]
        else:
            return self.__encode(data)

    def encode_and_compare(self, data, uids, metric, sim=True):
        available_metrics = ["jaccard", "dice"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        numex = len(uids)
        uids = np.array(uids, dtype=np.float32)

        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(delayed(self.__encode)(i) for i in data)
        cache = {}
        for i, enc in enumerate(output_generator):
            cache[uids[i]] = enc
        del output_generator, data
        gc.collect()

        output_generator = parallel(delayed(make_inds)(i, numex) for i in np.array_split(np.arange(numex),
                                                                                         self.workers))
        inds = np.vstack(output_generator)
        numinds = len(inds)
        inds = np.array_split(inds, self.workers)
        gc.collect()

        pw_metrics = parallel(delayed(compute_metrics)(i, cache, uids, metric, sim) for i in inds)
        pw_metrics = np.concatenate(pw_metrics, axis=None)
        del cache
        gc.collect()
        re = np.zeros((numinds, 3), dtype=np.float32)

        re[:, 2] = pw_metrics
        del pw_metrics
        gc.collect()

        start = 0
        for i, ind in enumerate(inds):
            end = start + len(ind)
            ind[:, 0] = uids[ind[:, 0]]
            ind[:, 1] = uids[ind[:, 1]]
            re[start:end, 0:2] = ind
            start += len(ind)
        del inds, uids
        gc.collect()
        # ...and add the metrics
        return re

    def encode_records(self, data, uids):

        data = ["".join(d).replace(" ", "").lower() for d in data]
        # Split each string in the data into a list of qgrams to process
        data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]

        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(delayed(self.encode)(i) for i in data)
        cache = {}
        for i, enc in enumerate(output_generator):
            cache[uids[i]] = enc
        del output_generator, data
        gc.collect()

        return cache
