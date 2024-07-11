import os
import hashlib
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def est_1bit_jacc(arr_a, arr_b):
    assert arr_a.shape[0] == arr_b.shape[0]
    collisions = sum(arr_a == arr_b)
    return max(0, 2 * (collisions / arr_a.shape[0]) - 1)


def est_1bit_dice(arr_a, arr_b):
    jacc = est_1bit_jacc(arr_a, arr_b)
    return (2 * jacc) / (1 + jacc)


def dice_sim(arr_a, arr_b):
    num_common = len(np.intersect1d(arr_a, arr_b))
    total_unique = len(np.unique(arr_a)) + len(np.unique(arr_b))
    return (2.0 * num_common) / (total_unique)


def jacc_sim(arr_a, arr_b):
    size_intersection = len(np.intersect1d(arr_a, arr_b))
    size_union = len(np.union1d(arr_a, arr_b))
    return float(size_intersection / size_union)


def make_inds(i_vals, numex):
    tmp1 = []
    for i in i_vals:
        tmp2 = []
        for j in range(i + 1, numex):
            tmp2.append(np.array([i, j], dtype=int))
        if len(tmp2) > 0:
            tmp1.append(np.vstack(tmp2))
    return np.vstack(tmp1) if len(tmp1) > 0 else np.ndarray(shape=(0, 2), dtype=int)


def compute_metrics(inds, cache, uids, metric, sim, onebit):
    tmp = np.zeros((len(inds), 3), dtype=np.float32)
    pos = 0
    prev_i = prev_j = None
    for i, j in inds:
        if i != prev_i:
            i_enc = cache[uids[i]]
            prev_i = i
        if j != prev_j:
            j_enc = cache[uids[j]]
            prev_j = j
        if onebit:
            if metric == "jaccard":
                val = est_1bit_jacc(i_enc, j_enc)
            else:
                val = est_1bit_dice(i_enc, j_enc)
        else:
            if metric == "jaccard":
                val = jacc_sim(i_enc, j_enc)
            else:
                val = dice_sim(i_enc, j_enc)

        if not sim:
            val = 1 - val
        tmp[pos] = np.array([uids[i], uids[j], val])
        pos += 1
    return tmp


class TMHEncoder():

    def __init__(self, num_hash_func, num_hash_bits, num_sub_keys,
                 ngram_size, one_bit_hash=True, random_seed=42, verbose=False, workers=-1):

        self.num_hash_func = num_hash_func
        self.num_hash_bits = num_hash_bits
        self.num_sub_keys = num_sub_keys
        self.ngram_size = ngram_size
        self.subkey_length = 64 // num_sub_keys
        self.verbose = verbose
        self.one_bit_hash = one_bit_hash
        self.workers = os.cpu_count() if workers == -1 else workers

        assert 64 % num_sub_keys == 0
        assert num_hash_bits in [8, 16, 32, 64]

        if self.subkey_length == 16:
            self.subkey_dtype = np.uint16
        elif self.subkey_length == 32:
            self.subkey_dtype = np.uint32
        elif self.subkey_length == 64:
            self.subkey_dtype = np.uint64
        else:
            self.subkey_dtype = np.uint8

        if num_hash_bits == 8:
            self.minhash_dtype = np.uint8
        elif num_hash_bits == 16:
            self.minhash_dtype = np.uint16
        elif num_hash_bits == 32:
            self.minhash_dtype = np.uint32
        else:
            self.minhash_dtype = np.uint64

        self.hashtables = np.random.randint(2, size=(
        self.num_hash_func, self.num_sub_keys, 2 ** self.subkey_length, self.num_hash_bits), dtype=bool)
        if random_seed != None:
            if type(random_seed) == str:
                random_seed = int(hashlib.md5(random_seed.encode()).hexdigest(), 16) % (2 ** 32 - 1)
            np.random.seed(random_seed)

    def get_min_hash(self, val):
        key = bin(int(hashlib.md5(val.encode()).hexdigest(), 16))[-64:]  # Extract 64 least significant bits
        key = np.array([int(digit) for digit in key], dtype=bool)  # To numpy aray
        subkeys = np.array_split(key, self.num_sub_keys)  # Split key into c equally sized subkeys
        indices = np.packbits(subkeys, axis=-1).view(
            self.subkey_dtype)  # Convert subkeys into indices used for the hashtbales

        minhashes = np.zeros((self.num_hash_func), dtype=self.minhash_dtype)
        for func_ind in range(self.num_hash_func):
            tmp = np.zeros((self.num_sub_keys, self.num_hash_bits), dtype=bool)
            for key_ind, table_ind in enumerate(indices):
                tmp[key_ind] = self.hashtables[func_ind][key_ind][table_ind]
            minhashes[func_ind] = np.packbits(np.bitwise_xor.reduce(tmp), axis=-1).view(self.minhash_dtype)  # identify minimum hash value
        return minhashes

    def hash_qgrams(self, qgrams):
        hashvals = np.zeros((len(qgrams), self.num_hash_func), dtype=self.minhash_dtype)
        for i, q in enumerate(qgrams):
            hashvals[i] = self.get_min_hash(q)
        hashvals = np.min(hashvals, 0)
        if self.one_bit_hash:
            hashvals = hashvals % 2  # Return only least significant bit
            return hashvals.astype(bool)
        return hashvals

    def encode(self, data):
        hashes = np.zeros((len(data), self.num_hash_func), dtype=bool if self.one_bit_hash else self.minhash_dtype)
        data = ["".join(d).replace(" ", "").lower() for d in data]
        data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]
        for i, qg in enumerate(data):
            hashes[i] = self.hash_qgrams(qg)
        return hashes

    def encode_and_compare(self, data, uids, metric, sim=True, store_encs=False):
        available_metrics = ["jaccard", "dice"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        uids = [float(u) for u in uids]
        data = ["".join(d).replace(" ", "").lower() for d in data]
        # Split each string in the data into a list of qgrams to process
        data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]
        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(delayed(self.hash_qgrams)(i) for i in data)
        cache = {}
        for i, enc in tqdm(enumerate(output_generator), desc="Encoding", disable=not self.verbose, total=len(uids)):
            cache[uids[i]] = enc
        del output_generator
        if store_encs:
            tmpdict = dict()

            for key, val in cache.items():
                tmpdict[str(int(key))] = val
            with open("./data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(tmpdict, f, pickle.HIGHEST_PROTOCOL)

        numex = len(uids)
        output_generator = parallel(
            delayed(make_inds)(i, numex) for i in np.array_split(np.arange(numex), self.workers * 4))
        inds = np.vstack(output_generator)
        inds = np.array_split(inds, self.workers)
        pw_metrics = parallel(delayed(compute_metrics)(i, cache, uids, metric, sim, self.one_bit_hash) for i in inds)
        del cache
        return np.vstack(pw_metrics)
