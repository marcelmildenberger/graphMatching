import sympy
import os
import scipy
import numpy as np
import galois
import pickle
import string
from tqdm import tqdm
from .encoder import Encoder
from numpy.random import Generator, PCG64
from primality import primality
from joblib import Parallel, delayed


def get_rand_safeprime(lower, upper):
    # Generate a random prime greater than or equal to the size of the universe of ngrams (using 10x universe size as  upper bound)
    p = sympy.randprime(lower, upper)
    while not primality.isprime((p - 1) // 2):
        p = sympy.randprime(lower, upper)
    return p


def safe_is_primitive_element(u, q, GF):
    prime_factors = np.array([2, (q - 1) // 2])
    prime_factors = np.mod((q - 1) // prime_factors, q)
    return not np.any(np.power(GF(u), prime_factors) == 1)


def ngramize(data, n):
    # Split the training data into n-grams.
    data = ["".join(d).replace(" ", "").lower() for d in data]  # Remove whitespace and normalize to lowercase
    return [[b[i:i + n] for i in range(len(b) - n + 1)] for b in data]


def to_hankel(data, m_coeff, l, expos, q, to_gf=True):
    # data = ngramize(data, 2)
    GF = galois.GF(q)
    if to_gf:
        data = GF(data)
        m_coeff = GF(m_coeff)
        expos = GF(expos)
        coefficients = np.multiply(data, m_coeff)
    encodings = []
    for record in coefficients:
        hankel_first_col = np.multiply(np.tile(record, (2 * l + 1, 1)), expos[0:2 * l + 1, ])
        hankel_last_row = np.multiply(np.tile(record, (2 * l + 1, 1)), expos[2 * l:4 * l + 1, ])
        # Sanity check
        assert hankel_first_col.shape == hankel_last_row.shape
        hankel_first_col = np.sum(hankel_first_col, axis=1)
        hankel_last_row = np.sum(hankel_last_row, axis=1)
        assert hankel_first_col.shape == hankel_last_row.shape
        hankel_enc = GF(scipy.linalg.hankel(hankel_first_col, hankel_last_row))
        encodings.append(hankel_enc)

    return encodings


def make_inds(i_vals, numex):
    tmp1 = []
    for i in i_vals:
        tmp2 = []
        for j in range(i + 1, numex):
            tmp2.append(np.array([i, j], dtype=int))
        if len(tmp2) > 0:
            tmp1.append(np.vstack(tmp2))
    return np.vstack(tmp1) if len(tmp1) > 0 else np.ndarray(shape=(0, 2), dtype=int)


def compute_metrics(inds, cache, uids, sim):
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

        val = int(np.linalg.det(i_enc - j_enc)) == 0
        if not sim:
            val = 1 - val
        tmp[pos] = np.array([uids[i], uids[j], val])
        pos += 1
    return tmp


class PSTEncoder(Encoder):

    def __init__(self, k, l, p=None, charset=string.printable, verbose=False, workers=-1):
        self.k = k
        self.l = l
        self.verbose = verbose
        self.workers = os.cpu_count() - 1 if workers == -1 else workers

        # Create the ngram Universe (All possible bigrams of allowed characters)
        ngram_universe = []
        for s1 in charset:
            for s2 in charset:
                ngram_universe.append(s1 + s2)

        # Create dictionary for indices
        self.ngram_ind = dict(zip(ngram_universe, range(len(ngram_universe))))

        if p is None:
            self.p = sympy.randprime(37 * 10 ** 7, 37 * 10 ** 8)
            if verbose:
                print("Chose prime p: %i" % self.p)
        else:
            self.p = p

        # Generate a random prime q for coefficient mapping
        q_lower = ((2 * (l ** 2)) + l) * (self.p - 1) * 2 ** (k + 1)
        assert q_lower < (2**63)-1, "Lower bound of prime q is out of range for 64 bit integers!"
        self.q = get_rand_safeprime(q_lower + 1, (2**63)-1)
        if verbose:
            print("Chose prime q: %i" % self.q)

        # Set up the Finite Field
        self.GF_q = galois.GF(self.q)

        # Create two mappings mapping each possible ngram to a coefficient and an exponent.
        # Note that the m_exp has to be injective, which is ensured by checking for duplicates
        rng = Generator(PCG64())
        if verbose:
            print("Generate mappings")

        m_exp = rng.integers(1, p, len(ngram_universe))
        unique_exp_inds = np.unique(m_exp, return_index=True)[1]
        while unique_exp_inds.shape[0] < m_exp.shape[0]:
            duplicate_map = np.isin(np.arange(m_exp.shape[0]), unique_exp_inds, invert=True)
            m_exp[duplicate_map] = rng.integers(0, self.q, np.sum(duplicate_map))
            unique_exp_inds = np.unique(m_exp, return_index=True)[1]

        self.m_coeff = self.GF_q(rng.integers(1, self.q, len(ngram_universe)))

        if verbose:
            print("Searching for primitive element of finite field.")

        self.u = rng.integers(1, self.q, 1)
        while not safe_is_primitive_element(self.u, self.q, self.GF_q):
            self.u = rng.integers(1, self.q, 1)

        if verbose:
            print("Chose primitive element: %i" % self.u)
            print("Pre-computing inputs and exponents for polynomials")

        self.poly_inputs = np.power(self.GF_q(self.u), np.arange(4 * l + 1))
        self.expos = np.power(self.poly_inputs.reshape(-1, 1), np.tile(m_exp, (self.poly_inputs.shape[0], 1)))

        if verbose:
            print("Done.")

    def _to_onehot(self, data):
        data = ngramize(data, 2)
        ngram_hotenc = np.zeros((len(data), len(self.ngram_ind)), dtype=np.uint8)

        # Insert the data into the prepared matrices.
        i = 0
        # Iterate over all records in the data
        for ngr in data:
            # Iterate over the record's n-grams and set the corresponding cell (row defined by record, column defined by index of unique n-gram) to 1.
            for n in ngr:
                ngram_hotenc[i, self.ngram_ind[n]] = 1
            i += 1
        return ngram_hotenc

    def encode(self, data):
        data = ngramize(data, 2)
        data = self.GF_q(self._to_onehot(data))
        data_chunks = np.array_split(np.arange(data.shape[0]), self.workers)
        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(
            delayed(to_hankel)(np.array(data[c]), np.array(self.m_coeff), self.l, np.array(self.expos), self.q) for c in
            data_chunks)
        enc_chunks = list(output_generator)
        return [enc for encs in enc_chunks for enc in encs]

    def encode_and_compare(self, data, uids, metric, sim=True, store_encs=False):
        available_metrics = ["heng"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        data = ngramize(data, 2)
        data = self.GF_q(self._to_onehot(data))
        uids = [float(u) for u in uids]
        data_chunks = np.array_split(np.arange(data.shape[0]), self.workers)
        parallel = Parallel(n_jobs=self.workers)
        if self.verbose:
            print("Encoding data into Hankel matrices")
        output_generator = parallel(
            delayed(to_hankel)(np.array(data[c]), np.array(self.m_coeff), self.l, np.array(self.expos), self.q) for c in
            data_chunks)
        cache = {}
        i = 0
        for enc_chunks in tqdm(output_generator, desc="Encoding", disable=not self.verbose, total=len(data_chunks)):
            for enc in enc_chunks:
                cache[uids[i]] = enc
                i += 1
        del output_generator
        if store_encs:
            tmpdict = dict()

            for key, val in cache.items():
                tmpdict[str(int(key))] = val
            with open("./data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(tmpdict, f, pickle.HIGHEST_PROTOCOL)
            del tmpdict

        numex = len(uids)
        output_generator = parallel(
            delayed(make_inds)(i, numex) for i in np.array_split(np.arange(numex), self.workers))
        inds = np.vstack(output_generator)
        inds = np.array_split(inds, self.workers)
        pw_metrics = parallel(delayed(compute_metrics)(i, cache, uids, sim) for i in inds)
        del cache
        if self.verbose:
            print("Computing similarities")
        return np.vstack(pw_metrics)