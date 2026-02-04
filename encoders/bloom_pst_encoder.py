import sympy
import os
import scipy
import numpy as np
import galois
import pickle
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison
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

def get_multiplicative_inverse(a, fieldsize):
    d, s, t = galois.egcd(a, fieldsize)
    return s % fieldsize


def compute_rank(A, q, normalize=True):
    m = A.shape[0]
    Ai = np.array(A.copy(), dtype='object')
    L = np.identity(m)
    # global inv_lookup
    for i in range(0, m - 1):
        if Ai[i, i] == 0:
            idxs = np.nonzero(Ai[i:, i])[0]  # The first non-zero entry in column `i` below row `i`
            if idxs.size == 0:
                L[i, i] = 1
                continue
            else:
                raise ValueError("The LU decomposition of 'A' does not exist. Use the PLU decomposition instead.")
        # if int(Ai[i, i]) not in inv_lookup:
        #    inv_lookup[int(int(Ai[i, i]))] = int(np.reciprocal(GF(Ai[i, i])))
        inv = get_multiplicative_inverse(Ai[i, i], q)
        l = np.mod(Ai[i + 1:, i] * inv, q, dtype='object')
        Ai[i + 1:, :] = np.mod(Ai[i + 1:, :] - np.mod(np.multiply.outer(l, Ai[i, :]), q), q)
        L[i + 1:, i] = l
    # U = Ai

    if normalize:
        return 1 - (np.sum(np.sum(Ai, axis=1) != 0) / m)
    else:
        return np.sum(np.sum(Ai, axis=1) != 0)


def compute_metrics(inds, cache, uids, sim, q):
    tmp = np.zeros((len(inds), 3), dtype=np.float32)
    pos = 0
    # GF = galois.GF(q)
    prev_i = prev_j = None
    for i, j in inds:
        if i != prev_i:
            i_enc = cache[uids[i]]
            prev_i = i
        if j != prev_j:
            j_enc = cache[uids[j]]
            prev_j = j

        val = compute_rank(i_enc - j_enc, q, normalize=True)
        # val = int(np.linalg.det(i_enc - j_enc)==0)
        if not sim:
            val = 1 - val
        tmp[pos] = np.array([uids[i], uids[j], val])
        pos += 1
    return tmp

class BloomPSTEncoder(Encoder):

    def __init__(self, k, l, bf_n_hash_func, bf_size, ngram_size, p=None, q=None, secret=None, verbose=False, workers=-1):
        self.k = k
        self.l = l
        self.n_hash_func = bf_n_hash_func
        self.ngram_size = ngram_size
        self.verbose = verbose
        self.workers = os.cpu_count() - 1 if workers == -1 else workers
        self.secret = secret
        self.bf_size = bf_size

        if p is None:
            self.p = sympy.randprime(37 * 10 ** 7, 37 * 10 ** 8)
            if verbose:
                print("Chose prime p: %i" % self.p)
        else:
            self.p = p

        # Generate a random prime q for coefficient mapping
        q_lower = ((2 * (l ** 2)) + l) * (self.p - 1) * 2 ** (k + 1)
        if q is None:
            assert q_lower < (2**63)-1, "Lower bound of prime q is out of range for 64 bit integers!"
            self.q = get_rand_safeprime(q_lower + 1, (2**63)-1)
            if verbose:
                print("Chose prime q: %i" % self.q)
        else:
            assert q > q_lower, ("Q is too small. Must be at least %i" % q_lower)
            self.q = q

        # Set up the Finite Field
        self.GF_q = galois.GF(self.q)

        # Create two mappings mapping each possible ngram to a coefficient and an exponent.
        # Note that the m_exp has to be injective, which is ensured by checking for duplicates
        rng = Generator(PCG64())
        if verbose:
            print("Generate mappings")

        m_exp = rng.integers(1, p, self.bf_size)
        unique_exp_inds = np.unique(m_exp, return_index=True)[1]
        while unique_exp_inds.shape[0] < m_exp.shape[0]:
            duplicate_map = np.isin(np.arange(m_exp.shape[0]), unique_exp_inds, invert=True)
            m_exp[duplicate_map] = rng.integers(0, self.q, np.sum(duplicate_map))
            unique_exp_inds = np.unique(m_exp, return_index=True)[1]

        self.m_coeff = self.GF_q(rng.integers(1, self.q, self.bf_size))

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

    def __create_schema(self, data):
        """
        Creates a linking schema for the CLKhash library based on the parameters specified during creation of the
        Encoder.
        :param data: The data to encode
        :return: Nothing.
        """
        fields = []
        i = 0
        for feature in data[0]:
            # Set StringSpec for string features and IntegerSpec for int features. Note: Right now,
            # only String and Integer features are allowed. Also, the data type at a specific index must be the same
            # across all records.
            if type(feature) == str:
                fields.append(StringSpec(str(i),
                                         FieldHashingProperties(comparator=NgramComparison(
                                             self.ngram_size if type(self.ngram_size) == int else self.ngram_size[i]),
                                             strategy=BitsPerTokenStrategy(
                                                 self.n_hash_func if type(self.n_hash_func) == int else
                                                 self.n_hash_func[i]
                                             ))))
            else:
                fields.append(IntegerSpec(str(i), FieldHashingProperties(comparator=NgramComparison(
                    self.ngram_size if type(self.ngram_size) == int else self.ngram_size[i]),
                    strategy=BitsPerTokenStrategy(self.n_hash_func if type(self.n_hash_func) == int else
                                                    self.n_hash_func[i]))))
            i += 1

        self.schema = Schema(fields, self.bf_size)

    def encode(self, data):
        data = [["".join(d).lower()] for d in data]

        if not type(self.n_hash_func) == int:
            assert len(self.n_hash_func) == len(data[0]), "Invalid number (" + str(len(self.n_hash_func)) + ") of "\
                "values for bits_per_feature. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        if not type(self.ngram_size) == int:
            assert len(self.ngram_size) == len(data[0]), "Invalid number (" + str(len(self.ngram_size)) + ") of " \
                "values for ngram_size. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        self.__create_schema(data)
        bloom_filters = clk.generate_clks(data, self.schema, self.secret)  # Returns a list of bitarrays
        bloom_filters = np.stack([list(barr) for barr in bloom_filters]).astype(int)

        data_chunks = np.array_split(np.arange(bloom_filters.shape[0]), self.workers)
        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(
            delayed(to_hankel)(np.array(bloom_filters[c]), np.array(self.m_coeff), self.l, np.array(self.expos), self.q) for c in
            data_chunks)
        enc_chunks = list(output_generator)
        return [np.array(enc) for encs in enc_chunks for enc in encs]

    def encode_and_compare(self, data, uids, metric, sim=True, store_encs=False):
        available_metrics = ["heng"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        if not type(self.n_hash_func) == int:
            assert len(self.n_hash_func) == len(data[0]), "Invalid number (" + str(len(self.n_hash_func)) + ") of "\
                "values for bits_per_feature. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        if not type(self.ngram_size) == int:
            assert len(self.ngram_size) == len(data[0]), "Invalid number (" + str(len(self.ngram_size)) + ") of " \
                "values for ngram_size. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        data = [["".join(d).lower()] for d in data]

        self.__create_schema(data)
        bloom_filters = clk.generate_clks(data, self.schema, self.secret)  # Returns a list of bitarrays
        bloom_filters = np.stack([list(barr) for barr in bloom_filters]).astype(int)

        uids = [float(u) for u in uids]
        data_chunks = np.array_split(np.arange(bloom_filters.shape[0]), self.workers)
        parallel = Parallel(n_jobs=self.workers)
        if self.verbose:
            print("Encoding data into Hankel matrices")
        output_generator = parallel(
            delayed(to_hankel)(bloom_filters[c], np.array(self.m_coeff), self.l, np.array(self.expos), self.q) for c in
            data_chunks)
        cache = {}
        i = 0
        for enc_chunks in tqdm(output_generator, desc="Encoding", disable=not self.verbose, total=len(data_chunks)):
            for enc in enc_chunks:
                cache[uids[i]] = np.array(enc)
                i += 1
        del output_generator
        if store_encs:
            tmpdict = dict()
            for key, val in cache.items():
                tmpdict[str(int(key))] = val
            with open("./graphMatching/data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(tmpdict, f, pickle.HIGHEST_PROTOCOL)
            del tmpdict

        numex = len(uids)
        output_generator = parallel(
            delayed(make_inds)(i, numex) for i in np.array_split(np.arange(numex), self.workers))
        inds = np.vstack(output_generator)
        inds = np.array_split(inds, self.workers)
        pw_metrics = parallel(delayed(compute_metrics)(i, cache, uids, sim, self.q) for i in inds)
        del cache
        if self.verbose:
            print("Computing similarities")
        return np.vstack(pw_metrics)