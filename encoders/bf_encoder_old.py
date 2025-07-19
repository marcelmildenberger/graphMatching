# Encodes a given using bloom filters for PPRL
import gc
import os
import pickle
from typing import Sequence, AnyStr, List, Tuple, Any
from .encoder import Encoder
import numpy as np
import hashlib
import math
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison
from sklearn.metrics import pairwise_distances_chunked
from joblib import Parallel, delayed

def numpy_pairwise_combinations(x):
    # https://carlostgameiro.medium.com/fast-pairwise-combinations-in-numpy-c29b977c33e2
    tmp = np.triu_indices(len(x), k=1)
    gc.collect()
    idx = np.stack(tmp, axis=-1)
    return x[idx]


def calc_metrics(uids, enc, metric, sim, inds):
    bf_length = enc.shape[1]

    left_matr = enc[inds[:, 0]]
    right_matr = enc[inds[:, 1]]

    del enc
    # Compute number of matching bits and overall bits
    and_sum = np.sum(np.logical_and(left_matr, right_matr), axis=1)
    # Adjust for length of BF
    and_sum = np.log(1 - (and_sum / bf_length))

    if metric == "jaccard":
        or_sum = np.sum(np.logical_or(left_matr, right_matr), axis=1)
        or_sum = np.log(1 - (or_sum / bf_length))
        pw_metrics = and_sum / or_sum
    else:
        hamming_wt_right = np.sum(right_matr, axis=1)
        hamming_wt_left = np.sum(left_matr, axis=1)
        hamming_wt_right = np.log(1 - (hamming_wt_right / bf_length))
        hamming_wt_left = np.log(1 - (hamming_wt_left / bf_length))
        pw_metrics = 2 * and_sum / (hamming_wt_left + hamming_wt_right)

    if not sim:
        pw_metrics = 1 - pw_metrics

    metrics_with_uids = np.zeros((len(inds), 3), dtype=np.float32)
    metrics_with_uids[:, 0] = uids[inds[:, 0]]
    metrics_with_uids[:, 1] = uids[inds[:, 1]]
    metrics_with_uids[:, 2] = pw_metrics
    return metrics_with_uids
class BFEncoder(Encoder):

    def __init__(self, secret: AnyStr, filter_size: int, bits_per_feature: Union[int, Sequence[int]],
                 ngram_size: Union[int, Sequence[int]], diffusion=False, eld_length = None,
                 t = None, workers=-1):
        """
        Constuctor for the BFEncoder class.
        :param secret: Secret to be used in the HMAC
        :param filter_size: Bloom Filter Size
        :param bits_per_feature: Bits to be set per feature (=Number of Hash functions). If an integer is passed, the
        same value is used for all attributes. If a list of integers is passed, one value per attribute must be
        specified.
        :param ngram_size: Size of the ngrams. If an integer is passed, the same value is used for all attributes. If a
        list of integers is passed, one value per attribute must be specified.
        :param diffusion: Specifies whether diffusion should be applied to the Bloom Filter
        See paper of Armknecht, Heng and Schnell for details: https://doi.org/10.56553/POPETS-2023-0054
        :param eld_length: Length of Bloom Filter after diffusion
        :param t: Number of bits to be xor-ed for positions in Bloom Filter.
        """
        self.secret = secret
        self.filter_size = filter_size
        self.bits_per_feature = bits_per_feature
        self.ngram_size = ngram_size
        self.diffusion = diffusion
        self.eld_length = eld_length
        self.t = t
        self.indices = None
        self.workers = os.cpu_count() if workers == -1 else workers

        if diffusion:
            assert eld_length is not None, "ELD length must be specified if diffusion is enabled"
            assert t is not None, "Number of XORed bits must be specified if diffusion is enabled"
            assert self.t <= self.filter_size, "Cannot select more bits for XORing than are present in the BF!"

            if type(self.secret) == str:
                random_seed = int(hashlib.md5(self.secret.encode()).hexdigest(), 16) % (2 ** 32 - 1)
            else:
                random_seed = self.secret
            np.random.seed(random_seed)
            self.indices = []
            available_indices = np.arange(self.filter_size)
            # Generate t random random indices per bit in the diffused BF. Bits at this position of the BF are XORed
            # to set the bit in the diffused BF. Refer to Algorithm 1 in the paper
            for j in range(self.eld_length):
                if available_indices.shape[0] >= self.t:
                    tmp = np.random.choice(available_indices, size=self.t, replace=False)
                    available_indices = np.setdiff1d(available_indices, tmp)
                else:
                    tmp = available_indices
                    available_indices = np.arange(self.filter_size)
                    tt = np.random.choice(np.setdiff1d(available_indices, tmp), size=self.t-tmp.shape[0], replace=False)
                    available_indices = np.setdiff1d(available_indices, tmp)
                    tmp = np.union1d(tmp, tt)

                self.indices.append(tmp)



    def __create_schema(self, data: Sequence[Sequence[Union[str, int]]]):
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
                                                 self.bits_per_feature if type(self.bits_per_feature) == int else
                                                 self.bits_per_feature[i]
                                             ))))
            else:
                fields.append(IntegerSpec(str(i), FieldHashingProperties(comparator=NgramComparison(
                    self.ngram_size if type(self.ngram_size) == int else self.ngram_size[i]),
                    strategy=BitsPerTokenStrategy(self.bits_per_feature if type(self.bits_per_feature) == int else
                                                    self.bits_per_feature[i]))))
            i += 1

        self.schema = Schema(fields, self.filter_size)

    def encode(self, data: Sequence[Sequence[Union[str, int]]]) -> np.ndarray:
        """
        Encodes the given data using bloom filter encoding (CLKHash), returns a MxN array of bits, where M is the number
        of records and N is the size of the bloom filter specified in schema.
        :param data: Data to encode. A list of lists: Inner list represents records with integers or strings as values.
        :return: a MxN array of bits, where M is the number of records (length of data) and N is the size of the bloom
        filter.
        """
        if not type(self.bits_per_feature) == int:
            assert len(self.bits_per_feature) == len(data[0]), "Invalid number (" + str(len(self.ngram_size)) + ") of "\
                "values for bits_per_feature. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        if not type(self.ngram_size) == int:
            assert len(self.ngram_size) == len(data[0]), "Invalid number (" + str(len(self.ngram_size)) + ") of " \
                "values for ngram_size. Must either be one value or one value per attribute (" + str(
                len(data[0])) + ")."

        # print("DEB: Schema")
        self.__create_schema(data)
        # print("DEB: CLKs")
        enc_data = clk.generate_clks(data, self.schema, self.secret)  # Returns a list of bitarrays
        # Convert the bitarrays into lists of bits, then stack them into a numpy array. Cannot stack directly, because
        # numpy would then pack the bits (https://numpy.org/doc/stable/reference/generated/numpy.packbits.html)
        # print("DEB: Stacking")
        enc_data = np.stack([list(barr) for barr in enc_data]).astype(bool)

        if self.diffusion:
            eld = np.zeros((enc_data.shape[0], self.eld_length), dtype=bool)

            #for i in range(enc_data.shape[0]):
            #    for j in range(self.eld_length):
            #        val = enc_data[i, self.indices[j][0]]
            #        for k in self.indices[j][1:]:
            #            val ^= enc_data[i,k]
            #        eld[i,j] = val
            for i, cur_inds in enumerate(self.indices):
                eld[:, i] = np.logical_xor.reduce(enc_data[:, cur_inds], axis=1)

            enc_data = eld

        return enc_data

    def encode_and_compare_and_append(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str],
                           metric: str, sim: bool = True, store_encs: bool = False) -> Tuple[np.ndarray, Sequence[Sequence[str]]]:
        """
        Encodes the given data using bloom filter encoding (CLKHash), then computes and returns the pairwise
        similarities/distances of the bloom filters as a list of tuples.
        :param data: Data to encode. A list of lists: Inner list represents records with integers or strings as values.
        :param uids: The uids of the records in the same order as the records in data
        :param metric: Similarity/Distance metric to compute. Any of the ones supported by scipy's pdist.
        :param sim: Choose if similarities (True) or distances (False) should be returned.
        :return: The similarities/distances as a list of tuples: [(i,j,val),...], where i and j are the indices of
        the records in data and val is the computed similarity/distance.
        """
        available_metrics = ["dice", "jaccard"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        #print("DEB: Encoding")
        data_joined = [["".join(d).lower()] for d in data]
        enc = self.encode(data_joined)
        enc_as_int = enc.astype(int)
        enc_as_string = [''.join(map(str, bits)) for bits in enc_as_int]
        combined_data = np.column_stack((data, enc_as_string, uids))

        if store_encs:
            cache = dict(zip(uids, enc))
            with open("./graphMatching/data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

        uids = np.array(uids)
        # Compute all possible unique combinations of indices and split them to as many sub-lists as workers
        ind_combinations = np.array_split(numpy_pairwise_combinations(np.arange(enc.shape[0])), self.workers)

        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(
            delayed(calc_metrics)(uids, enc, metric, sim, inds) for inds in ind_combinations)

        return np.vstack(output_generator), combined_data

    def get_encoding_dict(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str]):

        #print("DEB: Encoding")
        data = [["".join(d).lower()] for d in data]
        enc = self.encode(data)

        return dict(zip(uids, enc))