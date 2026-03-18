# Encodes a given using bloom filters for PPRL
import gc
import os
import pickle
import hashlib
from numbers import Integral
from typing import AnyStr, List, Sequence, Tuple, Union

import numpy as np
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison
import numba as nb
from scipy.special import binom
from .encoder import normalize_joined_record, normalize_record_values


def _is_scalar_int(value) -> bool:
    return isinstance(value, Integral)


def _pack_combined_data(data, enc, uids):
    enc_as_string = ["".join(map(str, bits)) for bits in enc.astype(int)]
    return np.column_stack((data, enc_as_string, uids))


def _store_encoding_cache(uids, enc):
    cache = dict(zip(uids, enc))
    with open("./data/encodings/encoding_dict.pck", "wb") as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)


def pack_rows(bools: np.ndarray) -> np.ndarray:
    """
    bool  (n, m)  ->  uint64 (n, ceil(m/64))
    """
    packed = np.packbits(bools, axis=1, bitorder='little')  # uint8
    return packed.view(np.uint64).copy()  # uint64


@nb.njit((nb.uint64)(nb.uint64), cache=True)
def popcnt64(x):
    # SWAR popcount that inlines nicely in Numba
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    return ((((x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56) & 0xFF


@nb.njit(parallel=True, fastmath=True, cache=True)
def calc_dice_fast(enc, uids, num_combs, n_threads):
    n, nWords = enc.shape
    weights = np.empty(n, dtype=np.uint64)
    for i in range(n):
        w = 0
        for w64 in enc[i]:
            w += popcnt64(w64)
        weights[i] = w

    out_i = np.zeros(num_combs, np.float64)
    out_j = np.zeros(num_combs, np.float64)
    out_sim = np.zeros(num_combs, np.float64)

    chunks = np.array_split(np.arange(n), n_threads)
    for ch in nb.prange(len(chunks)):
        for i in chunks[ch]:
            cnt = 0
            local_offset = sum([(n - 1) - x for x in range(i)])
            for j in range(i + 1, n):
                inter = 0
                for w in range(nWords):
                    inter += popcnt64(enc[i, w] & enc[j, w])
                dice = 2.0 * inter / (weights[i] + weights[j])
                out_i[local_offset + cnt] = uids[i]
                out_j[local_offset + cnt] = uids[j]
                out_sim[local_offset + cnt] = dice
                cnt += 1

    re = np.column_stack((out_i, out_j, out_sim))
    return re

class BFEncoder():

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

            if isinstance(self.secret, str):
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

    def __parameter_for_index(self, parameter, index):
        return parameter if _is_scalar_int(parameter) else parameter[index]

    def __validate_feature_configuration(self, num_columns):
        if not _is_scalar_int(self.bits_per_feature):
            assert len(self.bits_per_feature) == num_columns, (
                "Invalid number (" + str(len(self.bits_per_feature)) + ") of values for bits_per_feature. "
                "Must either be one value or one value per attribute (" + str(num_columns) + ")."
            )

        if not _is_scalar_int(self.ngram_size):
            assert len(self.ngram_size) == num_columns, (
                "Invalid number (" + str(len(self.ngram_size)) + ") of values for ngram_size. "
                "Must either be one value or one value per attribute (" + str(num_columns) + ")."
            )

    def __encode_joined_records(self, data):
        return self.encode([[normalize_joined_record(row)] for row in data])

    def __compute_pairwise_dice(self, enc, uids):
        uids_array = np.asarray(uids, dtype=np.float64)
        return calc_dice_fast(pack_rows(enc), uids_array, int(binom(enc.shape[0], 2)), self.workers)

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
            if isinstance(feature, str):
                fields.append(StringSpec(str(i),
                                         FieldHashingProperties(comparator=NgramComparison(
                                             self.__parameter_for_index(self.ngram_size, i)),
                                             strategy=BitsPerTokenStrategy(
                                                 self.__parameter_for_index(self.bits_per_feature, i)
                                             ))))
            else:
                fields.append(IntegerSpec(str(i), FieldHashingProperties(comparator=NgramComparison(
                    self.__parameter_for_index(self.ngram_size, i)),
                    strategy=BitsPerTokenStrategy(self.__parameter_for_index(self.bits_per_feature, i)))))
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
        self.__validate_feature_configuration(len(data[0]))

        normalized_data = [normalize_record_values(row) for row in data]

        # print("DEB: Schema")
        self.__create_schema(normalized_data)
        # print("DEB: CLKs")
        enc_data = clk.generate_clks(normalized_data, self.schema, self.secret)  # Returns a list of bitarrays
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

    def encode_and_compare(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str],
                           metric: str, sim: bool = True, store_encs: bool = False) -> np.ndarray:
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
        available_metrics = ["dice", "jaccard", "heng"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        enc = self.__encode_joined_records(data)

        if store_encs:
            _store_encoding_cache(uids, enc)

        pw_dice = self.__compute_pairwise_dice(enc, uids)
        return enc, pw_dice

    def encode_and_compare_and_append(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str],
                           metric: str, sim: bool = True, store_encs: bool = False) -> Tuple[np.ndarray, Sequence[Sequence[str]]]:
        available_metrics = ["dice", "jaccard", "heng"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        enc = self.__encode_joined_records(data)
        combined_data = _pack_combined_data(data, enc, uids)

        if store_encs:
            _store_encoding_cache(uids, enc)

        pw_dice = self.__compute_pairwise_dice(enc, uids)
        return pw_dice, combined_data

    def get_encoding_dict(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str]):
        enc = self.__encode_joined_records(data)
        return dict(zip(uids, enc))
