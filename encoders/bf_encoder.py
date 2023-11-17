# Encodes a given using bloom filters for PPRL
import gc
from typing import Sequence, AnyStr, List, Tuple, Any
from .encoder import Encoder
import numpy as np
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison
from sklearn.metrics import pairwise_distances_chunked

def numpy_pairwise_combinations(x):
    # https://carlostgameiro.medium.com/fast-pairwise-combinations-in-numpy-c29b977c33e2
    tmp = np.triu_indices(len(x), k=1)
    gc.collect()
    idx = np.stack(tmp, axis=-1)
    return x[idx]

def test(uids):
    dim = ((len(uids)*len(uids))-len(uids))//2
    tmp = np.triu_indices(len(uids), k=1)
    idx = np.zeros((dim, 2), dtype=np.float32)
    idx[:,0] = uids[tmp[0]]
    idx[:,1] = uids[tmp[1]]
    return idx

class BFEncoder(Encoder):

    def __init__(self, secret: AnyStr, filter_size: int, bits_per_feature: Union[int, Sequence[int]],
                 ngram_size: Union[int, Sequence[int]], diffusion=False, eld_length = None,
                 t = None):
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

        if diffusion:
            assert eld_length is not None, "ELD length must be specified if diffusion is enabled"
            assert t is not None, "Number of XORed bits must be specified if diffusion is enabled"
            assert self.t <= self.filter_size, "Cannot select more bits for XORing than are present in the BF!"
            # Generate t random random indices per bit in the diffused BF. Bits at this position of the BF are XORed
            # to set the bit in the diffused BF.
            self.indices = [np.random.choice(np.arange(self.filter_size), size=self.t, replace=False) for _ in range(self.eld_length)]


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

            for i, cur_inds in enumerate(self.indices):
                eld[:, i] = np.bitwise_xor.reduce(enc_data[:, cur_inds], axis=1)

            enc_data = eld

        return enc_data

    def encode_and_compare(self, data: Sequence[Sequence[Union[str, int]]], uids: List[str],
                           metric: str, sim: bool = True) -> np.ndarray:
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
        available_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                             "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis",
                             "matching",
                             "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                             "sqeuclidean", "yule"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        #print("DEB: Encoding")
        enc = self.encode(data)
        # Calculates pairwise distances between the provided data points
        if sim:
            vals = pairwise_distances_chunked(enc, metric=metric, n_jobs=-1)
        else:
            vals = pairwise_distances_chunked(enc, metric=metric, n_jobs=-1)

        vals = np.vstack(list(vals), dtype=np.float32)
        del enc, data
        gc.collect()
        #print("Subset")
        # Since we're comparing the values to themselves, we only need the values above the main diagonal
        # Otherwise we would have two edges per node-pair in the graph
        # lower_tri = np.tri(vals.shape[0], k=0, dtype=bool)
        vals = vals[np.invert(np.tri(vals.shape[0], k=0, dtype=bool))]
        gc.collect()
        #print("Sim")
        if sim:
            vals = 1-vals
        gc.collect()

        #print("Uids")
        # Compute the unique combinations of uids...
        uids = test(np.array(uids, dtype=int))
        gc.collect()

        #print("Re")
        #dim = ((len(uids)*len(uids))-len(uids))//2
        re = np.zeros((len(uids), 3), dtype=np.float32)

        #print("Add")
        re[:,2] = vals
        del vals
        gc.collect()

        #print("Add UIDs")
        re[:,0:2] = uids
        gc.collect()
        #...and add the metrics
        return re

