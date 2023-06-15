# Encodes a given using bloom filters for PPRL

from typing import Sequence, AnyStr, List, Tuple, Any
from .encoder import Encoder
import numpy as np
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison

from scipy.spatial.distance import pdist
from .utils import calc_condensed_index


class BFEncoder(Encoder):

    def __init__(self, secret: AnyStr, filter_size: int, bits_per_feature: Union[int, Sequence[int]],
                 ngram_size: Union[int, Sequence[int]]):
        """
        Constuctor for the BFEncoder class.
        :param secret: Secret to be used in the HMAC
        :param filter_size: Bloom Filter Size
        :param bits_per_feature: Bits to be set per feature (=Number of Hash functions). If an integer is passed, the
        same value is used for all attributes. If a list of integers is passed, one value per attribute must be
        specified.
        :param ngram_size: Size of the ngrams. If an integer is passed, the same value is used for all attributes. If a
        list of integers is passed, one value per attribute must be specified.
        """

        self.secret = secret
        self.filter_size = filter_size
        self.bits_per_feature = bits_per_feature
        self.ngram_size = ngram_size

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
                                             strategy=BitsPerFeatureStrategy(
                                                 self.bits_per_feature if type(self.bits_per_feature) == int else
                                                 self.bits_per_feature[i]
                                             ))))
            else:
                fields.append(IntegerSpec(str(i), FieldHashingProperties(comparator=NgramComparison(2),
                                                                         strategy=BitsPerFeatureStrategy(30))))
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
            assert len(self.bits_per_feature) == len(data[0]), "Invalid number ("+ str(len(self.ngram_size)) + ") of " \
            "values for bits_per_feature. Must either be one value or one value per attribute ("+ str(len(data[0])) + ")."


        if not type(self.ngram_size) == int:
            assert len(self.ngram_size) == len(data[0]), "Invalid number ("+ str(len(self.ngram_size)) + ") of "\
            "values for ngram_size. Must either be one value or one value per attribute ("+ str(len(data[0])) + ")."

        self.__create_schema()
        enc_data = clk.generate_clks(data, self.schema, self.secret)  # Returns a list of bitarrays
        # Convert the bitarrays into lists of bits, then stack them into a numpy array. Cannot stack directly, because
        # numpy would then pack the bits (https://numpy.org/doc/stable/reference/generated/numpy.packbits.html)
        enc_data = np.stack([list(barr) for barr in enc_data])
        return enc_data

    def encode_and_compare(self, data: Sequence[Sequence[Union[str, int]]], metric: str, sim: bool = True) -> List[
        Tuple[int, int, float]]:
        """
        Encodes the given data using bloom filter encoding (CLKHash), then computes and returns the pairwise
        similarities/distances of the bloom filters as a list of tuples.
        :param data: Data to encode. A list of lists: Inner list represents records with integers or strings as values.
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
        enc = self.encode(data)
        # Calculates pairwise distances between the provided data points
        pw_metrics = pdist(enc, metric=metric)
        # Convert to similarities if specified
        if sim:
            pw_metrics = [1 - p for p in pw_metric]

        # Convert to "long" format
        pw_metrics_long = []
        for i in range(len(enc)):
            for j in range(i + 1, len(enc)):
                pw_metrics_long.append((i, j, pw_metrics[calc_condensed_index(i, j, len(enc))]))

        return pw_metrics_long
