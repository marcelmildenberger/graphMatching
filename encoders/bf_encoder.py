# Encodes a given using bllom filters for PPRL

import io
from typing import Sequence, AnyStr, Union
from encoder import Encoder
import itertools
import pandas as pd
import numpy as np
import anonlink
import clkhash
from clkhash import clk
from clkhash.field_formats import *
from clkhash.schema import Schema
from clkhash.comparators import NgramComparison
from clkhash.serialization import serialize_bitarray


class BFEncoder(Encoder):

    def __init__(self, secret: AnyStr, filter_size: int, bits_per_feature: Union[int, Sequence[int]],
                 ngram_size: Union[int, Sequence[int]]):
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
                                             self.bits_per_feature if type(self.bits_per_feature) == int else self.bits_per_feature[i]
                                                                ))))
            else:
                fields.append(IntegerSpec(str(i), FieldHashingProperties(comparator=NgramComparison(2),
                                                                         strategy=BitsPerFeatureStrategy(30))))
            i += 1

        self.schema = Schema(fields, self.filter_size)

    def encode(self, data: Sequence[Sequence[Union[str, int]]]) -> np.ndarray:
        self.__create_schema()
        return clk.generate_clks(data, self.schema, self.secret)

    def encode_and_compare(self, data: Sequence[Sequence[str]]) -> np.ndarray:
        enc2 = clk.generate_clks(data, schema, self.secret)


def drop_redundancy(data):
    data = data[data.Node1 != data.Node2].copy()

    data["ID"] = data.apply(lambda x: str(min(x['Node1'], x['Node2'])) + "-" + str(max(x['Node1'], x['Node2'])), axis=1)
    data.drop_duplicates(subset="ID", inplace=True)
    data.drop(columns=["ID"], inplace=True)
    return data


def encode_and_compare(schema, data, secret="secret"):
    if "node_id" not in data.columns:
        node_ids = np.arange(len(data))
    else:
        node_ids = data["node_id"]
        data = data.drop(columns=["node_id"])

    tmp_csv = io.StringIO()
    data.to_csv(tmp_csv)
    tmp_csv.seek(0)
    hashed_data = clk.generate_clk_from_csv(tmp_csv, secret, schema)
    similarity, dataset_i, record_i = anonlink.candidate_generation.find_candidate_pairs(
        [hashed_data, hashed_data],
        anonlink.similarities.dice_coefficient, 0)
    pairw_sims_enc = [(node_ids[record_i[0][i]], node_ids[record_i[1][i]], similarity[i]) for i in
                      range(len(similarity))]
    return pairw_sims_enc
