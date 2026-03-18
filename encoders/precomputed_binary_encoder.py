import pickle
from typing import List, Sequence, Tuple

import numpy as np

from .encoder import Encoder


BYTE_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, np.newaxis],
    axis=1,
).sum(axis=1).astype(np.uint8)


def pack_rows(bools: np.ndarray) -> np.ndarray:
    return np.packbits(bools, axis=1, bitorder="little")


def calc_binary_metric_fast(enc, uids, metric: str, sim: bool, chunk_size: int = 1024) -> np.ndarray:
    n_records = enc.shape[0]
    num_combs = (n_records * (n_records - 1)) // 2
    result = np.empty((num_combs, 3), dtype=np.float32)

    weights = BYTE_POPCOUNT[enc].sum(axis=1, dtype=np.uint32)
    cursor = 0

    for left_idx in range(n_records - 1):
        left_row = enc[left_idx]
        left_weight = weights[left_idx]

        for start in range(left_idx + 1, n_records, chunk_size):
            end = min(start + chunk_size, n_records)
            right_rows = enc[start:end]
            intersections = BYTE_POPCOUNT[np.bitwise_and(left_row, right_rows)].sum(axis=1, dtype=np.uint32)

            if metric == "dice":
                denominators = left_weight + weights[start:end]
                scores = np.divide(
                    2.0 * intersections,
                    denominators,
                    out=np.ones(end - start, dtype=np.float32),
                    where=denominators != 0,
                )
            else:
                unions = left_weight + weights[start:end] - intersections
                scores = np.divide(
                    intersections,
                    unions,
                    out=np.ones(end - start, dtype=np.float32),
                    where=unions != 0,
                )

            if not sim:
                scores = 1.0 - scores

            count = end - start
            result[cursor:cursor + count, 0] = uids[left_idx]
            result[cursor:cursor + count, 1] = uids[start:end]
            result[cursor:cursor + count, 2] = scores.astype(np.float32, copy=False)
            cursor += count

    return result


def bitstrings_to_bool_matrix(bitstrings: Sequence[str]) -> np.ndarray:
    if not bitstrings:
        return np.empty((0, 0), dtype=bool)

    bit_length = len(bitstrings[0])
    if bit_length == 0:
        raise ValueError("Precomputed binary encodings must not be empty.")

    for bitstring in bitstrings:
        if len(bitstring) != bit_length:
            raise ValueError("All precomputed binary encodings must have the same length.")
        if set(bitstring) - {"0", "1"}:
            raise ValueError("Precomputed binary encodings must contain only '0' and '1'.")

    raw = np.frombuffer("".join(bitstrings).encode("ascii"), dtype=np.uint8)
    return (raw.reshape(len(bitstrings), bit_length) == ord("1"))


class PrecomputedBinaryEncoder(Encoder):
    def __init__(self, encoding_column: int = -1):
        self.encoding_column = encoding_column

    def encode_and_compare_and_append(
        self,
        data: Sequence[Sequence[str]],
        uids: List[str],
        metric: str,
        sim: bool = True,
        store_encs: bool = False,
    ) -> Tuple[np.ndarray, Sequence[Sequence[str]]]:
        available_metrics = ["dice", "jaccard", "heng"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        metric = "dice" if metric == "heng" else metric

        bitstrings = [str(row[self.encoding_column]).strip() for row in data]
        enc = bitstrings_to_bool_matrix(bitstrings)
        combined_data = np.column_stack((data, uids))

        if store_encs:
            cache = dict(zip(uids, enc))
            with open("./data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

        if enc.shape[0] < 2:
            return np.empty((0, 3), dtype=np.float32), combined_data

        packed = pack_rows(enc)
        uid_array = np.array(uids, dtype=np.float64)
        pairwise_metrics = calc_binary_metric_fast(packed, uid_array, metric, sim)
        return pairwise_metrics, combined_data
