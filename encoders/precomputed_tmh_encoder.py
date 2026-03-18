import pickle
from typing import List, Sequence, Tuple

import numpy as np

from .encoder import Encoder
from .precomputed_binary_encoder import bitstrings_to_bool_matrix


def est_1bit_jacc_batch(left_row: np.ndarray, right_rows: np.ndarray) -> np.ndarray:
    collisions = np.count_nonzero(left_row == right_rows, axis=1)
    return np.maximum(0.0, 2.0 * (collisions / left_row.shape[0]) - 1.0)


def est_1bit_dice_batch(left_row: np.ndarray, right_rows: np.ndarray) -> np.ndarray:
    jacc = est_1bit_jacc_batch(left_row, right_rows)
    return np.divide(2.0 * jacc, 1.0 + jacc, out=np.zeros_like(jacc, dtype=np.float32), where=(1.0 + jacc) != 0)


def calc_tmh_metric_fast(enc: np.ndarray, uids: np.ndarray, metric: str, sim: bool, chunk_size: int = 1024) -> np.ndarray:
    n_records = enc.shape[0]
    num_combs = (n_records * (n_records - 1)) // 2
    result = np.empty((num_combs, 3), dtype=np.float32)
    cursor = 0

    for left_idx in range(n_records - 1):
        left_row = enc[left_idx]
        for start in range(left_idx + 1, n_records, chunk_size):
            end = min(start + chunk_size, n_records)
            right_rows = enc[start:end]
            if metric == "jaccard":
                scores = est_1bit_jacc_batch(left_row, right_rows)
            else:
                scores = est_1bit_dice_batch(left_row, right_rows)

            if not sim:
                scores = 1.0 - scores

            count = end - start
            result[cursor:cursor + count, 0] = uids[left_idx]
            result[cursor:cursor + count, 1] = uids[start:end]
            result[cursor:cursor + count, 2] = scores.astype(np.float32, copy=False)
            cursor += count

    return result


class PrecomputedTMHEncoder(Encoder):
    def __init__(self, one_bit_hash: bool = True, encoding_column: int = -1):
        self.one_bit_hash = one_bit_hash
        self.encoding_column = encoding_column

    def encode_and_compare_and_append(
        self,
        data: Sequence[Sequence[str]],
        uids: List[str],
        metric: str,
        sim: bool = True,
        store_encs: bool = False,
    ) -> Tuple[np.ndarray, Sequence[Sequence[str]]]:
        available_metrics = ["dice", "jaccard"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        if not self.one_bit_hash:
            raise ValueError("Precomputed TMH support currently requires 1-bit hash outputs.")

        bitstrings = [str(row[self.encoding_column]).strip() for row in data]
        enc = bitstrings_to_bool_matrix(bitstrings)
        combined_data = np.column_stack((data, uids))

        if store_encs:
            cache = dict(zip(uids, enc))
            with open("./data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)

        if enc.shape[0] < 2:
            return np.empty((0, 3), dtype=np.float32), combined_data

        uid_array = np.array(uids, dtype=np.float64)
        pairwise_metrics = calc_tmh_metric_fast(enc, uid_array, metric, sim)
        return pairwise_metrics, combined_data
