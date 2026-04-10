import gc
import os
import pickle
from typing import Union

import numpy as np

from .encoder import Encoder, validate_metric
from record_encoder import (
    BigramRecordEncoder as BaseBigramRecordEncoder,
    DEFAULT_INPUT_CODEWORD_WEIGHT,
)


class BigramRecordEncoder(BaseBigramRecordEncoder, Encoder):
    def __init__(
        self,
        key: Union[str, int],
        round_structure: str = "D1S2",
        rng_bits: int = 32,
        input_encoding: str = "one_hot_encoding",
        input_codeword_weight: int | None = DEFAULT_INPUT_CODEWORD_WEIGHT,
        workers: int = -1,
    ):
        resolved_workers = (os.cpu_count() or 1) if workers == -1 else max(1, int(workers))
        super().__init__(
            key=key,
            round_structure=round_structure,
            rng_bits=rng_bits,
            input_encoding=input_encoding,
            input_codeword_weight=input_codeword_weight,
            dataset_workers=resolved_workers,
        )
        self.workers = resolved_workers

    def _pairwise_metric_values(self, encs: np.ndarray, metric: str, sim: bool) -> np.ndarray:
        encs_i32 = encs.astype(np.int32, copy=False)
        weights = encs_i32.sum(axis=1, dtype=np.int32)
        common_ones = encs_i32 @ encs_i32.T
        denom = weights[:, None] + weights[None, :]
        hamming_distance = denom - (2 * common_ones)

        if metric == "dice":
            values = np.ones_like(common_ones, dtype=np.float32)
            nonzero = denom > 0
            values[nonzero] = (2.0 * common_ones[nonzero]) / denom[nonzero]
            if not sim:
                values = 1.0 - values
        elif metric == "hamming_distance":
            values = hamming_distance.astype(np.float32, copy=False)
        elif metric == "hamming_similarity":
            values = (self.num_bits - hamming_distance).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return values

    def encode_and_compare(self, data, uids, metric, sim=True, store_encs=False, precomputed_encs=None):
        available_metrics = ("dice", "hamming_distance", "hamming_similarity")
        validate_metric(metric, available_metrics, label="metric")

        numex = len(uids)
        if numex < 2:
            return np.zeros((0, 3), dtype=np.float32)

        uids = np.asarray(uids, dtype=np.float64)

        if precomputed_encs is not None:
            encs = np.asarray(precomputed_encs, dtype=np.uint8)
            if encs.shape != (numex, self.num_bits):
                raise ValueError(
                    f"precomputed_encs has shape {encs.shape}, expected ({numex}, {self.num_bits})"
                )
        else:
            normalized = []
            for record in data:
                if isinstance(record, str):
                    normalized.append(record.lower())
                else:
                    normalized.append("".join(map(str, record)).lower())
            encs = self.encode_dataset(normalized).astype(np.uint8, copy=False)

        if store_encs:
            os.makedirs("./graphMatching/data/encodings", exist_ok=True)
            tmpdict = {str(uid): encs[i] for i, uid in enumerate(uids)}
            with open("./graphMatching/data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(tmpdict, f, pickle.HIGHEST_PROTOCOL)
            del tmpdict

        metric_values = self._pairwise_metric_values(encs, metric, sim)
        tri_upper = np.triu_indices(numex, k=1)
        numinds = tri_upper[0].shape[0]

        result = np.empty((numinds, 3), dtype=np.float32)
        result[:, 0] = uids[tri_upper[0]]
        result[:, 1] = uids[tri_upper[1]]
        result[:, 2] = metric_values[tri_upper]

        del metric_values, encs
        gc.collect()

        return result
