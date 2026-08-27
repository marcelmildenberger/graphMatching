"""Small GMA adapter for the canonical round-based LRE."""

from __future__ import annotations

import gc
import os
import pickle
from typing import Any

import numpy as np

from lre.config import LREConfig
from record_encoder import RoundBasedEncoder

from .encoder import Encoder, validate_metric


class RoundBasedLREEncoder(RoundBasedEncoder, Encoder):
    def __init__(
        self,
        *,
        key: str | int,
        config: LREConfig,
        workers: int = -1,
    ) -> None:
        resolved_workers = (os.cpu_count() or 1) if workers == -1 else max(1, int(workers))
        super().__init__(key=key, config=config, dataset_workers=resolved_workers)
        self.workers = resolved_workers

    def _pairwise_metric_values(
        self, encodings: np.ndarray, metric: str, similarity: bool
    ) -> np.ndarray:
        values = encodings.astype(np.int32, copy=False)
        weights = values.sum(axis=1, dtype=np.int32)
        common = values @ values.T
        denominator = weights[:, None] + weights[None, :]
        hamming_distance = denominator - 2 * common
        if metric == "dice":
            output = np.ones_like(common, dtype=np.float32)
            nonzero = denominator > 0
            output[nonzero] = 2.0 * common[nonzero] / denominator[nonzero]
            return output if similarity else 1.0 - output
        if metric == "hamming_distance":
            output = hamming_distance.astype(np.float32, copy=False)
            return 1.0 - output / float(self.num_bits) if similarity else output
        if metric == "hamming_similarity":
            output = 1.0 - hamming_distance.astype(np.float32) / float(self.num_bits)
            return output if similarity else 1.0 - output
        raise ValueError(f"unsupported LRE graph metric: {metric}")

    def _record_indices(self, record: Any) -> np.ndarray:
        if isinstance(record, str):
            return self.string_to_bigram_indices(record)
        joined = "".join(
            "".join(
                character
                for character in str(raw_field).lower()
                if character in self.alphabet
            )
            for raw_field in record
        )
        return self.string_to_bigram_indices(joined)

    def encode_and_compare(
        self,
        data,
        uids,
        metric,
        sim=True,
        store_encs=False,
        precomputed_encs=None,
    ):
        validate_metric(
            metric,
            ("dice", "hamming_distance", "hamming_similarity"),
            label="metric",
        )
        count = len(uids)
        if count < 2:
            return np.zeros((0, 3), dtype=np.float32)
        numeric_uids = np.asarray(uids, dtype=np.float64)
        if precomputed_encs is None:
            index_sets = [self._record_indices(record) for record in data]
            encodings = self.encode_dataset(index_sets).astype(np.uint8, copy=False)
        else:
            encodings = np.asarray(precomputed_encs, dtype=np.uint8)
            if encodings.shape != (count, self.num_bits):
                raise ValueError(
                    f"precomputed_encs has shape {encodings.shape}, "
                    f"expected ({count}, {self.num_bits})"
                )
        if store_encs:
            os.makedirs("./graphMatching/data/encodings", exist_ok=True)
            by_uid = {str(uid): encodings[index] for index, uid in enumerate(uids)}
            with open("./graphMatching/data/encodings/encoding_dict.pck", "wb") as handle:
                pickle.dump(by_uid, handle, pickle.HIGHEST_PROTOCOL)
        metric_values = self._pairwise_metric_values(encodings, metric, sim)
        upper = np.triu_indices(count, k=1)
        result = np.empty((upper[0].size, 3), dtype=np.float32)
        result[:, 0] = numeric_uids[upper[0]]
        result[:, 1] = numeric_uids[upper[1]]
        result[:, 2] = metric_values[upper]
        del metric_values, encodings
        gc.collect()
        return result
