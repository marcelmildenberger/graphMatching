import gc
import os
import pickle
from typing import Dict, List, Union
from .encoder import Encoder
from record_encoder import BigramRecordEncoder as BaseBigramRecordEncoder
import numpy as np
from joblib import Parallel, delayed

def make_inds(i_vals: np.ndarray, numex: int) -> np.ndarray:
    tmp1: List[np.ndarray] = []
    for i in i_vals:
        tmp2 = []
        for j in range(i + 1, numex):
            tmp2.append(np.array([i, j], dtype=int))
        if len(tmp2) > 0:
            tmp1.append(np.vstack(tmp2))
    return np.vstack(tmp1) if len(tmp1) > 0 else np.ndarray(shape=(0, 2), dtype=int)


def compute_metrics(
    encoder: BaseBigramRecordEncoder,
    inds: np.ndarray,
    encs: np.ndarray,
    metric: str,
    sim: bool,
) -> np.ndarray:
    """Compute pairwise metrics for the given index pairs using BaseBigramRecordEncoder.bit_vector_metrics.

    Supported metrics:
      - "dice": Dice similarity (or distance if sim=False)
      - "hamming_distance": Hamming distance (# differing bits)
      - "hamming_similarity": Hamming similarity (d - hamming_distance)

    Note: For Hamming metrics, `sim` is ignored because the metric name already determines the interpretation.
    """
    tmp = np.zeros(len(inds), dtype=np.float32)
    pos = 0

    prev_i = prev_j = None
    v_i = v_j = None

    for i, j in inds:
        if i != prev_i:
            v_i = encs[i]
            prev_i = i
        if j != prev_j:
            v_j = encs[j]
            prev_j = j

        m = encoder.bit_vector_metrics(v_i, v_j)
        val = float(m[metric])

        # Only Dice supports similarity vs distance toggle here.
        if metric == "dice" and not sim:
            val = 1.0 - val

        tmp[pos] = val
        pos += 1

    return tmp


class BigramRecordEncoder(BaseBigramRecordEncoder, Encoder):
    def __init__(
        self,
        key: Union[str, int],
        avg_record_bigrams: float,
        t: int | None = None,
        sbox_bits: int = 4,
        num_rounds: int = 1,
        rng_bits: int = 32,
        target_hw_fraction: float = 0.5,
        t_max_cap: int = 512,
        t_end: int = 2,
        xor_whitening: bool = False,
    ):
        super().__init__(key=key, avg_record_bigrams=avg_record_bigrams, t=t, sbox_bits=sbox_bits, num_rounds=num_rounds, rng_bits=rng_bits, target_hw_fraction=target_hw_fraction, t_max_cap=t_max_cap, t_end=t_end, xor_whitening=xor_whitening)
        self.workers = os.cpu_count() or 1
        
    def encode_and_compare(self, data, uids, metric, sim=True, store_encs=False):
        # Supported metrics. (We intentionally drop Jaccard here.)
        available_metrics = ["dice", "hamming_distance", "hamming_similarity"]
        assert metric in available_metrics, "Invalid metric. Must be one of " + str(available_metrics)

        numex = len(uids)
        uids = np.array(uids, dtype=np.float32)

        normalized = []
        for record in data:
            if isinstance(record, str):
                normalized.append(record)
            else:
                normalized.append("".join(map(str, record)))

        enc_list = [self.encode(rec) for rec in normalized]
        encs = np.stack(enc_list).astype(np.uint8)

        if store_encs:
            os.makedirs("./graphMatching/data/encodings", exist_ok=True)
            tmpdict = {str(int(uid)): encs[i] for i, uid in enumerate(uids)}
            with open("./graphMatching/data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(tmpdict, f, pickle.HIGHEST_PROTOCOL)
            del tmpdict

        parallel = Parallel(n_jobs=self.workers, prefer="threads")
        output_generator = parallel(delayed(make_inds)(i, numex) for i in np.array_split(np.arange(numex), self.workers * 4))
        inds = np.vstack(output_generator)
        numinds = len(inds)
        inds_split = np.array_split(inds, self.workers)
        pw_metrics = parallel(delayed(compute_metrics)(self, ind, encs, metric, sim) for ind in inds_split)
        pw_metrics = np.concatenate(pw_metrics, axis=None)
        re = np.zeros((numinds, 3), dtype=np.float32)
        re[:, 2] = pw_metrics

        start = 0
        for ind in inds_split:
            end = start + len(ind)
            ind[:, 0] = uids[ind[:, 0]]
            ind[:, 1] = uids[ind[:, 1]]
            re[start:end, 0:2] = ind
            start = end

        del inds_split, inds, pw_metrics, enc_list, encs
        gc.collect()
        print(re)
        return re