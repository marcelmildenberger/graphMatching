import pickle
from ast import literal_eval
from typing import List, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from .encoder import Encoder
from .pairwise_set_metrics import assemble_pairwise_results, compute_set_metrics, make_inds


def parse_serialized_set(value) -> set[int]:
    if isinstance(value, set):
        return {int(item) for item in value}
    serialized = str(value).strip()
    if serialized == "{}":
        return set()
    parsed = literal_eval(serialized)
    if isinstance(parsed, set):
        return {int(item) for item in parsed}
    if isinstance(parsed, (list, tuple)):
        return {int(item) for item in parsed}
    raise ValueError(f"Unsupported serialized set value: {value!r}")


class PrecomputedSetEncoder(Encoder):
    def __init__(self, workers: int = -1, encoding_column: int = -1):
        self.workers = workers
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

        numex = len(uids)
        uids_as_float = np.array(uids, dtype=np.float32)
        parallel = Parallel(n_jobs=self.workers)
        output_generator = parallel(delayed(parse_serialized_set)(row[self.encoding_column]) for row in data)

        cache = {}
        for idx, enc in enumerate(output_generator):
            cache[uids_as_float[idx]] = enc
        combined_data = np.column_stack((data, uids))

        if store_encs:
            tmpdict = {str(int(key)): val for key, val in cache.items()}
            with open("./data/encodings/encoding_dict.pck", "wb") as f:
                pickle.dump(tmpdict, f, pickle.HIGHEST_PROTOCOL)

        if numex < 2:
            return np.empty((0, 3), dtype=np.float32), combined_data

        output_generator = parallel(
            delayed(make_inds)(indices, numex) for indices in np.array_split(np.arange(numex), max(self.workers, 1) * 4)
        )
        inds = np.vstack(output_generator)
        inds = np.array_split(inds, max(self.workers, 1))

        pw_metrics = parallel(delayed(compute_set_metrics)(chunk, cache, uids_as_float, metric, sim) for chunk in inds)
        pw_metrics = np.concatenate(pw_metrics, axis=None)
        return assemble_pairwise_results(inds, uids_as_float, pw_metrics), combined_data
