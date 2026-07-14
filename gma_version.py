"""Versioned invariants for the empirical graph-matching harness."""

from __future__ import annotations

from functools import lru_cache
from hashlib import md5, sha256
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Tuple


GMA_HARNESS_VERSION = "independent_eve_key_v1"


@lru_cache(maxsize=64)
def _file_content_identity(path_text: str, mtime_ns: int, ctime_ns: int, size: int) -> str:
    del mtime_ns, ctime_ns  # Included in the cache key to invalidate changed files.
    digest = sha256()
    with Path(path_text).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"{path_text}:{size}:{digest.hexdigest()}"


def _input_identity(value: object) -> str:
    if value in (None, ""):
        return "none"
    path = Path(str(value)).expanduser().resolve()
    try:
        stat = path.stat()
    except OSError:
        return f"{path}:missing"
    return _file_content_identity(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def attack_cache_hashes(
    global_config: Mapping[str, object],
    enc_config: Mapping[str, object],
    emb_config: Mapping[str, object],
    *,
    cache_version: Optional[str] = GMA_HARNESS_VERSION,
) -> Tuple[str, str, str, str]:
    """Return the graph cache hashes, optionally salted by a harness version."""

    data_path = str(global_config["Data"])
    drop_from = global_config.get("DropFrom")
    overlap = global_config.get("Overlap")
    enc_key = str(enc_config) if cache_version is None else f"{cache_version}:{enc_config}"
    emb_key = str(emb_config) if cache_version is None else f"{cache_version}:{emb_config}"
    input_key = data_path
    if cache_version is not None:
        input_key = (
            f"data={_input_identity(global_config['Data'])}:"
            f"preencoded={_input_identity(global_config.get('PreencodedTsv'))}"
        )

    def digest(value: str) -> str:
        return md5(value.encode()).hexdigest()

    if drop_from == "Alice":
        return (
            digest(f"{enc_key}-{input_key}-DropAlice"),
            digest(f"{enc_key}-{input_key}-{overlap}-DropAlice"),
            digest(f"{emb_key}-{enc_key}-{input_key}-DropAlice"),
            digest(f"{emb_key}-{enc_key}-{input_key}-{overlap}-DropAlice"),
        )
    if drop_from == "Eve":
        return (
            digest(f"{enc_key}-{input_key}-{overlap}-DropEve"),
            digest(f"{enc_key}-{input_key}-DropEve"),
            digest(f"{emb_key}-{enc_key}-{input_key}-{overlap}-DropEve"),
            digest(f"{emb_key}-{enc_key}-{input_key}-DropEve"),
        )
    return (
        digest(f"{enc_key}-{input_key}-{overlap}-DropBoth"),
        digest(f"{enc_key}-{input_key}-{overlap}-DropBoth"),
        digest(f"{emb_key}-{enc_key}-{input_key}-{overlap}-DropBoth"),
        digest(f"{emb_key}-{enc_key}-{input_key}-{overlap}-DropBoth"),
    )


def eve_precomputed_vectors(
    _uids: Iterable[object],
    _alice_lookup: Callable[[Iterable[object]], object],
) -> None:
    """Keep Eve independent from Alice's optional pre-encoded vector table.

    The arguments make the forbidden Alice lookup explicit at the call site.
    They are intentionally not evaluated: Eve's records must flow through the
    encoder instantiated with ``EveSecret``.
    """

    return None
