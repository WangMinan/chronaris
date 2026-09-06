"""Reuse immutable native recording tables and their content fingerprints."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def read_native_table(path, columns):
    """Return a shared read-only frame; callers copy only the selected window."""
    path = Path(path)
    stat = path.stat()
    return _read_table(str(path.resolve()), stat.st_size, stat.st_mtime_ns, tuple(columns))


# ponytail: bounded by 16 recordings, not bytes; use a byte budget if measured RSS requires it.
@lru_cache(maxsize=16)
def _read_table(path, size, modified_ns, columns):
    return pd.read_csv(path, usecols=columns)


def native_file_sha256(path):
    path = Path(path)
    stat = path.stat()
    return _file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=4096)
def _file_digest(path, size, modified_ns):
    return sha256_file(path)


def select_native_subjects(paths, *, subject_limit, subject_ids):
    if subject_limit is not None and subject_limit <= 0:
        raise ValueError("native subject limit must be positive or None")
    subjects = sorted(path for path in paths if path.is_dir())
    if subject_ids is None:
        return subjects[:subject_limit]
    wanted = set(subject_ids)
    if len(wanted) != len(subject_ids) or not wanted:
        raise ValueError("native subject selection must be nonempty and unique")
    missing = wanted - {path.name for path in subjects}
    if missing:
        raise ValueError(f"fixed native subject list is missing: {sorted(missing)}")
    return [path for path in subjects if path.name in wanted]
