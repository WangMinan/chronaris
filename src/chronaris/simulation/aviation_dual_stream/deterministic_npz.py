"""Byte-reproducible compressed NumPy archive writer."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Mapping

import numpy as np


def write_deterministic_npz(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Write sorted .npy members with fixed ZIP metadata and return SHA-256."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for name, array in sorted(arrays.items()):
                member = io.BytesIO()
                np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        temporary.replace(resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(resolved)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
