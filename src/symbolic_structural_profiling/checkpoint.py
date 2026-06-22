from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import EnsembleConfig, FeatureConfig, WindowConfig


CHECKPOINT_MANIFEST_NAME = "_manifest.json"
CHECKPOINT_VERSION = 1

# Knobs that affect runtime/scheduling but not per-window output.
# Resuming with different values for these is allowed.
_EXCLUDED_ENSEMBLE_KEYS = frozenset(
    {
        "pysr_parallelism",
        "pysr_procs",
        "pysr_deterministic",
        "pysr_progress",
        "pysr_verbosity",
    }
)


class CheckpointMismatchError(RuntimeError):
    """Existing checkpoint manifest disagrees with the active run."""


def _normalize(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    return value


def make_config_signature(
    window: WindowConfig,
    ensemble: EnsembleConfig,
    features: FeatureConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cfg, name, skip in [
        (window, "window", frozenset()),
        (ensemble, "ensemble", _EXCLUDED_ENSEMBLE_KEYS),
        (features, "features", frozenset()),
    ]:
        d = asdict(cfg)
        out[name] = {k: _normalize(v) for k, v in sorted(d.items()) if k not in skip}
    return out


def make_data_fingerprint(values: np.ndarray, labels, record_ids) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(values, dtype=float)).tobytes())
    h.update(np.asarray(labels).astype(str).tobytes())
    if record_ids is not None:
        h.update(np.asarray(record_ids).astype(str).tobytes())
    return h.hexdigest()[:16]


def _hash_dict(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class WindowCheckpointStore:
    """Disk-backed per-window checkpoint store.

    Layout:
        <root>/
            _manifest.json
            <phase>_<window_index:06d>.pkl

    Atomic writes via tmp-and-rename. Manifest pins data fingerprint and
    science-relevant config; mismatch raises CheckpointMismatchError so we
    never silently merge incompatible windows.
    """

    def __init__(
        self,
        root: Path,
        config_signature: dict[str, Any],
        data_fingerprint: str,
        n_windows_expected: int,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_signature = config_signature
        self.config_hash = _hash_dict(config_signature)
        self.data_fingerprint = data_fingerprint
        self.n_windows_expected = int(n_windows_expected)
        self._validate_or_write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.root / CHECKPOINT_MANIFEST_NAME

    def _validate_or_write_manifest(self) -> None:
        manifest = {
            "version": CHECKPOINT_VERSION,
            "config_hash": self.config_hash,
            "data_fingerprint": self.data_fingerprint,
            "n_windows_expected": self.n_windows_expected,
            "config_signature": self.config_signature,
        }
        if not self.manifest_path.exists():
            self._write_atomic(
                self.manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True).encode(),
            )
            return

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for key in ("version", "config_hash", "data_fingerprint", "n_windows_expected"):
            if existing.get(key) != manifest[key]:
                raise CheckpointMismatchError(
                    f"Checkpoint manifest mismatch on '{key}': "
                    f"existing={existing.get(key)!r}, current={manifest[key]!r}. "
                    f"Delete {self.root} or use a fresh checkpoint directory."
                )

    def _path(self, phase: str, window_index: int) -> Path:
        return self.root / f"{phase}_{window_index:06d}.pkl"

    def _write_atomic(self, path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def get(self, phase: str, window_index: int) -> Optional[Any]:
        path = self._path(phase, window_index)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def put(self, phase: str, window_index: int, payload: Any) -> None:
        path = self._path(phase, window_index)
        self._write_atomic(
            path,
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def count(self, phase: str) -> int:
        return sum(1 for _ in self.root.glob(f"{phase}_*.pkl"))
