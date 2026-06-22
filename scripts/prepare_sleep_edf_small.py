from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

def count_homogeneous_windows_by_label(
    labels: np.ndarray, window_size: int, step_size: int
) -> dict[str, int]:
    """Count homogeneous-label windows per label."""
    counts: dict[str, int] = {}
    n = len(labels)
    for start in range(0, n - window_size + 1, step_size):
        window_labels = labels[start : start + window_size]
        unique = set(window_labels)
        if len(unique) == 1:
            lbl = next(iter(unique))
            counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def _parse_int_list(raw: str) -> list[int]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return [int(v) for v in values]


def _normalize_stage_token(description: str) -> str:
    text = description.strip().upper()
    text = text.replace("SLEEP STAGE", "").strip()
    text = text.replace("STAGE", "").strip()
    return text


def _map_stage_to_regime(description: str, mapping: str) -> str | None:
    token = _normalize_stage_token(description)

    wake_tokens = {"W", "WAKE"}
    rem_tokens = {"R", "REM"}
    nrem_tokens = {"1", "2", "3", "4", "N1", "N2", "N3", "N4"}

    if mapping == "wake_sleep":
        if token in wake_tokens:
            return "state_wake"
        if token in rem_tokens or token in nrem_tokens:
            return "state_sleep"
        return None

    if mapping == "rem_nrem":
        if token in rem_tokens:
            return "state_rem"
        if token in nrem_tokens:
            return "state_nrem"
        return None

    if mapping == "per_stage":
        if token in wake_tokens:
            return "stage_W"
        if token in rem_tokens:
            return "stage_REM"
        if token in {"1", "N1"}:
            return "stage_N1"
        if token in {"2", "N2"}:
            return "stage_N2"
        if token in {"3", "4", "N3", "N4"}:
            return "stage_N3"
        return None

    raise ValueError(f"Unsupported mapping: {mapping}")


def _fetch_sleep_edf_pairs(
    subjects: list[int],
    recordings: list[int],
    mne_data_dir: str | None,
) -> list[tuple[Path, Path]]:
    from mne.datasets.sleep_physionet import age

    fetch_data = age.fetch_data
    sig = inspect.signature(fetch_data)
    kwargs: dict[str, object] = {}

    if "subjects" in sig.parameters:
        kwargs["subjects"] = subjects

    if "recording" in sig.parameters:
        kwargs["recording"] = recordings
    elif "recordings" in sig.parameters:
        kwargs["recordings"] = recordings

    if mne_data_dir is not None and "path" in sig.parameters:
        kwargs["path"] = mne_data_dir

    if "on_missing" in sig.parameters:
        kwargs["on_missing"] = "warn"

    try:
        raw_pairs = fetch_data(**kwargs)
    except TypeError:
        if "recording" in kwargs:
            kwargs["recordings"] = kwargs.pop("recording")
        raw_pairs = fetch_data(**kwargs)

    pairs: list[tuple[Path, Path]] = []
    for item in raw_pairs:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise RuntimeError(f"Unexpected Sleep-EDF fetch_data output item: {item!r}")
        pairs.append((Path(item[0]), Path(item[1])))

    if not pairs:
        raise RuntimeError("No Sleep-EDF PSG/hypnogram pairs were downloaded")
    return pairs


def _extract_labeled_series(
    psg_path: Path,
    hypnogram_path: Path,
    channel: str,
    resample_hz: float | None,
    mapping: str,
) -> tuple[np.ndarray, np.ndarray]:
    import mne

    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose="ERROR")
    if channel not in raw.ch_names:
        raise ValueError(f"Channel '{channel}' not found in PSG file {psg_path.name}")

    raw.pick([channel])
    if resample_hz is not None:
        raw.resample(float(resample_hz), verbose="ERROR")

    annotations = mne.read_annotations(str(hypnogram_path))
    raw.set_annotations(annotations, emit_warning=False)

    values = raw.get_data()[0].astype(float)
    n_samples = int(values.shape[0])
    sfreq = float(raw.info["sfreq"])

    labels = np.full(n_samples, "", dtype=object)
    for ann in raw.annotations:
        regime = _map_stage_to_regime(str(ann["description"]), mapping)
        if regime is None:
            continue

        onset = float(ann["onset"])
        duration = float(ann["duration"])
        start = max(0, int(np.floor(onset * sfreq)))
        stop = min(n_samples, int(np.ceil((onset + duration) * sfreq)))
        if stop > start:
            labels[start:stop] = regime

    keep = labels != ""
    kept_values = values[keep]
    kept_labels = labels[keep].astype(str)
    if kept_values.size == 0:
        raise RuntimeError(f"No labeled samples remained after mapping in {psg_path.name}")
    if len(np.unique(kept_labels)) < 2:
        raise RuntimeError(
            f"Need at least two mapped regimes in {psg_path.name}; got {sorted(np.unique(kept_labels).tolist())}"
        )
    return kept_values, kept_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact univariate Sleep-EDF CSV for symbolic structural profiling."
    )
    parser.add_argument("--subjects", default="0", help="Comma-separated subject ids (default: 0)")
    parser.add_argument("--recordings", default="1", help="Comma-separated recording ids (default: 1)")
    parser.add_argument("--channel", default="EEG Fpz-Cz")
    parser.add_argument("--label-mapping", choices=["wake_sleep", "rem_nrem", "per_stage"], default="wake_sleep")
    parser.add_argument("--resample-hz", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--min-label-windows", type=int, default=20)
    parser.add_argument(
        "--fail-if-below-threshold",
        action="store_true",
        help="Exit non-zero when precheck min label windows is below threshold.",
    )
    parser.add_argument("--mne-data-dir", default=None, help="Optional MNE data directory")
    parser.add_argument("--output", default="data/external/sleep_edf_wake_sleep_small.csv")
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Also write one CSV per record_id into this directory "
        "(what run_sleep_edf.sh reads, e.g. outputs/sleep_edf/subjects).",
    )
    args = parser.parse_args()

    try:
        import mne  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'mne'. Install with: .venv/bin/pip install mne"
        ) from exc

    subjects = _parse_int_list(args.subjects)
    recordings = _parse_int_list(args.recordings)
    pairs = _fetch_sleep_edf_pairs(subjects, recordings, args.mne_data_dir)

    frames: list[pd.DataFrame] = []
    aggregated_window_counts: dict[str, int] = {}

    for record_id, (psg_path, hyp_path) in enumerate(pairs):
        values, labels = _extract_labeled_series(
            psg_path=psg_path,
            hypnogram_path=hyp_path,
            channel=args.channel,
            resample_hz=args.resample_hz,
            mapping=args.label_mapping,
        )

        frame = pd.DataFrame(
            {
                "V1": values,
                "regime": labels,
                "record_id": record_id,
                "psg_file": psg_path.name,
            }
        )
        frames.append(frame)

        record_counts = count_homogeneous_windows_by_label(
            labels=labels,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        for label, count in record_counts.items():
            aggregated_window_counts[label] = aggregated_window_counts.get(label, 0) + int(count)

        print(
            f"Record {record_id}: {psg_path.name} / {hyp_path.name} "
            f"rows={len(frame)} labels={frame['regime'].value_counts().to_dict()}"
        )

    dataset = pd.concat(frames, ignore_index=True)
    dataset["t"] = np.arange(len(dataset))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    if args.split_dir:
        split_dir = Path(args.split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for rid, group in dataset.groupby("record_id"):
            group.to_csv(split_dir / f"subject_{rid}.csv", index=False)
        print(f"Split into {dataset['record_id'].nunique()} per-subject CSVs in {split_dir}")

    min_label_windows = min(aggregated_window_counts.values()) if aggregated_window_counts else 0
    print(f"Wrote {len(dataset)} rows to {output_path}")
    print(f"Columns: {', '.join(dataset.columns)}")
    print(f"Label counts: {dataset['regime'].value_counts().to_dict()}")
    print(
        "Precheck windows by label "
        f"(w={args.window_size}, s={args.step_size}): {aggregated_window_counts}"
    )
    print(
        f"Precheck min_label_windows={min_label_windows} "
        f"(threshold={args.min_label_windows})"
    )

    if args.fail_if_below_threshold and min_label_windows < args.min_label_windows:
        raise SystemExit(
            f"Precheck failed: min_label_windows={min_label_windows} < {args.min_label_windows}"
        )


if __name__ == "__main__":
    main()
