#!/usr/bin/env python3
"""Evidence-limited NVBD construction and source-position audit (Comment #16).

This is a NEW verification utility, NOT the original dataset-generation script.
It NEVER edits the reference dataset or source JSON/ZIP files.

Modes:
  python NAGT_Comment16_Source_Offset_Audit.py --dataset dataset.zip
  python NAGT_Comment16_Source_Offset_Audit.py --dataset dataset.zip \
      --train-source path/to/Train --validation-source path/to/Validation \
      --test-source path/to/Test --outdir nvbd_offset_audit

Sources can be directories of veh_*.json, single veh_*.json files, or ZIPs
containing veh_*.json files. Unpack nested ZIPs first.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd

KEY = ["split", "source_vehicle_file", "rcvTime", "sender_id", "messageID"]
SPLITS = ("Train", "Validation", "Test")


def dataset_load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith("dataset.csv")]
            if len(names) != 1:
                raise ValueError("Expected exactly one dataset.csv inside dataset archive")
            with z.open(names[0]) as src:
                d = pd.read_csv(src, low_memory=False)
    else:
        d = pd.read_csv(path, low_memory=False)
    need = set(KEY) | {"attacker", "sender_pos_x", "sender_pos_y", "split_rho", "active_current_window", "split_attacker_fraction"}
    missing = sorted(need - set(d.columns))
    if missing:
        raise ValueError(f"Dataset missing required fields: {missing}")
    if not set(d.split.unique()).issubset(set(SPLITS)):
        raise ValueError(f"Unexpected split labels: {d.split.unique()}")
    d["source_vehicle_file"] = d.source_vehicle_file.astype(str).map(lambda v: Path(v).name)
    d["sender_id"] = d.sender_id.astype(str)
    for field in ("rcvTime", "messageID"):
        d[field] = pd.to_numeric(d[field], errors="raise").astype("int64")
    if d.duplicated(KEY).any():
        raise ValueError(f"Found {int(d.duplicated(KEY).sum())} duplicate reception keys")
    if not d.attacker.isin([0, 1]).all():
        raise ValueError("Attacker labels are not binary")
    return d


def dataset_report(d: pd.DataFrame) -> list[str]:
    lines = [f"Dataset shape: {len(d):,} reception rows; {len(d.columns)} columns",
             "Note: dataset-internal checks do not prove source-relative offsets without original clean source JSON."]
    for split in SPLITS:
        s = d[d.split == split]
        total = len(s)
        attackers = int(s.attacker.sum())
        unique_msgs = len(s[["sender_id", "messageID"]].drop_duplicates())
        lines.append(f"{split}: {total:,} rows; {unique_msgs:,} unique sender/message pairs; "
                     f"{attackers:,} attacker rows ({attackers/total:.4%}); "
                     f"{s.sender_id.nunique():,} senders")
        if "active_current_window" in s:
            n = int((s.active_current_window != s.attacker).sum())
            lines.append(f"  active_current_window differs from attacker label: {n} rows")
        if "split_rho" in s:
            mx = float(np.max(np.abs(s.split_rho.to_numpy(dtype=float) - (1 - attackers / total))))
            lines.append(f"  max |split_rho-(1-attacker fraction)|: {mx:.6g}")
    return lines


def messages_from_json_bytes(blob: bytes, label: str):
    text = blob.decode("utf-8-sig")
    try:
        x = json.loads(text)
    except json.JSONDecodeError:
        x = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(x, dict) and isinstance(x.get("messages"), list):
        x = x["messages"]
    if not isinstance(x, list):
        raise ValueError(f"{label} is not an array of vehicle messages")
    return x


def iter_source(source: Path):
    if source.is_dir():
        files = list(source.rglob("veh_*.json")) + list(source.rglob("veh_*.jsonl"))
        for f in files:
            yield f.name, f.read_bytes()
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            files = [name for name in z.namelist() if Path(name).name.startswith("veh_")
                     and name.lower().endswith((".json", ".jsonl"))]
            for name in files:
                with z.open(name) as fd:
                    yield Path(name).name, fd.read()
    elif source.name.startswith("veh_") and source.suffix.lower() in (".json", ".jsonl"):
        yield source.name, source.read_bytes()
    else:
        raise ValueError("Provide a directory, ZIP, or veh_*.json/veh_*.jsonl file for each split; "
                         f"unsupported source: {source}")


def read_source(source: Path, split: str) -> pd.DataFrame:
    records = []
    nfiles = 0
    for name, raw in iter_source(source):
        nfiles += 1
        for msg in messages_from_json_bytes(raw, name):
            if "sender" not in msg or "pos" not in msg["sender"]:
                raise ValueError(f"Malformed sender position in {name}")
            xyz = msg["sender"]["pos"]
            # The original InTAS receiver logs encode coordinates as a
            # comma-separated STRING: "216320.09,452693.46,0.0".
            # Accept numeric arrays too, without treating a string as an
            # array of characters (which would silently yield wrong values).
            if isinstance(xyz, str):
                xyz = [part.strip() for part in xyz.split(",")]
            if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
                raise ValueError(f"Missing 2D source position in {name}: {msg['sender']['pos']!r}")
            try:
                sx, sy = float(xyz[0]), float(xyz[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric sender position in {name}: {xyz!r}") from exc
            if not np.isfinite(sx) or not np.isfinite(sy):
                raise ValueError(f"Non-finite sender position in {name}: {xyz!r}")
            records.append({"split": split, "source_vehicle_file": name,
                            "rcvTime": int(msg["rcvTime"]), "sender_id": str(msg["sender_id"]),
                            "messageID": int(msg["messageID"]),
                            "source_sender_pos_x": sx,
                            "source_sender_pos_y": sy})
    if nfiles == 0:
        raise ValueError(f"No veh_*.json found in {source}. Verify you selected a split's receiver logs, "
                         "not a ground-truth summary, and extract any nested ZIPs.")
    out = pd.DataFrame.from_records(records)
    if out.duplicated(KEY).any():
        raise ValueError(f"Original {split} source has duplicate reception keys; "
                         "inspect split assignment and JSON file structure")
    return out


def audited_offsets(d: pd.DataFrame, raw: pd.DataFrame, *, atol: float) -> tuple[pd.DataFrame, list[str]]:
    x = d.merge(raw, on=KEY, how="outer", validate="one_to_one", indicator=True)
    gaps = x._merge.value_counts().to_dict()
    if gaps.get("left_only", 0) or gaps.get("right_only", 0):
        raise ValueError(f"Reception-record mismatch (unmatched): {gaps}; "
                         "check that sources are the clean matching original partitions")
    x = x.drop(columns=["_merge"])
    x["offset_x"] = x.sender_pos_x.astype(float) - x.source_sender_pos_x
    x["offset_y"] = x.sender_pos_y.astype(float) - x.source_sender_pos_y
    clean = x.attacker.eq(0)
    bad_clean = int(((x.loc[clean, "offset_x"].abs() > atol) |
                     (x.loc[clean, "offset_y"].abs() > atol)).sum())
    x_att = x[x.attacker.eq(1)].copy()
    agg = x_att.groupby(["split", "sender_id"], sort=True).agg(
        n_records=("offset_x", "size"),
        offset_x=("offset_x", "median"), offset_y=("offset_y", "median"),
        x_min=("offset_x", "min"), x_max=("offset_x", "max"),
        y_min=("offset_y", "min"), y_max=("offset_y", "max"),
    ).reset_index()
    spread = np.maximum(agg.x_max - agg.x_min, agg.y_max - agg.y_min)
    bad_constant = int((spread > 2*atol).sum())
    magnitudes = np.hypot(agg.offset_x.to_numpy(), agg.offset_y.to_numpy())
    bad_zero = int((magnitudes < 1e-7).sum())
    reports = [f"Matched {len(x):,} original reception records to NVBD by {KEY}.",
               f"Legitimate rows with modified sender x/y (tolerance={atol}): {bad_clean}",
               f"Attacker sender/split groups with NON-CONSTANT offsets: {bad_constant}",
               f"Attacker sender/split groups with zero offset: {bad_zero}",
               f"Inferred offset manifest groups: {len(agg)}",
               f"Range |offset_x|: {agg.offset_x.abs().min():.6f} to {agg.offset_x.abs().max():.6f} m",
               f"Range |offset_y|: {agg.offset_y.abs().min():.6f} to {agg.offset_y.abs().max():.6f} m",
               f"Vector offset magnitude: {magnitudes.min():.6f} to {magnitudes.max():.6f} m"]
    # Fail rather than silently writing a misleading verified offset manifest.
    if bad_clean or bad_constant or bad_zero:
        raise ValueError("Source-versus-NVBD positional audit FAILED:\n" + "\n".join(reports))
    # A transparent replay check. For each attacker message, reconstruct changed
    # coordinates from clean source coordinates plus inferred per-group displacement.
    x_att = x_att.merge(agg[["split", "sender_id", "offset_x", "offset_y"]],
                        on=["split", "sender_id"], suffixes=("_observed", "_manifest"),
                        validate="many_to_one")
    dx = (x_att.source_sender_pos_x + x_att.offset_x_manifest - x_att.sender_pos_x).abs()
    dy = (x_att.source_sender_pos_y + x_att.offset_y_manifest - x_att.sender_pos_y).abs()
    n_replay_mismatch = int(((dx > atol) | (dy > atol)).sum())
    reports.append(f"Attacker rows failing source+manifest=NVBD replay: {n_replay_mismatch}")
    if n_replay_mismatch:
        raise ValueError("Inferred-offset replay FAILED; maximum absolute error "
                         f"{max(dx.max(),dy.max()):.6g}")
    return agg[["split", "sender_id", "n_records", "offset_x", "offset_y"]], reports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="Existing NVBD dataset.csv or dataset.zip")
    for key in ("train", "validation", "test"):
        ap.add_argument(f"--{key}-source", type=Path, help=f"Original, CLEAN {key} receiver JSON directory or ZIP")
    ap.add_argument("--outdir", type=Path, default=Path("nvbd_offset_audit"),
                    help="NEW directory for report/manifest; will not overwrite existing directory")
    ap.add_argument("--tolerance", type=float, default=1e-5)
    args = ap.parse_args(argv)
    d = dataset_load(args.dataset)
    report = dataset_report(d)
    sources = [args.train_source, args.validation_source, args.test_source]
    if any(p is not None for p in sources) and not all(p is not None for p in sources):
        ap.error("Source matching requires ALL THREE of --train-source, --validation-source, --test-source")
    if all(p is not None for p in sources):
        raw = pd.concat([read_source(p, split) for split, p in zip(SPLITS, sources)], ignore_index=True)
        manifest, details = audited_offsets(d, raw, atol=args.tolerance)
        report.extend(details)
        report.append("Source-relative spoofing-offset audit: PASSED")
    else:
        manifest = None
        report.append("Source-relative spoofing-offset audit: NOT RUN (raw source files not provided)")
    # Validate everything BEFORE creating an output directory.
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to replace {args.outdir}. Choose a new --outdir or archive it first.")
    args.outdir.mkdir(parents=True)
    (args.outdir / "construction_audit.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    if manifest is not None:
        manifest.to_csv(args.outdir / "inferred_sender_offsets.csv", index=False, float_format="%.12g")
    print("\n".join(report))
    print("Saved report under", args.outdir)
    if manifest is None:
        print("INCOMPLETE: this run verified NVBD internal properties only, NOT source reconstruction.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, zipfile.BadZipFile) as e:
        print("AUDIT STOPPED:", e, file=sys.stderr)
        sys.exit(2)
