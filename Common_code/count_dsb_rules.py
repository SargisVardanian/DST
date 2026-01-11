"""
Count rules in .dsb (Dempster-Shafer Binary) rule files and export to CSV.

This utility script scans .dsb files and generates:
  1) per-file rule counts
  2) per-dataset rule counts grouped by algorithm (STATIC/RIPPER/FOIL)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import csv
import re
from collections import defaultdict

# Current repo format (Common_code/DSModelMultiQ.save_rules_dsb):
#   "class <label> :: <caption> || masses: [...]"
# Count rules as lines that start with "class" (case-insensitive).
RE_RULE_LINE = re.compile(rb"(?im)^\s*class\s+")

ALGOS = ("static", "ripper", "foil")

def _load_normalize_bytes(path: Path) -> bytes:
    """
    Load file bytes and normalize encoding issues.
    
    Args:
        path (Path): File path to load
        
    Returns:
        bytes: Normalized file content
    """
    b = path.read_bytes()
    if b.startswith(b'\xef\xbb\xbf'):  # Remove UTF-8 BOM
        b = b[3:]
    b = b.replace(b'\x00', b'')        # Remove NUL characters (UTF-16 artifacts)
    b = b.replace(b'\r\n', b'\n').replace(b'\r', b'\n')  # Normalize line endings
    return b

def count_rules_in_file(path: Path) -> int:
    """Count rules in a .dsb file."""
    data = _load_normalize_bytes(path)
    return len(RE_RULE_LINE.findall(data))


def parse_dsb_name(path: Path) -> tuple[str, str, str] | None:
    """Parse <algo>_<dataset>_<kind>.dsb (with optional prefix tag).

    Supports names like:
      - static_df_wine_dst.dsb
      - ripper_bank-full_raw.dsb
      - stable_ripper_german_dst.dsb
    """
    stem = path.stem
    parts = stem.split("_")
    algo_idx = None
    for i, p in enumerate(parts):
        if p.lower() in ALGOS:
            algo_idx = i
            break
    if algo_idx is None:
        return None
    if len(parts) < algo_idx + 3:
        return None
    kind = parts[-1].lower()
    if kind not in {"raw", "dst"}:
        return None
    algo = parts[algo_idx].lower()
    dataset = "_".join(parts[algo_idx + 1 : -1])
    if not dataset:
        return None
    return dataset, algo, kind

def write_rule_counts_csv(
    input_dir: str | Path = "dsb_rules",
    pattern: str = "*.dsb",
    output_csv: str | Path = "results/rule_counts.csv",
) -> Path:
    """
    Scan directory for .dsb files and write rule counts to CSV.
    
    Args:
        input_dir (str | Path): Directory containing .dsb files. Default: 'dsb_rules'
        pattern (str): Glob pattern for files to process. Default: '*.dsb'
        output_csv (str | Path): Output CSV file path. Default: 'results/rule_counts.csv'
        
    Returns:
        Path: Path to created CSV file
    """
    in_dir  = Path(input_dir)
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob(pattern))
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(["file", "rules"])
        for fp in files:
            try:
                n = count_rules_in_file(fp)
            except Exception:
                n = 0
            wr.writerow([str(fp), n])
    return out_csv


def write_dataset_rule_counts_csv(
    input_dir: str | Path = "dsb_rules",
    *,
    kind: str = "dst",
    output_csv: str | Path = "results/rule_counts_by_dataset.csv",
) -> Path:
    """Write a per-dataset CSV with number of rules per algorithm."""
    in_dir = Path(input_dir)
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    want_kind = str(kind).lower()
    if want_kind not in {"dst", "raw", "both"}:
        raise ValueError("kind must be one of: dst, raw, both")

    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {a: 0 for a in ALGOS})
    for fp in sorted(in_dir.rglob("*.dsb")):
        meta = parse_dsb_name(fp)
        if meta is None:
            continue
        dataset, algo, file_kind = meta
        if want_kind != "both" and file_kind != want_kind:
            continue
        try:
            n = count_rules_in_file(fp)
        except Exception:
            n = 0
        counts[(dataset, file_kind)][algo] = int(n)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Dataset", "Kind", "STATIC", "RIPPER", "FOIL"])
        for (dataset, file_kind) in sorted(counts.keys(), key=lambda t: (t[0], t[1])):
            row = counts[(dataset, file_kind)]
            static_n = int(row.get("static", 0))
            ripper_n = int(row.get("ripper", 0))
            foil_n = int(row.get("foil", 0))
            wr.writerow([dataset, file_kind, static_n, ripper_n, foil_n])

    return out_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="dsb_rules")
    p.add_argument("--mode", choices=["file", "dataset"], default="dataset")
    p.add_argument("--kind", choices=["dst", "raw", "both"], default="dst")
    p.add_argument("--output-csv", default=None)
    p.add_argument("--pattern", default="*.dsb", help="Used only with --mode file")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if args.mode == "file":
        out_csv = args.output_csv or "results/rule_counts.csv"
        out = write_rule_counts_csv(in_dir, args.pattern, out_csv)
    else:
        out_csv = args.output_csv or "results/rule_counts_by_dataset.csv"
        out = write_dataset_rule_counts_csv(in_dir, kind=args.kind, output_csv=out_csv)
    print(f"CSV created: {out}")
