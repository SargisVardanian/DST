"""
Count rules in .dsb (Dempster-Shafer Binary) rule files and export to CSV.

This utility script scans .dsb files for rule patterns and generates a CSV summary.
"""

from __future__ import annotations
from pathlib import Path
import re, csv

# Pattern 1: Standard rules with class label: "Class N: ... || mass=[...]"
RE_CLASS = re.compile(rb'(?m)^\s*Class\s+\d+\s*:.*?\|\|\s*mass=\[')
# Pattern 2: Rules without class label: "... || mass=[...]"
RE_ANY   = re.compile(rb'(?m)^\s*(?:Class\s+\d+\s*:\s*)?.*?\|\|\s*mass=\[')

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
    """
    Count the number of rules in a .dsb file.
    
    Args:
        path (Path): Path to .dsb file
        
    Returns:
        int: Number of rules found
    """
    data = _load_normalize_bytes(path)
    n_class = len(RE_CLASS.findall(data))
    if n_class:  # Standard case with class labels
        return n_class
    # Fallback for static/unlabeled rule dumps
    return len(RE_ANY.findall(data))

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

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    in_dir  = args[0] if len(args) > 0 else "dsb_rules"
    pattern = args[1] if len(args) >1 else "*.dsb"
    out_csv = args[2] if len(args) > 2 else "results/rule_counts.csv"
    out = write_rule_counts_csv(in_dir, pattern, out_csv)
    print(f"CSV created: {out}")
