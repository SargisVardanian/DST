# robust_count_dsb_rules_v2.py
from __future__ import annotations
from pathlib import Path
import re, csv

# 1) классические правила: "Class N: ... || mass=[...]"
RE_CLASS = re.compile(rb'(?m)^\s*Class\s+\d+\s*:.*?\|\|\s*mass=\[')
# 2) без метки класса: "... || mass=[...]"
RE_ANY   = re.compile(rb'(?m)^\s*(?:Class\s+\d+\s*:\s*)?.*?\|\|\s*mass=\[')

def _load_normalize_bytes(path: Path) -> bytes:
    b = path.read_bytes()
    if b.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
        b = b[3:]
    b = b.replace(b'\x00', b'')        # убрать NUL (вдруг UTF-16)
    b = b.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    return b

def count_rules_in_file(path: Path) -> int:
    data = _load_normalize_bytes(path)
    n_class = len(RE_CLASS.findall(data))
    if n_class:               # обычный случай
        return n_class
    # fallback для статических/безклассовых дампов
    return len(RE_ANY.findall(data))

def write_rule_counts_csv(
    input_dir: str | Path = "dsb_rules",
    pattern: str = "*.dsb",
    output_csv: str | Path = "results/rule_counts.csv",
) -> Path:
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
    pattern = args[1] if len(args) > 1 else "*.dsb"
    out_csv = args[2] if len(args) > 2 else "results/rule_counts.csv"
    out = write_rule_counts_csv(in_dir, pattern, out_csv)
    print(f"CSV создан: {out}")
