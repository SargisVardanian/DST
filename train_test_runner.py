from __future__ import annotations

import sys

from src.train_test_runner import benchmark_main


if __name__ == "__main__":
    raise SystemExit(benchmark_main(sys.argv[1:]))
