from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PAPER_SEEDS = [42]
DEFAULT_PAPER_TEST_SIZE = 0.2
DEFAULT_STANDARD_SEEDS = [42]
DEFAULT_STANDARD_TEST_SIZE = 0.2


@dataclass(frozen=True)
class BenchmarkProtocol:
    paper_mode: bool
    mode_label: str
    seeds: list[int]
    test_size: float
    requested_seeds: list[int]
    requested_test_size: float
    override_messages: list[str]


def parse_seed_list(raw: str | None) -> list[int]:
    tokens = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    return [int(token) for token in tokens] or list(DEFAULT_STANDARD_SEEDS)


def resolve_protocol(*, requested_seeds: list[int], requested_test_size: float, paper_mode: bool) -> BenchmarkProtocol:
    override_messages: list[str] = []
    if paper_mode:
        if requested_seeds != DEFAULT_PAPER_SEEDS:
            override_messages.append(
                f"[RUN] paper-mode fixed seeds={DEFAULT_PAPER_SEEDS}; ignoring requested seeds={requested_seeds}"
            )
        if abs(float(requested_test_size) - DEFAULT_PAPER_TEST_SIZE) > 1e-12:
            override_messages.append(
                f"[RUN] paper-mode fixed test_size={DEFAULT_PAPER_TEST_SIZE}; ignoring requested test_size={requested_test_size}"
            )
        return BenchmarkProtocol(
            paper_mode=True,
            mode_label="paper",
            seeds=list(DEFAULT_PAPER_SEEDS),
            test_size=float(DEFAULT_PAPER_TEST_SIZE),
            requested_seeds=list(requested_seeds),
            requested_test_size=float(requested_test_size),
            override_messages=override_messages,
        )
    return BenchmarkProtocol(
        paper_mode=False,
        mode_label="standard",
        seeds=list(requested_seeds) or list(DEFAULT_STANDARD_SEEDS),
        test_size=float(requested_test_size),
        requested_seeds=list(requested_seeds),
        requested_test_size=float(requested_test_size),
        override_messages=[],
    )


def protocol_from_cli(*, raw_seeds: str | None, raw_test_size: float, paper_mode: bool) -> BenchmarkProtocol:
    return resolve_protocol(
        requested_seeds=parse_seed_list(raw_seeds),
        requested_test_size=float(raw_test_size),
        paper_mode=bool(paper_mode),
    )


def protocol_from_passthrough(args: list[str]) -> BenchmarkProtocol:
    paper_mode = "--paper-mode" in args
    raw_seeds = ""
    raw_test_size = str(DEFAULT_STANDARD_TEST_SIZE)
    for idx, token in enumerate(args):
        if token == "--seeds" and idx + 1 < len(args):
            raw_seeds = str(args[idx + 1]).strip()
        elif token.startswith("--seeds="):
            raw_seeds = token.split("=", 1)[1].strip()
        elif token == "--test-size" and idx + 1 < len(args):
            raw_test_size = str(args[idx + 1]).strip()
        elif token.startswith("--test-size="):
            raw_test_size = token.split("=", 1)[1].strip()
    return protocol_from_cli(raw_seeds=raw_seeds, raw_test_size=float(raw_test_size), paper_mode=paper_mode)
