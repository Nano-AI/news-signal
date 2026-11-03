"""Trusted pass runner shared by the accuracy gate, the benchmark and the CPU check.

One pass = verify the candidate tree, build the train file, run the candidate's prepare step, replay stories
through newsd (in this process) against the candidate service (a subprocess), and score the pass.

`infosys` is imported only after its tree digest matches the pin in `gate.toml`, so an edited `infosys/` is
refused before any of its code runs.
"""

from __future__ import annotations

import asyncio
import gc
import gzip
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infosys.newsd import PassLog
    from infosys.replay import ReplayPlan
    from infosys.score import PassScore
    from infosys.store import Bars, Split, Story

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
GATE_PATH = TASK / "gate.toml"
MANIFEST_PATH = TASK / "data.manifest.json"

MAX_STORIES = 30000  # accuracy pass size; MAX_STORIES * ms_per_headline must stay under ~360 s
BURSTS_WINDOW_S = 300
BURSTS_BUDGET_S = 180.0  # raise to 240 only if the real data gives fewer than 300 measured stories
MAX_TRAIN_STORIES = 50_000
PREPARE_TIMEOUT_S = 1200
MAX_FILE_BYTES = 5 * 1024 * 1024
WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".npz", ".parquet"})
_EXCLUDED_TOP = frozenset({".git", ".venv", ".vibesys", "infosys"})
_STOP_GRACE_S = 10.0
_EXIT_GRACE_S = 10.0  # a clean exit just before newsd reports done is not a crash
_LOG_TAIL_LINES = 40


class Mode(StrEnum):
    BURSTS = "bursts"
    ACCURACY = "accuracy"
    SMOKE = "smoke"


class PassError(Exception):
    def __init__(self, message: str, requirement: str | None = None):
        super().__init__(message)
        self.message = message
        self.requirement = requirement


# -- settings (selectors import infosys lazily, after the digest check) ---------------------------------


def _bursts(stories: list[Story]) -> list[Story]:
    from infosys.replay import burst_windows

    return burst_windows(stories, window_s=BURSTS_WINDOW_S, budget_s=BURSTS_BUDGET_S, gap_cap_s=2.0)


def _accuracy(stories: list[Story]) -> list[Story]:
    from infosys.replay import sample_stories

    return sample_stories(stories, MAX_STORIES, seed=2)


def _all(stories: list[Story]) -> list[Story]:
    return list(stories)


@dataclass(frozen=True)
class PassSettings:
    select: Callable[[list[Story]], list[Story]]
    gap_cap_s: float
    seed: int
    drain_s: float
    ready_timeout_s: float
    warmup: int  # latest N train stories
    warmup_hz: float


SETTINGS: dict[Mode, PassSettings] = {
    Mode.BURSTS: PassSettings(_bursts, 2.0, 1, 10.0, 600.0, 64, 10.0),
    Mode.ACCURACY: PassSettings(_accuracy, 0.01, 2, 600.0, 600.0, 64, 10.0),
    Mode.SMOKE: PassSettings(_all, 0.05, 3, 5.0, 120.0, 16, 20.0),
}


# -- gate and candidate checks --------------------------------------------------------------------------

_GATE_KEYS = {
    "infosys_sha256": str,
    "h_minutes": int,
    "h_reason": str,
    "max_missed_frac": float,
    "max_duplicate_frac": float,
    "max_invalid": int,
    "baseline": dict,
}
_BASELINE_KEYS = {
    "frozen": bool,
    "ic": float,
    "ic_se": float,
    "ic_floor": float,
    "validation_stories": int,
    "scored_pairs": int,
    "p99_ms": list,
    "ms_per_headline": float,
}


def load_gate() -> dict:
    with GATE_PATH.open("rb") as f:
        gate = tomllib.load(f)
    for table, spec in ((gate, _GATE_KEYS), (gate.get("baseline", {}), _BASELINE_KEYS)):
        unknown = set(table) - set(spec)
        if unknown:
            raise PassError(f"gate.toml: unknown key {sorted(unknown)[0]}")
        for key, kind in spec.items():
            if key not in table:
                raise PassError(f"gate.toml: missing key {key}")
            ok = isinstance(table[key], kind) or (kind is float and isinstance(table[key], int))
            if isinstance(table[key], bool) and kind is not bool:
                ok = False
            if not ok:
                raise PassError(f"gate.toml: {key} must be {kind.__name__}")
    if len(gate["baseline"]["p99_ms"]) != 2:
        raise PassError("gate.toml: baseline.p99_ms must have two entries")
    return gate


def infosys_digest(root: Path) -> str:
    """sha256 over the sorted (relative path, bytes) of every file under `root/infosys`, caches excluded."""
    base = root / "infosys"
    files = sorted(
        p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    h = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        h.update(path.relative_to(base).as_posix().encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def check_candidate_files(root: Path) -> None:
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_TOP]
        for name in filenames:
            path = here / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES or path.suffix.lower() in WEIGHT_SUFFIXES:
                offenders.append(path.relative_to(root).as_posix())
    if offenders:
        first = sorted(offenders)[0]
        raise PassError(
            f"weights or large files must not be committed: {first}", requirement="no_weights_in_git"
        )


def _trust_infosys(root: Path, gate: dict) -> None:
    if infosys_digest(root) != gate["infosys_sha256"]:
        raise PassError("infosys/ differs from the digest pinned in gate.toml", requirement="infosys_unchanged")
    if "infosys" not in sys.modules and str(root) not in sys.path:
        sys.path.insert(0, str(root))


# -- subprocess helpers ---------------------------------------------------------------------------------


def _sanitized_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEWS_SIGNAL_")}
    env["HF_HUB_OFFLINE"] = "1"
    return env


def _tail(path: Path, lines: int = _LOG_TAIL_LINES) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "(no log)"
    return "\n".join(text.splitlines()[-lines:]) or "(empty log)"


def _kill_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_prepare(root: Path, train: Path, out: Path, device: str, env: dict, log: Path) -> None:
    cmd = [sys.executable, "-m", "serving.prepare", "--train", str(train), "--out", str(out), "--device", device]
    with log.open("wb") as lf:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=PREPARE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_group(proc.pid, signal.SIGKILL)
            proc.wait()
            raise PassError(f"prepare exceeded {PREPARE_TIMEOUT_S} s: {_tail(log)}") from None
        finally:
            _kill_group(proc.pid, signal.SIGKILL)  # stragglers of a finished prepare
    if rc != 0:
        raise PassError(f"prepare exited with code {rc}: {_tail(log)}")


async def _stop(proc: asyncio.subprocess.Process) -> None:
    _kill_group(proc.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), _STOP_GRACE_S)
    except TimeoutError:
        pass
    _kill_group(proc.pid, signal.SIGKILL)
    await proc.wait()


async def _drive(plan: ReplayPlan, *, ready_timeout_s: float, drain_s: float, make_cmd: Callable[[str], list[str]],
                 root: Path, env: dict, log: Path) -> PassLog:
    """Serve `plan` while racing the service process: its early exit fails the pass at once."""
    from infosys.newsd import ReadyTimeout, serve_pass

    port = _free_port()
    procs: list[asyncio.subprocess.Process] = []
    started = asyncio.Event()

    async def on_listening() -> None:
        with log.open("wb") as lf:
            procs.append(
                await asyncio.create_subprocess_exec(
                    *make_cmd(f"http://127.0.0.1:{port}"), cwd=root, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True
                )
            )
        started.set()

    async def exited() -> int:
        await started.wait()
        return await procs[0].wait()

    serve = asyncio.create_task(
        serve_pass(plan, port=port, ready_timeout_s=ready_timeout_s, drain_s=drain_s, on_listening=on_listening)
    )
    watch = asyncio.create_task(exited())
    try:
        await asyncio.wait({serve, watch}, return_when=asyncio.FIRST_COMPLETED)
        if not serve.done():
            rc = watch.result()
            if rc == 0:
                await asyncio.wait({serve}, timeout=_EXIT_GRACE_S)
            if not serve.done():
                serve.cancel()
                await asyncio.gather(serve, return_exceptions=True)
                raise PassError(f"serving exited with code {rc} before the pass ended: {_tail(log)}")
        return serve.result()
    except ReadyTimeout as e:
        raise PassError(f"serving never became ready ({e}): {_tail(log)}") from None
    finally:
        watch.cancel()
        if not serve.done():
            serve.cancel()
        await asyncio.gather(serve, watch, return_exceptions=True)
        if procs:
            await _stop(procs[0])


# -- the pass -------------------------------------------------------------------------------------------


@dataclass
class PassResult:
    plan: ReplayPlan
    log: PassLog
    bars: Bars
    score: PassScore


def rescore(result: PassResult, horizon_min: int) -> PassScore:
    from infosys.score import score_pass

    return score_pass(result.plan, result.log, result.bars, horizon_min=horizon_min)


def _write_train_file(path: Path, stories: list[Story], bars: Bars, horizon_min: int) -> None:
    from infosys.protocol import TrainRecord

    lines = []
    for s in sorted(stories, key=lambda s: (s.published_at, s.story_id))[-MAX_TRAIN_STORIES:]:
        rec = TrainRecord(
            story_id=s.story_id,
            source=s.source,
            published_at=s.published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            tickers=list(s.tickers),
            headline=s.headline,
            summary=s.summary,
            returns={t: bars.forward_return(t, s.published_at, horizon_min) for t in s.tickers},
        )
        lines.append(rec.model_dump_json() + "\n")
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write("".join(lines).encode())


def run_pass(
    mode: Mode,
    split: Split,
    *,
    device: str = "cuda",
    model: str = "ProsusAI/finbert",
    root: Path = ROOT,
    data_dir: Path | None = None,
    drain_s: float | None = None,
) -> PassResult:
    gate = load_gate()
    _trust_infosys(root, gate)
    check_candidate_files(root)
    from infosys.replay import plan_pass
    from infosys.score import score_pass
    from infosys.store import StoreMismatch, load_split, verify_store

    settings = SETTINGS[mode]
    if mode is Mode.SMOKE:
        data_dir = root / "infosys" / "fixtures" / "tiny"
        manifest_path = data_dir / "manifest.json"
    else:
        if data_dir is None:
            env_dir = os.environ.get("NEWS_SIGNAL_DATA")
            if not env_dir:
                raise PassError("NEWS_SIGNAL_DATA is not set")
            data_dir = Path(env_dir)
        manifest_path = MANIFEST_PATH
    try:
        pinned = json.loads(manifest_path.read_text())
        verify_store(data_dir, pinned, ("train", split))
    except StoreMismatch as e:
        raise PassError(f"data store file {e} does not match the pinned manifest") from None
    except (OSError, ValueError):
        raise PassError("data manifest is missing or unreadable") from None

    train, train_bars = load_split(data_dir, "train")
    stories, bars = load_split(data_dir, split)
    train = sorted(train, key=lambda s: (s.published_at, s.story_id))
    measured = settings.select(sorted(stories, key=lambda s: (s.published_at, s.story_id)))
    warmup = train[-settings.warmup:]
    plan = plan_pass(measured, warmup, gap_cap_s=settings.gap_cap_s, seed=settings.seed,
                     warmup_rate_hz=settings.warmup_hz)

    env = _sanitized_env()
    with tempfile.TemporaryDirectory(prefix="ns-pass-") as tmp:
        tmp_dir = Path(tmp)
        train_file, prepared = tmp_dir / "train.jsonl.gz", tmp_dir / "prepared"
        _write_train_file(train_file, train, train_bars, gate["h_minutes"])
        _run_prepare(root, train_file, prepared, device, env, tmp_dir / "prepare.log")

        def make_cmd(url: str) -> list[str]:
            return [sys.executable, "-m", "serving", "--newsd", url, "--model", model, "--device", device,
                    "--prepared", str(prepared)]

        drive = _drive(
            plan,
            ready_timeout_s=settings.ready_timeout_s,
            drain_s=settings.drain_s if drain_s is None else drain_s,
            make_cmd=make_cmd,
            root=root,
            env=env,
            log=tmp_dir / "serving.log",
        )
        # A full collection over the loaded stores would stall newsd's scheduler for ~100 ms (and fail the
        # send-lag gate), so the heap is frozen and collection is off while the pass runs.
        gc.collect()
        gc.freeze()
        gc.disable()
        try:
            log = asyncio.run(drive)
        finally:
            gc.enable()
            gc.unfreeze()
    score = score_pass(plan, log, bars, horizon_min=gate["h_minutes"])
    return PassResult(plan, log, bars, score)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--print-infosys-digest", action="store_true")
    if p.parse_args().print_infosys_digest:
        print(infosys_digest(ROOT))
