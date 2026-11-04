"""Trusted pass runner shared by the accuracy gate, the benchmark and the CPU check.

One pass = verify the candidate tree, build the train file, run the candidate's prepare step, replay stories
through newsd (in this process) against the candidate service (a subprocess), and score the pass.

`infosys` is imported only after its tree digest matches the pin in `gate.toml`, so an edited `infosys/` is
refused before any of its code runs.
"""

from __future__ import annotations

import asyncio
import gc
import glob
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
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
HIDE_FAILED_RC = 97  # the namespace wrapper could not set itself up
_UNSHARE = ["unshare", "-Urm"]  # unprivileged user + mount namespace
_STOP_GRACE_S = 10.0
_EXIT_GRACE_S = 10.0  # a clean exit just before newsd reports done is not a crash
_LOG_TAIL_LINES = 20
_LOG_TAIL_CHARS = 200  # per line: a log tail is the only text that flows back to the agent
MAX_TREE_BYTES = 50 * 1024 * 1024
SERVING_NICE = 10  # keeps the candidate's CPU threads from starving newsd's scheduler


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
    "repo_sha256": str,
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
    "bursts_ic": float,
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


_DIGEST_SKIP_NAMES = frozenset({"__pycache__", ".DS_Store", ".hypothesis", ".pytest_cache"})
REPO_PINNED = ("pyproject.toml", "uv.lock", "docs/protocol.md")


def _digest(base: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(files):
        data = path.read_bytes()
        h.update(path.relative_to(base).as_posix().encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def infosys_digest(root: Path) -> str:
    """sha256 over the sorted (relative path, bytes) of every file under `root/infosys`, caches excluded."""
    base = root / "infosys"
    return _digest(
        base,
        [p for p in base.rglob("*") if p.is_file() and not _DIGEST_SKIP_NAMES & set(p.parts) and p.suffix != ".pyc"],
    )


def repo_digest(root: Path) -> str:
    """sha256 over the dependency pins and the wire protocol; a missing file is its own digest."""
    return _digest(root, [root / name for name in REPO_PINNED if (root / name).is_file()])


def check_candidate_files(root: Path) -> None:
    offenders, total = [], 0
    real_root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_TOP]
        for name in [*dirnames, *filenames]:
            path = here / name
            if path.is_symlink() and not path.resolve().is_relative_to(real_root):
                offenders.append(f"{path.relative_to(root).as_posix()} (symlink outside the tree)")
        for name in filenames:
            path = here / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            total += size
            if size > MAX_FILE_BYTES or path.suffix.lower() in WEIGHT_SUFFIXES:
                offenders.append(path.relative_to(root).as_posix())
    if not offenders and total > MAX_TREE_BYTES:
        offenders.append(f"the tree is {total >> 20} MB, over {MAX_TREE_BYTES >> 20} MB")
    if offenders:
        raise PassError(
            f"weights, large files or links outside the tree must not be committed: {sorted(offenders)[0]}",
            requirement="no_weights_in_git",
        )


def _trust_infosys(root: Path, gate: dict) -> None:
    if infosys_digest(root) != gate["infosys_sha256"]:
        raise PassError("infosys/ differs from the digest pinned in gate.toml", requirement="infosys_unchanged")
    if repo_digest(root) != gate["repo_sha256"]:
        raise PassError(
            "pyproject.toml, uv.lock or docs/protocol.md differ from the digest pinned in gate.toml",
            requirement="infosys_unchanged",
        )
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
    tail = [line[:_LOG_TAIL_CHARS] for line in text.splitlines()[-lines:]]
    return "\n".join(tail) or "(empty log)"


def _serving_preexec() -> None:
    """Runs in the serving child before exec: lower priority, and die with the harness (no orphan on the GPU)."""
    os.nice(SERVING_NICE)
    if sys.platform.startswith("linux"):
        _libc().prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _libc():
    import ctypes

    return ctypes.CDLL(None, use_errno=True)


_SETPRIV = ["setpriv", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs"]
_ENV_DIR_KEY = re.compile(r"(CACHE|_HOME|_DIR|_PATH_DB)$")
_SCRATCH_ENV = (
    "TMPDIR=$S XDG_CACHE_HOME=$S/cache TRITON_CACHE_DIR=$S/triton TORCHINDUCTOR_CACHE_DIR=$S/inductor "
    "MIOPEN_USER_DB_PATH=$S/miopen MIOPEN_CUSTOM_CACHE_DIR=$S/miopen PYTHONPYCACHEPREFIX=$S/pyc"
)


def protected_paths(root: Path, work_dir: Path, env: dict[str, str]) -> list[Path]:
    """Directories the serving process must not be able to change: /tmp, $HOME, the checkout, the work dir
    (prepared outputs), HF_HOME and every cache or home directory named in the environment."""
    found = [Path("/tmp"), root, work_dir]
    for key in ("HOME", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME"):
        if env.get(key):
            found.append(Path(env[key]))
    for key, value in env.items():
        if _ENV_DIR_KEY.search(key) and value.startswith("/"):
            found.append(Path(value))
    real = sorted({p.resolve() for p in found if p.is_dir() and p.resolve() != Path("/")})
    return [p for p in real if not any(q != p and p.is_relative_to(q) for q in real)]  # drop nested ones


def hidden(cmd: list[str], data_dir: Path | None, protect: list[Path] | None = None) -> list[str]:
    """Run `cmd` in an unprivileged user + mount namespace and drop every capability before exec.

    - An empty tmpfs hides the scoring data directory.
    - With `protect` (the serving phase): those directories become read-only binds, /var/tmp and /dev/shm are
      fresh tmpfs, and TMPDIR and the torch, triton, miopen and XDG caches point at a scratch tmpfs, so nothing
      the process writes survives the attempt.
    - Capabilities and the bounding set are then dropped with no-new-privs, so the mounts cannot be undone and a
      nested namespace gains nothing.
    Any mount or setpriv failure exits non-zero; there is no fallback. Paths are arguments, not environment.
    """
    if data_dir is None:
        return cmd
    iso = protect is not None
    lines = [f'd="$1"; iso="$2"; n="$3"; shift 3', f'mount -t tmpfs tmpfs "$d" || exit {HIDE_FAILED_RC}']
    lines += [
        'if [ "$iso" = 1 ]; then',
        f'  shm=$(readlink -f /dev/shm); mount -t tmpfs tmpfs "$shm" || exit {HIDE_FAILED_RC}',
        f'  mount -t tmpfs tmpfs /var/tmp || exit {HIDE_FAILED_RC}',
        '  while [ "$n" -gt 0 ]; do p="$1"; shift; n=$((n - 1))',
        f'    mount --rbind "$p" "$p" && mount -o remount,bind,ro "$p" || exit {HIDE_FAILED_RC}',
        "  done",
        '  S="$shm/ns"; mkdir -p "$S/cache" "$S/triton" "$S/inductor" "$S/miopen" "$S/pyc"',
        f"  export {_SCRATCH_ENV}",
        "fi",
        f'exec {" ".join(_SETPRIV)} "$@"',
    ]
    prot = [str(p) for p in protect or []]
    return [*_UNSHARE, "sh", "-c", "\n".join(lines), "sh", str(data_dir), "1" if iso else "0", str(len(prot)),
            *prot, *cmd]


def _check_can_hide(data_dir: Path, protect: list[Path]) -> None:
    """Refuse the pass unless the full wrapper can be set up and nothing the candidate needs sits inside the data."""
    real = data_dir.resolve()
    for path in protect:
        if path.resolve().is_relative_to(real):
            raise PassError("the repo, work files or model cache lie inside the scoring data directory")
    try:
        r = subprocess.run(hidden(["true"], data_dir, protect), capture_output=True, timeout=60)
        ok = r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        raise PassError("the scoring data cannot be hidden from the candidate (unshare, mount or setpriv failed); "
                        "the pass is refused")


def _make_undumpable() -> None:
    """PR_SET_DUMPABLE=0: the same-user serving child cannot read this process's /proc files (the store path)."""
    if sys.platform.startswith("linux"):
        _libc().prctl(4, 0)


def _kill_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_prepare(root: Path, train: Path, out: Path, device: str, env: dict, log: Path,
                 hide: Path | None = None) -> None:
    cmd = hidden([sys.executable, "-m", "serving.prepare", "--train", str(train), "--out", str(out), "--device", device], hide)
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
    if rc == HIDE_FAILED_RC and hide is not None:
        raise PassError("the scoring data cannot be hidden from the candidate (tmpfs mount failed in prepare)")
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


# -- serve-phase probes (record only; nothing here decides a pass) ---------------------------------------

GPU_SAMPLE_PERIOD_S = 0.25  # at or above 100 ms so the sampler cannot disturb latency
_VRAM_USED_GLOB = "/sys/class/drm/card[0-9]*/device/mem_info_vram_used"  # bytes, one file per GPU
_MIB = 1 << 20


def _rocm_smi_vram_used() -> int | None:
    exe = shutil.which("rocm-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--showmeminfo", "vram", "--json"], capture_output=True, text=True, timeout=10,
                             check=True).stdout
        return sum(int(v["VRAM Total Used Memory (B)"]) for k, v in json.loads(out).items() if k.startswith("card"))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, AttributeError):
        return None


def read_vram_used_bytes() -> int | None:
    """VRAM in use summed over the host's GPUs (the node is exclusive): sysfs, else rocm-smi, else None."""
    total, found = 0, False
    for path in glob.glob(_VRAM_USED_GLOB):
        try:
            total += int(Path(path).read_text())
            found = True
        except (OSError, ValueError):
            pass
    return total if found else _rocm_smi_vram_used()


class GpuMemSampler:
    """Polls whole-device VRAM from the harness (outside the candidate's namespace; the candidate reports nothing).

    `idle_mib` is the first reading, taken before the service is spawned; `peak_mib` the maximum since. Both stay
    None when the first reading fails (CPU host, macOS, fixture), and a later failed reading is skipped.
    """

    def __init__(self, read: Callable[[], int | None] | None = None, period_s: float = GPU_SAMPLE_PERIOD_S):
        self.read, self.period_s = read or read_vram_used_bytes, period_s
        self.idle_mib: float | None = None
        self.peak_mib: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> bool:
        try:
            used = self.read()
        except Exception:  # a probe must never fail a pass
            used = None
        if used is None:
            return False
        mib = used / _MIB
        self.idle_mib = mib if self.idle_mib is None else self.idle_mib
        self.peak_mib = mib if self.peak_mib is None else max(self.peak_mib, mib)
        return True

    def start(self) -> None:
        if self.sample():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            self.sample()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)  # a rocm-smi read can take a few seconds
            self.sample()  # the reading at the end of the pass


@dataclass
class ServeProbe:
    """What the harness itself observed about the serving process in one attempt."""

    cold_start_s: float | None = None  # spawn of the serving process to its POST /ready, on the harness clock
    gpu_mem_idle_mib: float | None = None  # before the spawn
    gpu_mem_peak_mib: float | None = None  # from the spawn to the end of the pass


async def _drive(plan: ReplayPlan, *, ready_timeout_s: float, drain_s: float, make_cmd: Callable[[str], list[str]],
                 root: Path, env: dict, log: Path, hiding: bool = False, probe: ServeProbe | None = None,
                 sampler: GpuMemSampler | None = None) -> PassLog:
    """Serve `plan` while racing the service process: its early exit fails the pass at once."""
    from infosys.newsd import ReadyTimeout, serve_pass

    _make_undumpable()
    port = _free_port()
    procs: list[asyncio.subprocess.Process] = []
    started = asyncio.Event()

    probe = probe if probe is not None else ServeProbe()
    sampler = sampler if sampler is not None else GpuMemSampler()
    spawned: list[float] = []

    async def on_listening() -> None:
        sampler.start()
        spawned.append(time.monotonic())  # newsd's default clock; the same one stamps log.ready_at
        with log.open("wb") as lf:
            procs.append(
                await asyncio.create_subprocess_exec(
                    *make_cmd(f"http://127.0.0.1:{port}"), cwd=root, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                    preexec_fn=_serving_preexec,
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
                if rc == HIDE_FAILED_RC and hiding:
                    raise PassError("the scoring data cannot be hidden from the candidate (tmpfs mount failed); "
                                    "the pass is refused")
                raise PassError(f"serving exited with code {rc} before the pass ended: {_tail(log)}")
        result = serve.result()
        if spawned and result.ready_at is not None:
            probe.cold_start_s = result.ready_at - spawned[0]
        return result
    except ReadyTimeout as e:
        raise PassError(f"serving never became ready ({e}): {_tail(log)}") from None
    finally:
        sampler.stop()
        probe.gpu_mem_idle_mib, probe.gpu_mem_peak_mib = sampler.idle_mib, sampler.peak_mib
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
    score: PassScore  # of the last attempt
    first_invalid: PassScore | None = None  # the first attempt, when it was invalid and the pass was rerun
    serve: ServeProbe = field(default_factory=ServeProbe)  # of the last attempt


def run_with_retry(attempt: Callable[[], PassLog], score: Callable[[PassLog], PassScore]):
    """Run a pass; if it comes out invalid (newsd fell behind), run it once more and use the rerun.

    Returns (log, score, first_invalid_score_or_None). The gate fails only if the rerun is also invalid.
    """
    log = attempt()
    first = score(log)
    if first.valid:
        return log, first, None
    log = attempt()
    return log, score(log), first


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
    force_hide: bool = False,
) -> PassResult:
    gate = load_gate()
    _trust_infosys(root, gate)
    check_candidate_files(root)
    _make_undumpable()
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
    hide = None if mode is Mode.SMOKE and not force_hide else data_dir
    with tempfile.TemporaryDirectory(prefix="ns-pass-") as tmp:
        tmp_dir = Path(tmp)
        protect = protected_paths(root, tmp_dir, env) if hide is not None else None
        if hide is not None:
            _check_can_hide(hide, protect)
        train_file, prepared = tmp_dir / "train.jsonl.gz", tmp_dir / "prepared"
        _write_train_file(train_file, train, train_bars, gate["h_minutes"])
        _run_prepare(root, train_file, prepared, device, env, tmp_dir / "prepare.log", hide)

        def make_cmd(url: str) -> list[str]:
            return hidden([sys.executable, "-m", "serving", "--newsd", url, "--model", model, "--device", device,
                           "--prepared", str(prepared)], hide, protect)

        probes: list[ServeProbe] = []

        def attempt() -> PassLog:
            probes.append(ServeProbe())
            return asyncio.run(
                _drive(
                    plan,
                    ready_timeout_s=settings.ready_timeout_s,
                    drain_s=settings.drain_s if drain_s is None else drain_s,
                    make_cmd=make_cmd,
                    root=root,
                    env=env,
                    log=tmp_dir / "serving.log",
                    hiding=hide is not None,
                    probe=probes[-1],
                )
            )

        # A full collection over the loaded stores would stall newsd's scheduler for ~100 ms (and fail the
        # send-lag gate), so the heap is frozen while the pass runs.
        gc.collect()
        gc.freeze()
        try:
            log, score, first = run_with_retry(
                attempt, lambda lg: score_pass(plan, lg, bars, horizon_min=gate["h_minutes"])
            )
        finally:
            gc.unfreeze()
    return PassResult(plan, log, bars, score, first, probes[-1])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--print-infosys-digest", action="store_true")
    p.add_argument("--print-repo-digest", action="store_true")
    a = p.parse_args()
    if a.print_infosys_digest:
        print(infosys_digest(ROOT))
    if a.print_repo_digest:
        print(repo_digest(ROOT))
