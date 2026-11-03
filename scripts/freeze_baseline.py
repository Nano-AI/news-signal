"""Freeze the news-signal baseline: every real-data step of the task setup, in order, idempotent.

Run once the real store has been pulled, from the repo root of a clean checkout on the run host:

    NS_GPU_SSH=<ssh target of the GPU cluster> NS_CAMP=<work dir there> NS_SBATCH=<sbatch wrapper there> \\
    NS_PARTITION=<partition> NEWS_SIGNAL_DATA=<store dir here> python3 scripts/freeze_baseline.py

Stages (an artifact is reused only if its `.inputs` sidecar matches the current commit, digests, H and
passrun.py; `--force STAGE[,STAGE]` redoes one; `--stages a,b` limits the run):

    manifest  copy the store's manifest.json into the task as data.manifest.json
    pin       write the current infosys and repo digests into gate.toml
    budget    check the bursts replay has >= 300 stories, raising the window budget to 240 s if not
    cpu-acc   CPU dry run of the accuracy checker on every horizon
    h         pick H: the first of 5, 15, 30, 60 whose IC is more than 2 standard errors above zero
    sizing    cap MAX_STORIES so the accuracy replay takes <= 360 s at the baseline's ms per headline
    sync      refuse a dirty serving/ or infosys/, then rsync the checkout and the store to the cluster
    probe     (not in the default run) GPU job: the fixture pass in forced wrapper mode, umount and latency probes
    f1        GPU job: the accuracy checker on the baseline, with the gate's own drain (report mode)
    f2        GPU job: the bursts benchmark, twice
    freeze    write gate.toml, objectives.toml and the OBJECTIVE.md baseline section; print the numbers

Artifacts live in $NS_RUNS (default ~/ns-runs); a submitted job's id is kept in `<name>.jobid` until its
output is collected, so an interrupted run resumes the same job. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / ".vibesys" / "tasks" / "news-signal"
GATE, OBJECTIVES, PASSRUN, OBJECTIVE_MD = (
    TASK / n for n in ("gate.toml", "objectives.toml", "passrun.py", "OBJECTIVE.md")
)
HORIZONS = (5, 15, 30, 60)
MS_PER_HEADLINE = 8.51  # FinBERT baseline, median over single-headline calls on the target GPU (install smoke)
REPLAY_BUDGET_S = 360
F1_LIMIT = "00:20:00"  # replay (<= 5 min) + backlog + startup; the gate's drain ends early once pairs are in
F2_LIMIT = "00:20:00"  # two benchmark runs, each may rerun an invalid pass once
STAGES = ["manifest", "pin", "budget", "cpu-acc", "h", "sizing", "sync", "f1", "f2", "freeze"]
MAX_PENDING_S = 3 * 3600
SSH_TRIES = 3
MAX_UNKNOWN_POLLS = 30  # consecutive polls that could not tell the job's state
TERMINAL = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
}  # fmt: skip
POLL_S = 60


def say(msg: str) -> None:
    print(f"[freeze] {msg}", flush=True)


def need_env(name: str) -> str:
    if not os.environ.get(name):
        sys.exit(f"[freeze] {name} is not set (see the module docstring)")
    return os.environ[name]


def redact(text: str) -> str:
    """Hide the cluster's address and paths: they must not reach a terminal log that may be pasted."""
    for name, mark in (("NS_GPU_SSH", "<host>"), ("NS_CAMP", "<camp>"), ("NS_SBATCH", "<sbatch>")):
        if os.environ.get(name):
            text = text.replace(os.environ[name], mark)
    return text


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command; on failure exit with a redacted command stub and the tail of its stderr."""
    kw.setdefault("capture_output", True)
    try:
        return subprocess.run(cmd, check=True, text=True, **kw)
    except subprocess.CalledProcessError as e:
        tail = "\n".join((e.stderr or "").strip().splitlines()[-10:])
        sys.exit(redact(f"[freeze] command failed (exit {e.returncode}): {cmd[0]} ...\n{tail}"))


def uv_run(*args: str) -> subprocess.CompletedProcess:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(cache / "news-signal" / "venv")}
    return sh(["uv", "run", "--frozen", "python", *args], cwd=ROOT, env=env)


def ssh(script: str, *, input: str | None = None, tries: int = SSH_TRIES) -> str | None:
    """Run a remote script, retrying a bounded number of times; None if every try failed."""
    for attempt in range(tries):
        r = subprocess.run(["ssh", need_env("NS_GPU_SSH"), script], capture_output=True, text=True, input=input)
        if r.returncode == 0:
            return r.stdout
        if attempt + 1 < tries:
            time.sleep(5 * (attempt + 1))
    return None


def set_toml(path: Path, key: str, value: str, section: str | None = None) -> None:
    """Replace `key = ...` (the first one, inside `section` if given) with a literal TOML value."""
    text = path.read_text()
    start = text.index(f"[{section}]") if section else 0
    m = re.compile(rf"^{re.escape(key)} = .*$", re.M).search(text, start)
    if not m:
        sys.exit(f"[freeze] {path.name}: no key {key}")
    path.write_text(text[: m.start()] + f"{key} = {value}" + text[m.end():])


def toml_str(s: str) -> str:
    return json.dumps(s)


def derive(ic: float, ic_se: float, p99_a: float, p99_b: float) -> tuple[float, float]:
    """(ic_floor, relative_noise): the floor is two standard errors under the baseline IC, truncated to 4 decimals."""
    floor = math.floor((ic - 2 * ic_se) * 1e4) / 1e4
    noise = max(0.03, math.ceil(abs(p99_a - p99_b) / ((p99_a + p99_b) / 2) * 100) / 100)
    return floor, noise


def require_finite(**values: object) -> None:
    for name, v in values.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            sys.exit(f"[freeze] {name} is not a finite number ({v!r}): rerun the stage that produced it")


def tree_hash(base: Path, skip: frozenset[str] = frozenset()) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in skip):
        h.update(path.relative_to(base).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


# the freeze rewrites these three; every other task file is an input of the measurements
TASK_OUTPUTS = frozenset({"gate.toml", "OBJECTIVE.md", "objectives.toml"})


def task_hash() -> str:
    return tree_hash(TASK, TASK_OUTPUTS)


def git(*args: str) -> str:
    return sh(["git", *args], cwd=ROOT).stdout.strip()


class Freezer:
    def __init__(self, runs: Path, force: set[str]):
        self.runs, self.force = runs, force
        runs.mkdir(parents=True, exist_ok=True)

    # -- artifact freshness -----------------------------------------------------------------------

    def inputs(self, *, full: bool = True) -> dict:
        """What an artifact depends on. The CPU dry run picks H and sizes the gate, so it keys on the code only."""
        gate = tomllib.loads(GATE.read_text())
        base = {"head": git("rev-parse", "HEAD"), "infosys": gate["infosys_sha256"], "repo": gate["repo_sha256"]}
        if not full:
            return base
        return {**base, "h_minutes": gate["h_minutes"], "serving": tree_hash(ROOT / "serving"), "task": task_hash()}

    def todo(self, stage: str, artifact: Path) -> bool:
        sidecar = Path(f"{artifact}.inputs")
        if stage in self.force or not artifact.exists() or not sidecar.exists():
            return True
        if json.loads(sidecar.read_text()) != self.inputs(full=stage != "cpu-acc"):
            say(f"{artifact.name} was made from other inputs: redoing it")
            return True
        return False

    def stamp(self, artifact: Path, *, full: bool = True) -> None:
        Path(f"{artifact}.inputs").write_text(json.dumps(self.inputs(full=full), sort_keys=True))

    # -- CPU stages -------------------------------------------------------------------------------

    def manifest(self) -> None:
        src = Path(need_env("NEWS_SIGNAL_DATA")) / "manifest.json"
        dst = TASK / "data.manifest.json"
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            dst.write_bytes(src.read_bytes())
        say("data.manifest.json matches the store manifest")

    def pin(self) -> None:
        out = sh([sys.executable, str(PASSRUN), "--print-infosys-digest", "--print-repo-digest"]).stdout.split()
        set_toml(GATE, "infosys_sha256", toml_str(out[0]))
        set_toml(GATE, "repo_sha256", toml_str(out[1]))
        say(f"digests pinned: infosys {out[0][:12]}..., repo {out[1][:12]}...")

    def budget(self) -> None:
        # count the measured stories of the bursts pass without running it
        code = (
            f"import sys; sys.path[:0] = [{str(ROOT)!r}, {str(TASK)!r}]\n"
            "import os, passrun\n"
            "from pathlib import Path\n"
            "from infosys.store import load_split\n"
            "stories, _ = load_split(Path(os.environ['NEWS_SIGNAL_DATA']), 'validation')\n"
            "print(len(passrun.SETTINGS[passrun.Mode.BURSTS].select(stories)))\n"
        )

        def count() -> int:
            return int(uv_run("-c", code).stdout.split()[-1])

        n = count()
        say(f"bursts replay: {n} measured stories at the current window budget")
        if n < 300:
            PASSRUN.write_text(
                re.sub(r"^BURSTS_BUDGET_S = .*$", "BURSTS_BUDGET_S = 240.0", PASSRUN.read_text(), flags=re.M)
            )
            note = f"# bursts window budget raised from 180 s to 240 s: {n} stories at 180 s (< 300)\n"
            if note not in GATE.read_text():
                GATE.write_text(GATE.read_text() + note)
            n = count()
            say(f"fewer than 300 stories: BURSTS_BUDGET_S raised to 240.0, now {n}")
            if n < 300:
                sys.exit("[freeze] still fewer than 300 bursts stories at 240 s: stop and report")

    def cpu_acc(self) -> dict:
        out = self.runs / "cpu-acc.json"
        if self.todo("cpu-acc", out):
            say("CPU dry run of the accuracy checker (long)")
            uv_run(str(TASK / "accuracy_checker" / "checker.py"), "--report", "--device", "cpu", "--drain-s", "1800",
                   "--horizons", ",".join(map(str, HORIZONS)), "--json-out", str(out))  # fmt: skip
            self.stamp(out, full=False)
        return json.loads(out.read_text())

    def pick_h(self) -> None:
        horizons = self.cpu_acc()["horizons"]
        for h in HORIZONS:
            r = horizons[str(h)]
            if r["ic"] is not None and r["ic_se"] is not None and r["ic"] - 2 * r["ic_se"] > 0:
                reason = f"{h} min: ic {r['ic']:.3f}, 2se {2 * r['ic_se']:.3f}"
                set_toml(GATE, "h_minutes", str(h))
                set_toml(GATE, "h_reason", toml_str(reason))
                say(f"H = {reason}")
                return
        sys.exit("[freeze] no horizon has ic - 2*se > 0 on the CPU run: the universe must widen (new data pull)")

    def sizing(self) -> None:
        cap = min(30000, int(REPLAY_BUDGET_S / (MS_PER_HEADLINE / 1000)))
        PASSRUN.write_text(re.sub(r"^MAX_STORIES = \d+", f"MAX_STORIES = {cap}", PASSRUN.read_text(), flags=re.M))
        say(f"MAX_STORIES = {cap}")

    # -- GPU stages -------------------------------------------------------------------------------

    def sync(self, *, data: bool = True) -> None:
        if git("status", "--porcelain", "--", "serving", "infosys"):
            sys.exit("[freeze] serving/ or infosys/ has uncommitted changes: the baseline must be the committed tree")
        host, camp = need_env("NS_GPU_SSH"), need_env("NS_CAMP")
        sh(["rsync", "-a", "--delete", "--exclude", ".git", "--exclude", ".venv", f"{ROOT}/", f"{host}:{camp}/news-signal/"])
        if data:
            sh(["rsync", "-a", f"{need_env('NEWS_SIGNAL_DATA')}/", f"{host}:{camp}/news-signal-data/"])
        say("checkout" + (" and store" if data else "") + " synced to the cluster")

    def probe(self) -> None:
        """One short GPU job through the same submit, poll and fetch machinery: the wrapper on the fixture."""
        self.sync(data=False)
        log = self.job("probe", "$PY scripts/wrapper_probe.py\n", "PROBE_DONE", "00:10:00")
        print("\n".join(line for line in log.splitlines() if line.startswith(("WRAPPED", "PLAIN", "UMOUNT", "NESTED", "LAT", "HIDDEN"))))

    def state(self, job_id: str) -> str | None:
        out = ssh(f"sacct -n -P -X -o State -j {job_id}", tries=1)
        if not out or not out.strip():
            return None  # unknown: ssh failed or the job is not visible yet
        return out.split()[0].split("+")[0]

    def job(self, name: str, body: str, done: str, limit: str) -> str:
        """Submit (or resume) a job, wait for a terminal state, fetch its artifacts, return its log text."""
        camp = need_env("NS_CAMP")
        idfile = self.runs / f"{name}.jobid"
        if idfile.exists() and name not in self.force:
            job_id = idfile.read_text().strip()
            say(f"{name}: resuming job {job_id}")
        else:
            script = (
                "#!/bin/bash\nset -euo pipefail\n"
                f"CAMP={camp}\nsource $CAMP/jobs/vibesys-setup.sh\n"
                "export NEWS_SIGNAL_DATA=$CAMP/news-signal-data HF_HOME=$CAMP/hf HF_HUB_OFFLINE=1\n"
                f"cd $CAMP/news-signal\nPY=$CAMP/venv/bin/python\nT=.vibesys/tasks/news-signal\n{body}echo {done}\n"
            )
            path = f"{camp}/jobs/ns-{name}.sh"
            if ssh(f"mkdir -p {camp}/jobs {camp}/logs && cat > {path} && chmod +x {path}", input=script) is None:
                sys.exit("[freeze] could not write the job script on the cluster")
            reply = ssh(
                f"{need_env('NS_SBATCH')} -p {need_env('NS_PARTITION')} -N 1 -t {limit} -J ns-{name} "
                f"-o {camp}/logs/ns-{name}-%j.log {path} 2>&1"
            )
            m = re.search(r"Submitted batch job (\d+)", reply or "")
            if not m:
                sys.exit(redact(f"[freeze] sbatch gave no job id: {(reply or 'no reply').strip()[-300:]}"))
            job_id = m.group(1)
            idfile.write_text(job_id)
            say(f"{name}: submitted job {job_id} (limit {limit})")
        waited = unknown = 0
        while True:
            state = self.state(job_id)
            if state in TERMINAL:
                break
            unknown = unknown + 1 if state is None else 0
            if unknown > MAX_UNKNOWN_POLLS:
                sys.exit(f"[freeze] cannot read the state of job {job_id}; its id is kept in {idfile.name}, rerun to resume")
            if state == "PENDING" and waited > MAX_PENDING_S:
                sys.exit(f"[freeze] job {job_id} pending for 3 h: give up and report")
            time.sleep(POLL_S)
            waited += POLL_S
        log = ssh(f"cat {camp}/logs/ns-{name}-{job_id}.log")
        if log is None:
            sys.exit(f"[freeze] job {job_id} ended ({state}) but its log could not be read; rerun to retry")
        (self.runs / f"{name}.log").write_text(redact(log))
        if state != "COMPLETED" or done not in log:
            idfile.unlink()
            sys.exit(
                f"[freeze] {name} job {job_id} ended {state} without {done}; log kept as {name}.log. If it ran out "
                f"of time, raise the job time limit (now {limit}) and rerun; never change the gate sample."
            )
        for artifact in re.findall(r"ARTIFACT (\S+)", log):
            sh(["scp", "-q", f"{need_env('NS_GPU_SSH')}:{camp}/logs/{artifact}", str(self.runs / artifact)])
        idfile.unlink()
        return log

    def f1(self) -> dict:
        out = self.runs / "f1.json"
        if self.todo("f1", out):
            log = self.job(
                "f1",
                "$PY $T/accuracy_checker/checker.py --report --json-out $CAMP/logs/f1-$SLURM_JOB_ID.json\n"
                "echo ARTIFACT f1-$SLURM_JOB_ID.json\n",
                "F1_DONE",
                F1_LIMIT,
            )
            (self.runs / next(iter(re.findall(r"ARTIFACT (\S+)", log)))).replace(out)
            self.stamp(out)
        res = json.loads(out.read_text())
        sc = res["score"]
        if sc["missed"] or sc["duplicates"] or sc["invalid"] or not sc["valid"]:
            sys.exit(
                f"[freeze] baseline accuracy pass is not clean: {sc['missed']} missed, {sc['duplicates']} duplicates, "
                f"{sc['invalid']} invalid, valid={sc['valid']}. Do not change the gate sample; raise the job time "
                f"limit if it ran short, then rerun with --force f1"
            )
        return res

    def f2(self) -> list[dict]:
        paths = [self.runs / f"bench-{i}.jsonl" for i in (1, 2)]
        if any(self.todo("f2", p) for p in paths):
            log = self.job(
                "f2",
                "for i in 1 2; do\n"
                "  $PY $T/benchmark/run.py --mode bursts --vs-output $CAMP/logs/bench-$i-$SLURM_JOB_ID.jsonl\n"
                "  echo ARTIFACT bench-$i-$SLURM_JOB_ID.jsonl\n"
                "done\n",
                "F2_DONE",
                F2_LIMIT,
            )
            for src, dst in zip(re.findall(r"ARTIFACT (\S+)", log), paths, strict=True):
                (self.runs / src).replace(dst)
                self.stamp(dst)
        results = []
        for p in paths:
            last = json.loads(p.read_text().splitlines()[-1])
            if last["kind"] != "result":
                sys.exit(f"[freeze] {p.name} is not a result: {str(last.get('message'))[:300]}")
            if last["values"].get("reruns"):
                say(f"{p.name}: the first pass was invalid (send lag), the rerun is the result")
            results.append(last["values"])
        return results

    # -- freeze -----------------------------------------------------------------------------------

    def freeze(self) -> None:
        f1, (r1, r2) = self.f1(), self.f2()
        s = f1["score"]
        ic, se = s["ic"], s["ic_se"]
        p1, p2 = r1.get("p99_latency_ms"), r2.get("p99_latency_ms")
        b1, b2 = r1.get("ic"), r2.get("ic")
        require_finite(ic=ic, ic_se=se, p99_a=p1, p99_b=p2, bursts_ic_a=b1, bursts_ic_b=b2)
        floor, noise = derive(ic, se, p1, p2)
        bursts_ic = (b1 + b2) / 2
        head = git("rev-parse", "--short", "HEAD")
        other = [ln for ln in git("status", "--porcelain", "--", ".vibesys").splitlines()
                 if Path(ln[3:]).name not in TASK_OUTPUTS | {"passrun.py"}]
        dirty_note = f"; uncommitted task files: {len(other)}" if other else ""
        for key, val in (("frozen", "true"), ("ic", repr(ic)), ("ic_se", repr(se)), ("ic_floor", repr(floor)),
                         ("bursts_ic", repr(bursts_ic)), ("validation_stories", str(s["stories"])),
                         ("scored_pairs", str(s["scored_pairs"])), ("p99_ms", f"[{p1!r}, {p2!r}]"),
                         ("ms_per_headline", repr(MS_PER_HEADLINE))):  # fmt: skip
            set_toml(GATE, key, val, section="baseline")
        GATE.write_text(re.sub(
            r"\A(?:#[^\n]*\n)+", "# Frozen by scripts/freeze_baseline.py from the unmodified serving/ on the target GPU.\n",
            GATE.read_text(),
        ))
        GATE.write_text(re.sub(r"  # PLACEHOLDER[^\n]*", "", GATE.read_text()))
        set_toml(OBJECTIVES, "relative_noise", repr(noise), section="pareto")
        OBJECTIVES.write_text(OBJECTIVES.read_text().replace(
            "# PLACEHOLDER until the baseline is frozen: scripts/freeze_baseline.py sets it from the two baseline runs.",
            "# Set by scripts/freeze_baseline.py: max(0.03, the relative spread of the two baseline p99 runs, rounded up).",
        ))
        section = (
            "## Baseline\n\n"
            f"Unmodified `serving/` (plain FinBERT, one headline per call) at commit `{head}` on 1x MI210, "
            f"validation split, H = {tomllib.loads(GATE.read_text())['h_reason']}. "
            f"`serving/` tree {tree_hash(ROOT / 'serving')[:12]}, task files {task_hash()[:12]}{dirty_note}.\n\n"
            f"- Accuracy: IC {ic:.4f} (se {se:.4f}) on {s['stories']} stories, {s['scored_pairs']} scored pairs; "
            f"gate floor `ic_floor` = {floor}.\n"
            f"- Bursts: IC {bursts_ic:.4f} (mean of two runs); p99 latency {p1:.1f} ms and {p2:.1f} ms "
            f"(p50 {r1['p50_latency_ms']:.1f} and {r2['p50_latency_ms']:.1f} ms; {r1['stories']:.0f} stories). "
            f"Pareto `relative_noise` = {noise}.\n"
        )
        md = OBJECTIVE_MD.read_text()
        OBJECTIVE_MD.write_text(md[: md.index("## Baseline")] + section)
        say(f"frozen at commit {head}:")
        print(json.dumps({"ic": ic, "ic_se": se, "ic_floor": floor, "bursts_ic": bursts_ic,
                          "validation_stories": s["stories"], "scored_pairs": s["scored_pairs"],
                          "p99_ms": [p1, p2], "relative_noise": noise}, indent=2))  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--force", default="")
    a = p.parse_args()
    runs = Path(os.environ.get("NS_RUNS", Path.home() / "ns-runs"))
    f = Freezer(runs, set(filter(None, a.force.split(","))))
    steps = {"manifest": f.manifest, "pin": f.pin, "budget": f.budget, "cpu-acc": f.cpu_acc, "h": f.pick_h,
             "sizing": f.sizing, "sync": f.sync, "probe": f.probe, "f1": f.f1, "f2": f.f2, "freeze": f.freeze}  # fmt: skip
    unknown = set(a.stages.split(",")) - steps.keys()
    if unknown:
        sys.exit(f"[freeze] unknown stage {sorted(unknown)[0]}")
    for stage in [*STAGES, "probe"]:
        if stage in a.stages.split(","):
            say(f"stage {stage}")
            steps[stage]()


if __name__ == "__main__":
    main()
