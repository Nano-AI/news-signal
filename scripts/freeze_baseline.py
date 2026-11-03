"""Freeze the news-signal baseline: every real-data step of the task setup, in order, idempotent.

Run once the real store has been pulled, from the repo root of a checkout on the run host:

    NS_GPU_SSH=<ssh target of the GPU cluster> NS_CAMP=<work dir there> NS_SBATCH=<sbatch wrapper there> \\
    NS_PARTITION=<partition> NEWS_SIGNAL_DATA=<store dir here> python3 scripts/freeze_baseline.py

Stages (each skipped when its artifact exists; `--force STAGE[,STAGE]` redoes it; `--stages a,b` limits the run):

    manifest  copy the store's manifest.json into the task as data.manifest.json
    pin       write the current infosys digest into gate.toml
    budget    check the bursts replay has >= 300 stories, raising the window budget to 240 s if not
    cpu-acc   CPU dry run of the accuracy checker on every horizon
    h         pick H: the first of 5, 15, 30, 60 whose IC is more than 2 standard errors above zero
    sizing    cap MAX_STORIES so the accuracy replay takes <= 360 s at the baseline's ms per headline
    sync      rsync the checkout and the store to the cluster
    f1        GPU job: the accuracy checker on the baseline (report mode)
    f2        GPU job: the bursts benchmark, twice
    freeze    write gate.toml, objectives.toml and the OBJECTIVE.md baseline section; print the numbers

Artifacts live in $NS_RUNS (default ~/ns-runs). Stdlib only.
"""

from __future__ import annotations

import argparse
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
GATE, OBJECTIVES, PASSRUN, OBJECTIVE_MD = (TASK / n for n in ("gate.toml", "objectives.toml", "passrun.py", "OBJECTIVE.md"))
HORIZONS = (5, 15, 30, 60)
MS_PER_HEADLINE = 8.51  # FinBERT baseline, median over single-headline calls on the target GPU (install smoke)
REPLAY_BUDGET_S = 360
F1_DRAIN_S = 90  # the gate's 600 s drain would not fit a 10-minute job; the pass is checked for missed == 0
STAGES = ["manifest", "pin", "budget", "cpu-acc", "h", "sizing", "sync", "f1", "f2", "freeze"]
MAX_PENDING_S = 3 * 3600


def say(msg: str) -> None:
    print(f"[freeze] {msg}", flush=True)


def need_env(name: str) -> str:
    if not os.environ.get(name):
        sys.exit(f"[freeze] {name} is not set (see the module docstring)")
    return os.environ[name]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


def uv_run(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(cache / "news-signal" / "venv")}
    return sh(["uv", "run", "--frozen", "python", *args], cwd=ROOT, env=env, capture_output=capture)


def set_toml(path: Path, key: str, value: str, section: str | None = None) -> None:
    """Replace `key = ...` (the first one, inside `section` if given) with a literal TOML value."""
    text = path.read_text()
    start = text.index(f"[{section}]") if section else 0
    pattern = re.compile(rf"^{re.escape(key)} = .*$", re.M)
    m = pattern.search(text, start)
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


class Freezer:
    def __init__(self, runs: Path, force: set[str]):
        self.runs, self.force = runs, force
        runs.mkdir(parents=True, exist_ok=True)

    def todo(self, stage: str, artifact: Path) -> bool:
        return stage in self.force or not artifact.exists()

    # -- CPU stages -------------------------------------------------------------------------------

    def manifest(self) -> None:
        src = Path(need_env("NEWS_SIGNAL_DATA")) / "manifest.json"
        dst = TASK / "data.manifest.json"
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            dst.write_bytes(src.read_bytes())
        say("data.manifest.json matches the store manifest")

    def pin(self) -> None:
        out = sh([sys.executable, str(PASSRUN), "--print-infosys-digest"], capture_output=True).stdout.strip()
        set_toml(GATE, "infosys_sha256", toml_str(out))
        say(f"infosys digest pinned: {out[:12]}...")

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
            return int(uv_run("-c", code, capture=True).stdout.split()[-1])

        n = count()
        say(f"bursts replay: {n} measured stories at budget_s=180")
        if n < 300:
            PASSRUN.write_text(re.sub(r"^BURSTS_BUDGET_S = .*$", "BURSTS_BUDGET_S = 240.0", PASSRUN.read_text(), flags=re.M))
            n = count()
            say(f"fewer than 300 stories: BURSTS_BUDGET_S raised to 240.0, now {n}")
            if n < 300:
                sys.exit("[freeze] still fewer than 300 bursts stories at 240 s: stop and report")

    def cpu_acc(self) -> dict:
        out = self.runs / "cpu-acc.json"
        if self.todo("cpu-acc", out):
            say("CPU dry run of the accuracy checker (long)")
            uv_run(str(TASK / "accuracy_checker" / "checker.py"), "--report", "--device", "cpu", "--drain-s", "1800",
                   "--horizons", ",".join(map(str, HORIZONS)), "--json-out", str(out))
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

    def ssh(self, script: str) -> str:
        return sh(["ssh", need_env("NS_GPU_SSH"), script], capture_output=True).stdout

    def sync(self) -> None:
        host, camp = need_env("NS_GPU_SSH"), need_env("NS_CAMP")
        sh(["rsync", "-a", "--delete", "--exclude", ".git", "--exclude", ".venv", f"{ROOT}/", f"{host}:{camp}/news-signal/"])
        sh(["rsync", "-a", f"{need_env('NEWS_SIGNAL_DATA')}/", f"{host}:{camp}/news-signal-data/"])
        say("checkout and store synced to the cluster")

    def job(self, name: str, body: str, done: str) -> str:
        """Submit a <=10 min job, wait for it, fetch its artifacts, return its log text."""
        camp = need_env("NS_CAMP")
        cached = self.runs / f"{name}.log"
        script = (
            "#!/bin/bash\nset -euo pipefail\n"
            f"CAMP={camp}\nsource $CAMP/jobs/vibesys-setup.sh\n"
            "export NEWS_SIGNAL_DATA=$CAMP/news-signal-data HF_HOME=$CAMP/hf HF_HUB_OFFLINE=1\n"
            f"cd $CAMP/news-signal\nPY=$CAMP/venv/bin/python\nT=.vibesys/tasks/news-signal\n{body}echo {done}\n"
        )
        path = f"{camp}/jobs/ns-{name}.sh"
        sh(["ssh", need_env("NS_GPU_SSH"), f"cat > {path} && chmod +x {path}"], input=script)
        out = self.ssh(
            f"{need_env('NS_SBATCH')} -p {need_env('NS_PARTITION')} -N 1 -t 00:10:00 -J ns-{name} "
            f"-o {camp}/logs/ns-{name}-%j.log {path} 2>&1"
        )
        job_id = re.findall(r"\d+", out.strip().splitlines()[-1])[-1]
        say(f"{name}: submitted job {job_id}")
        waited = 0
        while True:
            queue = self.ssh(f"squeue -h -j {job_id} -o '%T' 2>/dev/null || true").strip()
            if not queue:
                break
            if queue == "PENDING" and waited > MAX_PENDING_S:
                sys.exit(f"[freeze] job {job_id} pending for 3 h: give up and report")
            time.sleep(60)
            waited += 60
        log = self.ssh(f"cat {camp}/logs/ns-{name}-{job_id}.log")
        cached.write_text(log)
        if done not in log:
            sys.exit(f"[freeze] {name} did not finish ({done} missing); log kept at {cached}")
        for artifact in re.findall(r"ARTIFACT (\S+)", log):
            sh(["scp", "-q", f"{need_env('NS_GPU_SSH')}:{camp}/logs/{artifact}", str(self.runs / artifact)])
        return log

    def f1(self) -> dict:
        out = self.runs / "f1.json"
        if self.todo("f1", out):
            log = self.job(
                "f1",
                f"$PY $T/accuracy_checker/checker.py --report --drain-s {F1_DRAIN_S} --json-out $CAMP/logs/f1-$SLURM_JOB_ID.json\n"
                "echo ARTIFACT f1-$SLURM_JOB_ID.json\n",
                "F1_DONE",
            )
            (self.runs / next(a for a in re.findall(r"ARTIFACT (\S+)", log))).replace(out)
        res = json.loads(out.read_text())
        if res["score"]["missed"] or res["score"]["duplicates"] or res["score"]["invalid"] or not res["score"]["valid"]:
            sys.exit(f"[freeze] baseline accuracy pass is not clean: {res['score']['missed']} missed, "
                     f"{res['score']['duplicates']} duplicates, valid={res['score']['valid']}; lower MAX_STORIES or rerun --force f1")
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
            )
            for src, dst in zip(re.findall(r"ARTIFACT (\S+)", log), paths, strict=True):
                (self.runs / src).replace(dst)
        results = []
        for p in paths:
            last = json.loads(p.read_text().splitlines()[-1])
            if last["kind"] != "result":
                sys.exit(f"[freeze] {p.name} is not a result: {last.get('message')}")
            results.append(last["values"])
        return results

    # -- freeze -----------------------------------------------------------------------------------

    def freeze(self) -> None:
        f1, (r1, r2) = self.f1(), self.f2()
        s = f1["score"]
        ic, se = s["ic"], s["ic_se"]
        p1, p2 = r1["p99_latency_ms"], r2["p99_latency_ms"]
        floor, noise = derive(ic, se, p1, p2)
        for key, val in (("frozen", "true"), ("ic", repr(ic)), ("ic_se", repr(se)), ("ic_floor", repr(floor)),
                         ("validation_stories", str(s["stories"])), ("scored_pairs", str(s["scored_pairs"])),
                         ("p99_ms", f"[{p1!r}, {p2!r}]"), ("ms_per_headline", repr(MS_PER_HEADLINE))):
            set_toml(GATE, key, val, section="baseline")
        text = GATE.read_text()
        GATE.write_text(re.sub(r"\A(?:#[^\n]*\n)+", "# Frozen by scripts/freeze_baseline.py from the unmodified serving/ on the target GPU.\n", text))
        set_toml(OBJECTIVES, "relative_noise", repr(noise), section="pareto")
        OBJECTIVES.write_text(OBJECTIVES.read_text().replace("# PLACEHOLDER until the baseline is frozen: scripts/freeze_baseline.py sets it from the two baseline runs.", "# Set by scripts/freeze_baseline.py: max(0.03, the relative spread of the two baseline p99 runs, rounded up)."))
        section = (
            "## Baseline\n\n"
            f"Unmodified `serving/` (plain FinBERT, one headline per call) on 1x MI210, validation split, H = "
            f"{tomllib.loads(GATE.read_text())['h_reason']}.\n\n"
            f"- Accuracy: IC {ic:.4f} (se {se:.4f}) on {s['stories']} stories, {s['scored_pairs']} scored pairs; "
            f"gate floor `ic_floor` = {floor}.\n"
            f"- Bursts p99 latency: {p1:.1f} ms and {p2:.1f} ms on two runs "
            f"(p50 {r1['p50_latency_ms']:.1f} and {r2['p50_latency_ms']:.1f} ms; "
            f"{r1['stories']:.0f} stories). Pareto `relative_noise` = {noise}.\n"
        )
        md = OBJECTIVE_MD.read_text()
        OBJECTIVE_MD.write_text(md[: md.index("## Baseline")] + section)
        say("frozen:")
        print(json.dumps({"ic": ic, "ic_se": se, "ic_floor": floor, "validation_stories": s["stories"],
                          "scored_pairs": s["scored_pairs"], "p99_ms": [p1, p2], "relative_noise": noise}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--force", default="")
    a = p.parse_args()
    runs = Path(os.environ.get("NS_RUNS", Path.home() / "ns-runs"))
    f = Freezer(runs, set(filter(None, a.force.split(","))))
    steps = {"manifest": f.manifest, "pin": f.pin, "budget": f.budget, "cpu-acc": f.cpu_acc, "h": f.pick_h,
             "sizing": f.sizing, "sync": f.sync, "f1": f.f1, "f2": f.f2, "freeze": f.freeze}
    unknown = set(a.stages.split(",")) - steps.keys()
    if unknown:
        sys.exit(f"[freeze] unknown stage {sorted(unknown)[0]}")
    for stage in STAGES:
        if stage in a.stages.split(","):
            say(f"stage {stage}")
            steps[stage]()
    # the digest covers infosys/ only, so edits above (passrun.py, gate.toml) never invalidate the pin


if __name__ == "__main__":
    main()
