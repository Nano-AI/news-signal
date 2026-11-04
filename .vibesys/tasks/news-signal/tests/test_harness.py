"""Harness tests on the synthetic fixture (CPU). Everything real runs in subprocesses of the pass runner."""

import json
import math
import dataclasses
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import infosys  # noqa: F401  imported first so a temp-copy run_pass never shadows the real package

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
for sub in ("", "benchmark", "accuracy_checker", "cpu_check"):
    sys.path.insert(0, str(TASK / sub))

import checker  # noqa: E402
import passrun  # noqa: E402
import run as bench  # noqa: E402
from passrun import Mode, PassError  # noqa: E402
from tiny_model import build_tiny_model  # noqa: E402

from infosys.score import PassScore  # noqa: E402

BOUND_S = 300  # only guards against hangs


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(ROOT, dest, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache"))
    return dest


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny") / "model")


def _score(**kw) -> PassScore:
    base = dict(
        stories=100, pairs=1000, signaled=1000, missed=0, duplicates=0, invalid=0,
        latency_ms={"p50": 5.0, "p90": 9.0, "p99": 20.0, "max": 30.0}, throughput_per_s=50.0,
        scored_pairs=800, ic=0.05, ic_se=0.01, hit_rate=0.52, send_lag_ms={"p50": 0.2, "p90": 0.3, "p99": 0.4, "max": 0.9},
    )  # fmt: skip
    return PassScore(**{**base, **kw})


DIRS = {-1, 0, 1}


def _gate(frozen=True, floor=0.03) -> dict:
    return {
        "max_missed_frac": 0.001, "max_duplicate_frac": 0.001, "max_invalid": 0,
        "baseline": {"frozen": frozen, "ic_floor": floor, "bursts_ic": 0.05},
    }  # fmt: skip


# -- pins and gate file ---------------------------------------------------------------------------------


def test_infosys_pin_matches_tree():
    assert passrun.infosys_digest(ROOT) == passrun.load_gate()["infosys_sha256"]


def test_repo_pin_matches_tree():
    assert passrun.repo_digest(ROOT) == passrun.load_gate()["repo_sha256"]


def test_gate_toml_complete():
    gate = passrun.load_gate()
    assert isinstance(gate["h_minutes"], int) and gate["h_minutes"] > 0
    assert len(gate["infosys_sha256"]) == 64
    base = gate["baseline"]
    assert isinstance(base["frozen"], bool) and isinstance(base["bursts_ic"], float)
    assert all(isinstance(v, float) for v in base["p99_ms"]) and len(base["p99_ms"]) == 2
    if base["frozen"]:
        assert base["ic_floor"] <= base["ic"] and base["validation_stories"] > 0 and base["scored_pairs"] > 0


def test_digest_ignores_caches_and_sees_edits(tmp_path):
    (tmp_path / "infosys" / "__pycache__").mkdir(parents=True)
    (tmp_path / "infosys" / "a.py").write_text("x = 1\n")
    before = passrun.infosys_digest(tmp_path)
    (tmp_path / "infosys" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"junk")
    assert passrun.infosys_digest(tmp_path) == before
    (tmp_path / "infosys" / "a.py").write_text("x = 2\n")
    assert passrun.infosys_digest(tmp_path) != before


# -- candidate checks -----------------------------------------------------------------------------------


def test_weights_rejected(repo):
    (repo / "serving" / "w.safetensors").write_bytes(b"x")
    with pytest.raises(PassError) as e:
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    assert e.value.requirement == "no_weights_in_git" and "serving/w.safetensors" in e.value.message


def test_large_file_rejected_but_fixtures_and_vibesys_are_not(tmp_path):
    (tmp_path / "infosys").mkdir()
    (tmp_path / "infosys" / "data.npz").write_bytes(b"x")
    (tmp_path / ".vibesys").mkdir()
    (tmp_path / ".vibesys" / "w.bin").write_bytes(b"x")
    passrun.check_candidate_files(tmp_path)
    (tmp_path / "blob.dat").write_bytes(bytes(passrun.MAX_FILE_BYTES + 1))
    with pytest.raises(PassError) as e:
        passrun.check_candidate_files(tmp_path)
    assert e.value.requirement == "no_weights_in_git" and "blob.dat" in e.value.message


def test_infosys_edit_rejected(repo):
    with (repo / "infosys" / "score.py").open("ab") as f:
        f.write(b"\n")
    with pytest.raises(PassError) as e:
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    assert e.value.requirement == "infosys_unchanged"


def test_missing_data_env_names_the_variable(monkeypatch):
    monkeypatch.delenv("NEWS_SIGNAL_DATA", raising=False)
    with pytest.raises(PassError, match="NEWS_SIGNAL_DATA is not set"):
        passrun.run_pass(Mode.ACCURACY, "validation", device="cpu")


# -- serving process handling ---------------------------------------------------------------------------


def test_serving_crash_fails_fast(repo):
    (repo / "serving" / "__main__.py").write_text("raise SystemExit(3)\n")
    start = time.monotonic()
    with pytest.raises(PassError, match="exited with code 3"):
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    assert time.monotonic() - start < 60  # the ready timeout is 120 s


def test_env_sanitized(repo, tmp_path, monkeypatch):
    dump, extra = tmp_path / "env.json", tmp_path / "extra.json"
    (repo / "serving" / "__main__.py").write_text(
        "import json, os\n"
        f"open({str(dump)!r}, 'w').write(json.dumps(sorted(os.environ)))\n"
        "try:\n    open('/proc/%d/environ' % os.getppid(), 'rb').read(); parent = 'readable'\n"
        "except OSError:\n    parent = 'denied'\n"
        f"open({str(extra)!r}, 'w').write(json.dumps({{'nice': os.nice(0), 'parent': parent}}))\n"
    )
    base_nice = os.nice(0)
    monkeypatch.setenv("NEWS_SIGNAL_DATA", "/nowhere")
    monkeypatch.setenv("NEWS_SIGNAL_OTHER", "x")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(PassError):  # exits 0 without ever calling /ready
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    names = json.loads(dump.read_text())
    assert not [n for n in names if n.startswith("NEWS_SIGNAL_")]
    assert "HF_HUB_OFFLINE" in names
    seen = json.loads(extra.read_text())
    assert seen["nice"] == min(19, base_nice + passrun.SERVING_NICE)  # serving runs at lower priority
    assert seen["parent"] == "denied"  # the harness is not dumpable: its /proc files are closed to the child


def test_prepare_failure_is_reported(repo):
    (repo / "serving" / "prepare.py").write_text("import sys\nprint('boom')\nsys.exit(4)\n")
    with pytest.raises(PassError, match="prepare exited with code 4: boom"):
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")


@pytest.fixture(scope="module")
def smoke(tiny_model):
    readings = iter([3 << 20])  # 3 MiB before the spawn, 7 MiB at every later reading (the end of the pass included)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(passrun, "read_vram_used_bytes", lambda: next(readings, 7 << 20))
        return passrun.run_pass(Mode.SMOKE, "validation", device="cpu", model=str(tiny_model))


def test_smoke_pass_records_cold_start_gpu_memory_and_per_feed_latency(smoke):
    from infosys.replay import FEEDS

    assert 0 < smoke.serve.cold_start_s < BOUND_S
    assert (smoke.serve.gpu_mem_idle_mib, smoke.serve.gpu_mem_peak_mib) == (3.0, 7.0)
    feeds = smoke.score.feed_latency_ms
    assert set(feeds) == {f.name for f in FEEDS}
    assert sum(f["first"] for f in feeds.values()) == smoke.score.stories
    real = [f.name for f in FEEDS if f.real]  # real feeds never drop and split the stories by source
    assert sum(feeds[n]["pairs"] for n in real) == smoke.score.signaled


def test_smoke_pass_loses_no_early_story(smoke):
    """Real newsd and real `python -m serving`: ready is sent only once every subscription is live."""
    s = smoke.score
    assert (s.missed, s.duplicates, s.invalid) == (0, 0, 0)
    first = smoke.plan.measured[0].story_id
    assert first in smoke.log.sent_s
    assert any(sig.story_id == first for sig in smoke.log.signals)


def test_cpu_check_passes():
    out = subprocess.run(
        [sys.executable, str(TASK / "cpu_check" / "check.py")], cwd=ROOT, capture_output=True, text=True,
        timeout=BOUND_S,
    )  # fmt: skip
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASS" in out.stdout


# -- gates refuse invalid passes and bad counts ---------------------------------------------------------


@settings(deadline=None)
@given(
    ic=st.one_of(st.none(), st.floats(-0.2, 0.2)),
    missed=st.integers(0, 5), dup=st.integers(0, 5), invalid=st.integers(0, 2), valid=st.booleans(),
    frozen=st.booleans(),
)  # fmt: skip
def test_gate_passes_exactly_when_all_checks_hold(ic, missed, dup, invalid, valid, frozen):
    score = _score(
        ic=ic, missed=missed, duplicates=dup, invalid=invalid, valid=valid,
        invalid_reason=None if valid else "send lag p90 9.0 ms exceeds 5.0 ms",
    )  # fmt: skip
    expected = frozen and valid and ic is not None and ic >= 0.03 and missed <= 1 and dup <= 1 and invalid == 0
    assert (checker.evaluate_gate(score, _gate(frozen=frozen)) is None) == expected


def test_gate_names_the_invalid_pass():
    msg = checker.evaluate_gate(_score(valid=False, invalid_reason="send lag p90 9.0 ms exceeds 5.0 ms"), _gate())
    assert msg is not None and "send lag" in msg


def test_gate_mode_refuses_report_only_flags(capsys):
    for flags in (["--drain-s", "1"], ["--horizons", "5"], ["--split", "holdout"]):
        assert checker.main(flags) == 1
        assert "gate mode refuses" in capsys.readouterr().out


def test_unfrozen_baseline_fails_gate_without_running(capsys, monkeypatch):
    monkeypatch.setattr(checker, "load_gate", lambda: {"baseline": {"frozen": False}})
    assert checker.main([]) == 1
    assert checker.NOT_FROZEN in capsys.readouterr().out


# -- benchmark record stream ----------------------------------------------------------------------------


def _check_stream(lines: list[str]) -> dict:
    """The reader obligations of sdk/vs-evaluator/PROTOCOL.md that this writer can break."""
    recs = [json.loads(line) for line in lines]
    assert recs[0]["kind"] == "hello" and recs[0]["protocol"] == 2
    assert [r["kind"] for r in recs].count("hello") == 1
    metrics = recs[0]["metrics"]
    assert metrics and all(n and not any(c.isspace() for c in n) for n in metrics)
    for spec in metrics.values():
        assert set(spec) <= {"unit", "direction", "required"}
        assert spec.get("direction", "min") in ("min", "max")
    outcome = [r for r in recs[1:] if r["kind"] in ("result", "error")]
    assert len(outcome) == 1 and recs[-1] is outcome[0]
    r = outcome[0]
    if r["kind"] == "result":
        assert set(r) == {"kind", "label", "values"} and r["label"] == ""
        assert set(r["values"]) <= set(metrics)
        assert {n for n, s in metrics.items() if s.get("required", True)} <= set(r["values"])
        assert all(type(v) in (int, float) and math.isfinite(v) for v in r["values"].values())
    else:
        assert set(r) <= {"kind", "message", "partial", "error_kind", "requirement"} and r["message"]
        assert ("requirement" in r) == (r.get("error_kind") == "requirement_not_met")
        if "requirement" in r:
            assert r["requirement"] and not any(c.isspace() for c in r["requirement"])
        if "partial" in r:
            p = r["partial"]
            assert set(p) <= {"name", "value", "direction", "unit", "target", "progress"}
            assert math.isfinite(p["value"]) and p["direction"] in ("min", "max")
    return r


def _write(tmp_path, records) -> list[str]:
    path = tmp_path / "out.jsonl"
    bench.write_records(path, records)
    return path.read_text().splitlines()


def test_vs_output_records(tmp_path):
    gate = _gate()
    ok = _check_stream(_write(tmp_path, bench.records_for(_score(), gate, DIRS)))
    assert ok["kind"] == "result" and ok["values"]["p99_latency_ms"] == 20.0
    for bad in (
        _score(missed=5), _score(duplicates=5), _score(invalid=1),
        _score(valid=False, invalid_reason="send lag p90 9.0 ms exceeds 5.0 ms"),
        _score(latency_ms={"p50": None, "p90": None, "p99": None, "max": None}, missed=1000, signaled=0),
    ):  # fmt: skip
        err = _check_stream(_write(tmp_path, bench.records_for(bad, gate, DIRS)))
        assert err["kind"] == "error" and "error_kind" not in err
    named = bench.records_for(_score(missed=5), gate, DIRS)[-1]
    assert named["message"].startswith("missed 5") and named["partial"]["value"] == 20.0
    invalid = bench.records_for(_score(valid=False, invalid_reason="send lag"), gate, DIRS)[-1]
    assert "partial" not in invalid  # an invalid pass's p99 is not to be trusted
    req = bench.error_record("weights", requirement="no_weights_in_git")
    assert _check_stream(_write(tmp_path, [bench.HELLO, req]))["error_kind"] == "requirement_not_met"


def test_benchmark_main_turns_pass_error_into_record(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise PassError("infosys/ differs", requirement="infosys_unchanged")

    monkeypatch.setattr(bench, "run_pass", boom)
    monkeypatch.setattr(bench, "load_gate", _gate)
    out = tmp_path / "o.jsonl"
    assert bench.main(["--mode", "bursts", "--vs-output", str(out)]) == 1
    err = _check_stream(out.read_text().splitlines())
    assert err["requirement"] == "infosys_unchanged" and err["error_kind"] == "requirement_not_met"


def test_benchmark_quality_gate():
    gate = _gate()
    assert bench.records_for(_score(), gate, {1})[-1]["message"].startswith("prediction quality")
    low = bench.records_for(_score(ic=0.0, ic_se=0.01), gate, DIRS)[-1]
    assert low["kind"] == "error" and "ic 0.0000" in low["message"] and low["partial"]["name"] == "p99_latency_ms"
    assert bench.records_for(_score(ic=None), gate, DIRS)[-1]["kind"] == "error"
    assert bench.records_for(_score(ic=0.05 - 2 * 0.01), gate, DIRS)[-1]["kind"] == "result"  # within 3 se
    unfrozen = _gate(frozen=False)  # the baseline run that freezes bursts_ic
    ok = bench.records_for(_score(ic=0.0), unfrozen, DIRS)[-1]
    assert ok["kind"] == "result" and ok["values"]["ic"] == 0.0
    assert bench.records_for(_score(), unfrozen, {0})[-1]["kind"] == "error"


def test_benchmark_main_turns_any_exception_into_a_data_free_record(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("story text and /secret/path")

    monkeypatch.setattr(bench, "run_pass", boom)
    monkeypatch.setattr(bench, "load_gate", _gate)
    out = tmp_path / "o.jsonl"
    assert bench.main(["--mode", "bursts", "--vs-output", str(out)]) == 1
    err = _check_stream(out.read_text().splitlines())
    assert "OSError" in err["message"] and "story text" not in err["message"] and "error_kind" not in err


def test_checker_reports_bad_horizons_without_running(capsys):
    assert checker.main(["--report", "--horizons", "5,x"]) == 1
    assert "--horizons must be" in capsys.readouterr().out


# -- failure paths of the pass runner -------------------------------------------------------------------


def _fast_ready(monkeypatch, seconds=10.0):  # a hang guard: the tests end at this timeout, whatever the host's speed
    smoke = passrun.SETTINGS[Mode.SMOKE]
    monkeypatch.setitem(passrun.SETTINGS, Mode.SMOKE, dataclasses.replace(smoke, ready_timeout_s=seconds))


def test_log_tail_is_bounded(repo):
    (repo / "serving" / "__main__.py").write_text(
        "import sys\nfor i in range(100):\n    print('x' * 5000)\nsys.exit(3)\n"
    )
    with pytest.raises(PassError) as e:
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    lines = e.value.message.splitlines()
    assert len(lines) <= passrun._LOG_TAIL_LINES and max(map(len, lines)) <= passrun._LOG_TAIL_CHARS + 100


def test_serving_that_never_becomes_ready_times_out_with_its_log(repo, monkeypatch):
    _fast_ready(monkeypatch)
    (repo / "serving" / "__main__.py").write_text("import time\nprint('loading', flush=True)\ntime.sleep(600)\n")
    with pytest.raises(PassError, match="never became ready.*loading"):
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")


def test_serving_that_ignores_sigterm_is_killed(repo, tmp_path, monkeypatch):
    _fast_ready(monkeypatch)
    monkeypatch.setattr(passrun, "_STOP_GRACE_S", 0.5)
    pidfile = tmp_path / "pid"
    (repo / "serving" / "__main__.py").write_text(
        "import os, signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(600)\n"
    )
    with pytest.raises(PassError):
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_prepare_timeout_is_reported(repo, monkeypatch):
    monkeypatch.setattr(passrun, "PREPARE_TIMEOUT_S", 1)
    (repo / "serving" / "prepare.py").write_text("import time\ntime.sleep(600)\n")
    with pytest.raises(PassError, match="prepare exceeded 1 s"):
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")


def test_store_that_is_not_the_pinned_one_is_refused():
    fixture = ROOT / "infosys" / "fixtures" / "tiny"
    with pytest.raises(PassError, match="does not match the pinned manifest"):  # the pending manifest pins nothing
        passrun.run_pass(Mode.ACCURACY, "validation", data_dir=fixture, device="cpu")


def test_repo_pin_edit_rejected(repo):
    with (repo / "pyproject.toml").open("a") as f:
        f.write("# edited\n")
    with pytest.raises(PassError) as e:
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    assert e.value.requirement == "infosys_unchanged"


def test_symlink_out_of_the_tree_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    tree = tmp_path / "tree"
    (tree / "serving").mkdir(parents=True)
    (tree / "serving" / "w").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PassError, match="symlink") as e:
        passrun.check_candidate_files(tree)
    assert e.value.requirement == "no_weights_in_git"


# -- baseline freeze arithmetic and failure paths -------------------------------------------------------


@pytest.fixture
def freezer(repo, tmp_path):
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)
    import importlib.util

    spec = importlib.util.spec_from_file_location("fz_copy", repo / "scripts" / "freeze_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = tmp_path / "runs"
    return module, module.Freezer(runs, set()), runs, repo


def _artifacts(fz, runs, *, ic=0.04125, clean=True, bursts_ic=0.03, lag_p90=1.0, sidecars=True):
    score = {"stories": 1000, "scored_pairs": 700, "ic": ic, "ic_se": 0.01, "missed": 0 if clean else 3,
             "duplicates": 0, "invalid": 0, "valid": True,
             "latency_ms": {"p50": 5.0, "p90": 9.0, "p99": 20.0, "max": 30.0}, "throughput_per_s": 50.0,
             "send_lag_ms": {"p50": 0.2, "p90": 0.3, "p99": 0.4, "max": 0.9},
             "feed_latency_ms": FEEDS_STATS if sidecars else {}}  # fmt: skip
    serve = {"cold_start_s": 12.5, "gpu_mem_idle_mib": 300.0, "gpu_mem_peak_mib": 4096.0} if sidecars else {}
    (runs / "f1.json").write_text(json.dumps({"score": score, "serve": serve}))
    (runs / "cpu-acc.json").write_text(json.dumps({"horizons": {}}))
    fz.stamp(runs / "f1.json")
    for i, p99 in enumerate((100.0, 104.0), start=1):
        values = {"p99_latency_ms": p99, "p50_latency_ms": 10.0, "stories": 300, "ic": bursts_ic,
                  "newsd_send_lag_p50_ms": 0.7, "newsd_send_lag_p90_ms": lag_p90, "newsd_send_lag_p99_ms": 20.0,
                  "newsd_send_lag_max_ms": 25.0, "p90_latency_ms": 50.0, "max_latency_ms": 30.0,
                  "throughput_signals_per_s": 50.0}  # fmt: skip
        rec = {"kind": "result", "label": "", "values": values}
        (runs / f"bench-{i}.jsonl").write_text(json.dumps(bench.HELLO) + "\n" + json.dumps(rec) + "\n")
        fz.stamp(runs / f"bench-{i}.jsonl")
        if sidecars:
            (runs / f"bench-{i}.jsonl.stats.json").write_text(json.dumps({"feeds": FEEDS_STATS, "serve": serve}))


def test_freeze_rewrites_task_files_idempotently(freezer):
    import tomllib

    module, fz, runs, repo = freezer
    _artifacts(fz, runs)
    task = repo / ".vibesys" / "tasks" / "news-signal"

    def freeze() -> dict:
        fz.freeze()
        return {n: (task / n).read_bytes() for n in ("gate.toml", "objectives.toml", "OBJECTIVE.md")}

    first = freeze()
    base = tomllib.loads(first["gate.toml"].decode())["baseline"]
    assert base["frozen"] is True and base["ic_floor"] == 0.0212 and base["p99_ms"] == [100.0, 104.0]
    assert base["bursts_ic"] == 0.03 and base["validation_stories"] == 1000 and base["scored_pairs"] == 700
    assert b"PLACEHOLDER" not in first["gate.toml"]
    assert tomllib.loads(first["objectives.toml"].decode())["pareto"]["relative_noise"] == 0.04
    assert b"IC 0.04" in first["OBJECTIVE.md"] and b"Pending" not in first["OBJECTIVE.md"] and b"commit `" in first["OBJECTIVE.md"]
    assert freeze() == first


def test_freeze_refuses_null_ic_and_unclean_f1(freezer):
    module, fz, runs, _ = freezer
    _artifacts(fz, runs, ic=None)
    with pytest.raises(SystemExit, match="not a finite number"):
        fz.freeze()
    _artifacts(fz, runs, clean=False)
    with pytest.raises(SystemExit, match="never change the gate sample|Do not change the gate sample"):
        fz.freeze()


def test_stale_artifact_is_redone_and_missing_one_needs_the_cluster(freezer, monkeypatch):
    module, fz, runs, repo = freezer
    for name in ("NS_GPU_SSH", "NS_CAMP", "NS_SBATCH", "NS_PARTITION"):
        monkeypatch.delenv(name, raising=False)
    _artifacts(fz, runs)
    assert not fz.todo("f1", runs / "f1.json")
    module.set_toml(repo / ".vibesys" / "tasks" / "news-signal" / "gate.toml", "h_minutes", "15")
    assert fz.todo("f1", runs / "f1.json")  # made at another H
    with pytest.raises(SystemExit, match="is not set"):
        fz.f1()


def test_freeze_refuses_a_dirty_serving_tree(freezer):
    _, fz, _, repo = freezer
    (repo / "serving" / "model.py").write_text("# edited\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        fz.sync()


def test_freeze_job_resumes_and_does_not_mistake_a_failed_poll_for_done(freezer, monkeypatch):
    module, fz, runs, _ = freezer
    for name, val in (("NS_GPU_SSH", "host.example"), ("NS_CAMP", "/camp/dir"), ("NS_SBATCH", "sb"), ("NS_PARTITION", "p")):
        monkeypatch.setenv(name, val)
    calls, states = [], iter(["RUNNING", None, "RUNNING", "COMPLETED"])

    def fake_ssh(script, *, input=None, tries=3):
        calls.append(script.split()[0])
        if script.startswith("sb "):
            return "Submitted batch job 77\n"
        if script.startswith("sacct"):
            s = next(states)
            return None if s is None else s + "\n"
        if script.startswith("cat /camp"):
            return "F1_DONE\nARTIFACT a.json\n"
        return ""

    monkeypatch.setattr(module, "ssh", fake_ssh)
    monkeypatch.setattr(module, "sh", lambda cmd, **kw: (runs / "a.json").write_text("{}"))
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    log = fz.job("f1", "", "F1_DONE", "00:10:00")
    assert "F1_DONE" in log and calls.count("sacct") == 4 and not (runs / "f1.jobid").exists()
    # an interrupted wait resumes the recorded job instead of submitting again
    (runs / "f1.jobid").write_text("77")
    states = iter(["COMPLETED"])
    calls.clear()
    fz.job("f1", "", "F1_DONE", "00:10:00")
    assert "sb" not in calls


def test_freeze_job_rejects_a_reply_without_a_job_id_and_hides_the_host(freezer, monkeypatch, capsys):
    module, fz, runs, _ = freezer
    for name, val in (("NS_GPU_SSH", "host.example"), ("NS_CAMP", "/camp/dir"), ("NS_SBATCH", "sb"), ("NS_PARTITION", "p")):
        monkeypatch.setenv(name, val)
    monkeypatch.setattr(module, "ssh", lambda script, *, input=None, tries=3: "" if "cat >" in script else "error on host.example /camp/dir\n")
    with pytest.raises(SystemExit) as e:
        fz.job("f1", "", "F1_DONE", "00:10:00")
    assert "no job id" in str(e.value) and "host.example" not in str(e.value) and "/camp/dir" not in str(e.value)


def _load_freeze():
    import importlib.util

    spec = importlib.util.spec_from_file_location("freeze_baseline", ROOT / "scripts" / "freeze_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@settings(deadline=None)
@given(ic=st.floats(-0.5, 0.5), se=st.floats(0, 0.1), a=st.floats(0.1, 1e4), b=st.floats(0.1, 1e4))
def test_freeze_derivation(ic, se, a, b):
    floor, noise = _FREEZE.derive(ic, se, a, b)
    assert floor <= ic - 2 * se + 1e-12 and ic - 2 * se - floor < 1.1e-4
    assert noise >= 0.03 and noise + 1e-9 >= abs(a - b) / ((a + b) / 2)


def test_freeze_set_toml_edits_only_the_named_section(tmp_path):
    path = tmp_path / "g.toml"
    path.write_text("ic = 1\n\n[baseline]\nic = 0.0\nfrozen = false\n")
    _FREEZE.set_toml(path, "ic", "0.25", section="baseline")
    _FREEZE.set_toml(path, "frozen", "true", section="baseline")
    assert path.read_text() == "ic = 1\n\n[baseline]\nic = 0.25\nfrozen = true\n"


_FREEZE = _load_freeze()


# -- scoring data hidden from the candidate (real-data passes) -------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    dest = tmp_path / "store"
    shutil.copytree(ROOT / "infosys" / "fixtures" / "tiny", dest)
    monkeypatch.setattr(passrun, "MANIFEST_PATH", dest / "manifest.json")
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "marker.txt").write_text("cache")
    monkeypatch.setenv("HF_HOME", str(hf))
    return dest


_PROBE = (
    "import json, os, sys\n"
    "data = {data!r}\n"
    "out = {{}}\n"
    "out['listdir'] = os.listdir(data)\n"
    "try:\n    open(os.path.join(data, 'stories-validation.jsonl.gz'), 'rb'); out['open'] = 'readable'\n"
    "except OSError:\n    out['open'] = 'denied'\n"
    "out['hf'] = open(os.path.join(os.environ['HF_HOME'], 'marker.txt')).read()\n"
    "{extra}"
    "open({dump!r}, 'w').write(json.dumps(out))\n"
)


def test_candidate_cannot_see_the_scoring_data_but_keeps_what_it_needs(repo, store, tmp_path):
    prep_dump = tmp_path / "prep.json"
    train_read = "i = sys.argv.index('--train') + 1\nout['train_bytes'] = os.path.getsize(sys.argv[i])\n"
    (repo / "serving" / "prepare.py").write_text(
        _PROBE.format(data=str(store), extra=train_read, dump=str(prep_dump))
        + "os.makedirs(sys.argv[sys.argv.index('--out') + 1], exist_ok=True)\n"
    )
    # serving's filesystem is read-only, so it reports through its exit code and the log tail
    extra = "out['tree'] = os.path.isfile('serving/model.py')\nprint(json.dumps(out)); sys.exit(3)\n"
    (repo / "serving" / "__main__.py").write_text(_PROBE.format(data=str(store), extra=extra, dump="/dev/null"))
    with pytest.raises(PassError, match="exited with code 3") as e:
        passrun.run_pass(Mode.ACCURACY, "validation", root=repo, data_dir=store, device="cpu")
    serve = json.loads(e.value.message.split("pass ended: ", 1)[1])
    for seen in (json.loads(prep_dump.read_text()), serve):
        assert seen["listdir"] == [] and seen["open"] == "denied"  # the data dir is an empty tmpfs
        assert seen["hf"] == "cache"
    assert json.loads(prep_dump.read_text())["train_bytes"] > 0  # the train export is readable
    assert serve["tree"] is True  # so is the serving tree
    assert (store / "stories-validation.jsonl.gz").is_file()  # the mount was private to the child


def _wrapper(tmp_path, code, protect=()):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "secret.txt").write_text("validation headline")
    cmd = passrun.hidden([sys.executable, "-c", code], data, list(protect))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60), data


def test_candidate_cannot_unmount_or_reveal_the_hidden_data(tmp_path):
    code = (
        "import ctypes, os, subprocess\n"
        f"d = {str(tmp_path / 'data')!r}\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "print('umount', libc.umount2(d.encode(), 0), ctypes.get_errno(), os.listdir(d))\n"
        "r = subprocess.run(['unshare', '-Urm', 'sh', '-c', 'umount ' + d + '; ls ' + d], capture_output=True, text=True)\n"
        "print('nested', r.returncode != 0 or r.stdout.strip() == '', 'secret.txt' in r.stdout)\n"
        "st = dict(l.split(':\\t') for l in open('/proc/self/status') if l.startswith(('CapEff', 'NoNewPrivs')))\n"
        "print('capeff', int(st['CapEff'], 16), 'nnp', st['NoNewPrivs'].strip())\n"
    )
    (tmp_path / "w").mkdir()
    r, data = _wrapper(tmp_path, code, [tmp_path / "w"])
    assert r.returncode == 0, r.stderr
    assert "umount -1 1 []" in r.stdout  # EPERM, and the directory is still empty
    assert "nested True False" in r.stdout  # a nested namespace gains nothing
    assert "capeff 0 nnp 1" in r.stdout  # no capabilities, no new privileges
    assert (data / "secret.txt").is_file()  # outside the namespace nothing changed


def test_writes_do_not_survive_the_attempt(tmp_path, monkeypatch):
    home, hf, work, repo_dir = (tmp_path / n for n in ("home", "hf", "work", "checkout"))
    for d in (home, hf, work, repo_dir):
        d.mkdir()
    token = f"memo-{os.getpid()}-{time.monotonic_ns()}"  # a headline-keyed memo file, as an agent would write
    targets = {"home": home, "hf": hf, "work": work, "repo": repo_dir, "tmp": Path("/tmp"),
               "vartmp": Path("/var/tmp"), "shm": Path("/dev/shm")}  # fmt: skip
    code = (
        "import json, os\n"
        f"targets = {{k: v for k, v in {({k: str(v) for k, v in targets.items()})!r}.items()}}\n"
        "out = {}\n"
        "for k, p in targets.items():\n"
        f"    try:\n        open(os.path.join(p, {token!r}), 'w').write('x'); out[k] = 'written'\n"
        "    except OSError as e:\n        out[k] = e.errno\n"
        "tmp = os.environ['TMPDIR']\n"
        "open(os.path.join(tmp, 'scratch'), 'w').write('x')\n"
        "out['scratch_env'] = [tmp.startswith('/dev/shm'), os.environ['XDG_CACHE_HOME'].startswith(tmp)]\n"
        "print(json.dumps(out))\n"
    )
    protect = passrun.protected_paths(repo_dir, work, {"HOME": str(home), "HF_HOME": str(hf)})
    assert Path("/tmp") in protect  # the test dirs live under /tmp, so one bind covers them
    for d in (home, hf, work, repo_dir):
        assert any(d.resolve().is_relative_to(p) for p in protect)
    for _ in range(2):  # the second attempt must find nothing from the first
        r, _ = _wrapper(tmp_path, code, protect)
        assert r.returncode == 0, r.stderr
        seen = json.loads(r.stdout.strip().splitlines()[-1])
        assert all(seen[k] == 30 for k in ("home", "hf", "work", "repo", "tmp"))  # EROFS
        assert seen["vartmp"] == "written" and seen["shm"] == "written" and seen["scratch_env"] == [True, True]
        for d in (*targets.values(), tmp_path / "data"):
            assert not (d / token).exists(), d  # nothing reached the host


def test_protected_paths_cover_env_named_caches_and_drop_nested():
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:  # outside /tmp, which would cover everything
        cache = Path(tmp) / "c"
        (cache / "inner").mkdir(parents=True)
        paths = passrun.protected_paths(Path(tmp) / "r", Path(tmp) / "w",
                                        {"FOO_CACHE": str(cache), "BAR_HOME": str(cache / "inner")})
        assert cache.resolve() in paths and (cache / "inner").resolve() not in paths


def test_missing_setpriv_refuses_a_real_data_pass(repo, store, monkeypatch):
    monkeypatch.setattr(passrun, "_SETPRIV", ["no-such-setpriv"])
    with pytest.raises(PassError, match="cannot be hidden"):
        passrun.run_pass(Mode.ACCURACY, "validation", root=repo, data_dir=store, device="cpu")


def test_checker_pass_and_fail_output_with_a_frozen_gate(capsys, monkeypatch):
    gate = {**_gate(), "h_minutes": 5}
    monkeypatch.setattr(checker, "load_gate", lambda: gate)
    for score, code, text in ((_score(), 0, "PASS"), (_score(ic=0.0), 1, "FAIL: ic 0.0000 is below the floor")):
        result = passrun.PassResult(None, None, None, score)
        monkeypatch.setattr(checker, "run_pass", lambda *a, _r=result, **k: _r)
        assert checker.main([]) == code
        assert text in capsys.readouterr().out


def test_unshare_failure_refuses_a_real_data_pass_without_fallback(repo, store, monkeypatch):
    monkeypatch.setattr(passrun, "_UNSHARE", ["false"])
    with pytest.raises(PassError, match="cannot be hidden") as e:
        passrun.run_pass(Mode.ACCURACY, "validation", root=repo, data_dir=store, device="cpu")
    assert str(store) not in e.value.message


def test_failed_tmpfs_mount_is_reported_data_free(repo, store, monkeypatch):
    monkeypatch.setattr(passrun, "_UNSHARE", ["unshare", "-Ur"])  # no mount namespace: the mount is refused
    monkeypatch.setattr(passrun, "_check_can_hide", lambda *a: None)  # skip the early probe to hit the child path
    with pytest.raises(PassError, match="cannot be hidden") as e:
        passrun.run_pass(Mode.ACCURACY, "validation", root=repo, data_dir=store, device="cpu")
    assert str(store) not in e.value.message


def test_smoke_pass_does_not_need_the_namespace(repo, monkeypatch):
    monkeypatch.setattr(passrun, "_UNSHARE", ["false"])
    (repo / "serving" / "__main__.py").write_text("raise SystemExit(3)\n")
    with pytest.raises(PassError, match="exited with code 3"):
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")


def test_data_dir_must_not_contain_what_the_candidate_needs(tmp_path):
    data = tmp_path / "data"
    (data / "cache").mkdir(parents=True)
    with pytest.raises(PassError, match="lie inside"):
        passrun._check_can_hide(data, [data / "cache"])


# -- an invalid pass is rerun once ---------------------------------------------------------------------


LAG = "send lag p90 9.0 ms exceeds 5.0 ms"


def _attempts(*scores):
    it = iter(scores)
    calls = []

    def attempt():
        calls.append(1)
        return next(it)

    return attempt, calls


def test_invalid_pass_is_rerun_once_and_the_rerun_is_used(tmp_path):
    attempt, calls = _attempts("a", "b")
    scores = {"a": _score(valid=False, invalid_reason=LAG, send_lag_ms={"p50": 1.0, "p90": 9.0, "p99": 9.0, "max": 9.0}), "b": _score()}
    log, score, first = passrun.run_with_retry(attempt, scores.__getitem__)
    assert (log, len(calls), score.valid, first.invalid_reason) == ("b", 2, True, LAG)
    gate = _gate()
    rec = _check_stream(_write(tmp_path, bench.records_for(score, gate, DIRS, first)))
    assert rec["kind"] == "result" and rec["values"]["reruns"] == 1
    assert rec["values"]["first_pass_send_lag_p90_ms"] == 9.0
    assert checker.evaluate_gate(score, gate) is None


def test_valid_pass_is_not_rerun(tmp_path):
    attempt, calls = _attempts("a")
    _, score, first = passrun.run_with_retry(attempt, lambda _: _score())
    assert len(calls) == 1 and first is None
    rec = _check_stream(_write(tmp_path, bench.records_for(score, _gate(), DIRS, first)))
    assert rec["values"]["reruns"] == 0


def test_two_invalid_passes_fail_and_name_both_reasons(tmp_path):
    attempt, calls = _attempts("a", "b")
    second = "send lag p90 7.0 ms exceeds 5.0 ms"
    bad = {"a": _score(valid=False, invalid_reason=LAG), "b": _score(valid=False, invalid_reason=second)}
    _, score, first = passrun.run_with_retry(attempt, bad.__getitem__)
    assert len(calls) == 2 and not score.valid
    err = _check_stream(_write(tmp_path, bench.records_for(score, _gate(), DIRS, first)))
    assert err["kind"] == "error" and "after a rerun" in err["message"]
    assert "7.0" in err["message"] and "9.0" in err["message"]
    assert "invalid pass" in checker.evaluate_gate(score, _gate())


def test_bursts_mode_uses_the_named_gap_cap():
    assert passrun.SETTINGS[Mode.BURSTS].gap_cap_s == passrun.BURSTS_GAP_CAP_S == 0.25


def test_task_timeouts_cover_two_passes_and_prepare():
    import tomllib

    cfg = tomllib.loads((TASK / "vibesys.input.toml").read_text())
    acc, burst = passrun.SETTINGS[Mode.ACCURACY], passrun.SETTINGS[Mode.BURSTS]
    worst_acc = passrun.PREPARE_TIMEOUT_S + 2 * (
        acc.ready_timeout_s + passrun.MAX_STORIES * acc.gap_cap_s + acc.drain_s + 60
    )
    worst_bench = passrun.PREPARE_TIMEOUT_S + 2 * (
        burst.ready_timeout_s + passrun.BURSTS_BUDGET_S + 60 + burst.drain_s + 60
    )
    assert cfg["accuracy"]["timeout_seconds"] >= worst_acc
    assert cfg["benchmark"]["timeout_seconds"] >= worst_bench


def test_cpu_acc_artifact_survives_h_and_sizing_but_not_code_changes(freezer):
    module, fz, runs, repo = freezer
    art = runs / "cpu-acc.json"
    art.write_text("{}")
    fz.stamp(art, full=False)
    assert not fz.todo("cpu-acc", art)
    task = repo / ".vibesys" / "tasks" / "news-signal"
    module.set_toml(task / "gate.toml", "h_minutes", "30")  # what the h stage does
    (task / "passrun.py").write_text((task / "passrun.py").read_text() + "\n# sizing\n")
    assert not fz.todo("cpu-acc", art)
    module.set_toml(task / "gate.toml", "infosys_sha256", '"' + "1" * 64 + '"')
    assert fz.todo("cpu-acc", art)


def test_baseline_artifacts_are_stale_after_a_serving_or_task_file_edit(freezer):
    module, fz, runs, repo = freezer
    art = runs / "f1.json"
    art.write_text("{}")
    fz.stamp(art)
    assert not fz.todo("f1", art)
    (repo / "serving" / "model.py").write_text("# edited\n")
    assert fz.todo("f1", art)
    fz.stamp(art)
    (repo / ".vibesys" / "tasks" / "news-signal" / "benchmark" / "run.py").write_text("# edited\n")
    assert fz.todo("f1", art)


def test_send_lag_fields_are_in_the_benchmark_record(tmp_path):
    rec = _check_stream(_write(tmp_path, bench.records_for(_score(), _gate(), DIRS)))
    assert [rec["values"][f"newsd_send_lag_{k}_ms"] for k in ("p50", "p90", "p99", "max")] == [0.2, 0.3, 0.4, 0.9]


def test_send_lag_fields_are_in_the_checker_json(tmp_path):
    from dataclasses import asdict

    assert set(asdict(_score())["send_lag_ms"]) == {"p50", "p90", "p99", "max"}


def test_freeze_rule_matches_the_scorer():
    import importlib.util

    from infosys import score

    spec = importlib.util.spec_from_file_location("freeze_baseline", ROOT / "scripts" / "freeze_baseline.py")
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)
    assert fb.SEND_LAG_MAX_MS == {"p90": score.MAX_SEND_LAG_P90_MS, "p99": score.MAX_SEND_LAG_P99_MS}


def test_freeze_refuses_a_bench_result_that_breaks_the_send_lag_rule(freezer):
    _, fz, runs, _ = freezer
    _artifacts(fz, runs, lag_p90=6.0)
    with pytest.raises(SystemExit, match="send lag p90 6.0 ms"):
        fz.freeze()


def test_gate_commands_are_plain_uv_run_python():
    """VibeSys's Slurm policy rewrites only `uv run python <script>` to the operator interpreter; a flag between breaks it."""
    import tomllib

    spec = tomllib.loads((TASK / "vibesys.input.toml").read_text())
    for gate, script in (("accuracy", "accuracy_checker/checker.py"), ("benchmark", "benchmark/run.py")):
        cmd = spec[gate]["command"]
        assert cmd[:3] == ["uv", "run", "python"], cmd
        assert cmd[3] == f"${{PROJECT_ROOT}}/.vibesys/tasks/news-signal/{script}", cmd


# -- record-only stats: p90/max latency, throughput, cold start, GPU memory, per-feed latency ------------

FEEDS_STATS = {"benzinga": {"pairs": 900, "p50": 6.0, "p99": 25.0, "first": 60},
               "syn-1": {"pairs": 0, "p50": None, "p99": None, "first": 0}}  # fmt: skip


def test_benchmark_record_carries_the_record_only_scalars(tmp_path):
    serve = passrun.ServeProbe(cold_start_s=12.5, gpu_mem_idle_mib=300.0, gpu_mem_peak_mib=4096.0)
    rec = _check_stream(_write(tmp_path, bench.records_for(_score(), _gate(), DIRS, None, serve)))
    v = rec["values"]
    assert (v["p90_latency_ms"], v["max_latency_ms"], v["throughput_signals_per_s"]) == (9.0, 30.0, 50.0)
    assert (v["cold_start_s"], v["gpu_mem_peak_mib"]) == (12.5, 4096.0)


def test_unreadable_gpu_memory_and_unknown_cold_start_are_absent_not_failures(tmp_path):
    for serve in (None, passrun.ServeProbe()):
        rec = _check_stream(_write(tmp_path, bench.records_for(_score(), _gate(), DIRS, None, serve)))
        assert rec["kind"] == "result" and "gpu_mem_peak_mib" not in rec["values"] and "cold_start_s" not in rec["values"]


def test_record_only_numbers_change_no_decision():
    base, odd = _score(), _score(feed_latency_ms=FEEDS_STATS)
    gate = _gate()
    assert checker.evaluate_gate(base, gate) == checker.evaluate_gate(odd, gate) is None
    serve = passrun.ServeProbe(cold_start_s=1e6, gpu_mem_peak_mib=1e9)
    kinds = {bench.records_for(base, gate, DIRS, None, s)[-1]["kind"] for s in (None, serve)}
    assert kinds == {"result"}


def test_benchmark_main_writes_a_sidecar_with_the_per_feed_stats(tmp_path, monkeypatch):
    from types import SimpleNamespace

    serve = passrun.ServeProbe(cold_start_s=2.0, gpu_mem_peak_mib=None)
    result = SimpleNamespace(score=_score(feed_latency_ms=FEEDS_STATS), log=SimpleNamespace(signals=[]), first_invalid=None, serve=serve)
    result.log.signals = [SimpleNamespace(direction=d) for d in (-1, 1)]
    monkeypatch.setattr(bench, "run_pass", lambda *a, **k: result)
    monkeypatch.setattr(bench, "load_gate", _gate)
    out = tmp_path / "o.jsonl"
    assert bench.main(["--mode", "bursts", "--vs-output", str(out)]) == 0
    _check_stream(out.read_text().splitlines())
    stats = json.loads((tmp_path / "o.jsonl.stats.json").read_text())
    assert stats["feeds"] == FEEDS_STATS and stats["serve"]["gpu_mem_peak_mib"] is None
    assert stats["latency_ms"]["max"] == 30.0 and stats["throughput_signals_per_s"] == 50.0


def test_unwritable_sidecar_does_not_fail_a_written_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    result = SimpleNamespace(score=_score(), log=SimpleNamespace(signals=[SimpleNamespace(direction=d) for d in (-1, 1)]),
                             first_invalid=None, serve=passrun.ServeProbe())
    monkeypatch.setattr(bench, "run_pass", lambda *a, **k: result)
    monkeypatch.setattr(bench, "load_gate", _gate)
    out = tmp_path / "o.jsonl"
    (tmp_path / "o.jsonl.stats.json").mkdir()  # a directory where the sidecar goes
    assert bench.main(["--mode", "bursts", "--vs-output", str(out)]) == 0
    assert _check_stream(out.read_text().splitlines())["kind"] == "result"


def test_failed_benchmark_writes_no_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "run_pass", lambda *a, **k: (_ for _ in ()).throw(PassError("x")))
    monkeypatch.setattr(bench, "load_gate", _gate)
    assert bench.main(["--mode", "bursts", "--vs-output", str(tmp_path / "o.jsonl")]) == 1
    assert not (tmp_path / "o.jsonl.stats.json").exists()


def test_checker_json_has_the_new_fields(tmp_path):
    from dataclasses import asdict

    assert asdict(_score(feed_latency_ms=FEEDS_STATS))["feed_latency_ms"] == FEEDS_STATS
    assert set(asdict(passrun.ServeProbe())) == {"cold_start_s", "gpu_mem_idle_mib", "gpu_mem_peak_mib"}


def test_gpu_sampler_keeps_the_peak_and_the_first_reading_as_idle():
    readings = iter([1 << 20, 9 << 20, 4 << 20])
    sampler = passrun.GpuMemSampler(lambda: next(readings))
    assert [sampler.sample() for _ in range(3)] == [True] * 3
    assert (sampler.idle_mib, sampler.peak_mib) == (1.0, 9.0)


def test_gpu_sampler_is_null_when_nothing_can_be_read_and_never_raises():
    def broken():
        raise RuntimeError("driver")

    for read in (lambda: None, broken):
        sampler = passrun.GpuMemSampler(read)
        sampler.start()
        sampler.stop()
        assert (sampler.idle_mib, sampler.peak_mib) == (None, None)


def test_gpu_sampler_skips_a_failed_reading_after_a_good_one():
    readings = iter([2 << 20, None, 5 << 20])
    sampler = passrun.GpuMemSampler(lambda: next(readings))
    assert [sampler.sample() for _ in range(3)] == [True, False, True]
    assert sampler.peak_mib == 5.0


def test_gpu_sampler_thread_samples_at_start_and_at_stop():
    readings = iter([1 << 20])
    sampler = passrun.GpuMemSampler(lambda: next(readings, 6 << 20), period_s=3600)  # the period never elapses
    sampler.sample()
    sampler.start()
    sampler.stop()
    assert (sampler.idle_mib, sampler.peak_mib) == (1.0, 6.0)


def test_gpu_sampling_period_does_not_disturb_latency():
    assert passrun.GPU_SAMPLE_PERIOD_S >= 0.1


def test_vram_is_summed_over_sysfs_cards_and_null_without_any_source(tmp_path, monkeypatch):
    for i, used in enumerate((100, 250)):
        card = tmp_path / f"card{i}" / "device"
        card.mkdir(parents=True)
        (card / "mem_info_vram_used").write_text(f"{used}\n")
    (tmp_path / "card1-DP-1").mkdir()
    monkeypatch.setattr(passrun, "_VRAM_USED_GLOB", str(tmp_path / "card[0-9]*/device/mem_info_vram_used"))
    assert passrun.read_vram_used_bytes() == 350
    monkeypatch.setattr(passrun, "_VRAM_USED_GLOB", str(tmp_path / "none*"))
    assert passrun.read_vram_used_bytes() is None


def test_freeze_surfaces_record_only_stats_without_touching_decisions(freezer):
    module, fz, runs, repo = freezer
    _artifacts(fz, runs)
    task = repo / ".vibesys" / "tasks" / "news-signal"
    gate_before = None
    fz.freeze()
    gate_before = (task / "gate.toml").read_bytes()
    md = (task / "OBJECTIVE.md").read_text()
    assert "### Baseline (record only)" in md and "peak GPU memory 4096 MiB" in md and "model load (spawn to ready) 12.5 s" in md
    assert "feed `benzinga`: 900 pairs" in md and "first on 60 stories" in md and "max 30.0" in md
    assert "throughput 50.0 signals/s" in md
    full = json.loads((runs / "baseline-stats.json").read_text())
    assert set(full) == {"commit", "f1", "f2", "cpu_acc"} and full["f2"][1]["stats"]["serve"]["cold_start_s"] == 12.5
    assert full["cpu_acc"] == {"horizons": {}}
    for key in ("cold_start", "gpu_mem", "feed", "throughput", "max_latency"):
        assert key not in gate_before.decode() and key not in (task / "objectives.toml").read_text()


def test_baseline_stats_file_is_never_overwritten(freezer):
    _, fz, runs, _ = freezer
    _artifacts(fz, runs)
    fz.freeze()
    first = (runs / "baseline-stats.json").read_text()
    fz.freeze()
    fz.freeze()
    assert (runs / "baseline-stats.json").read_text() == first
    siblings = sorted(p.name for p in runs.glob("baseline-stats-*.json"))
    assert len(siblings) == 2 and len(set(siblings)) == 2  # same second or not, each write got its own file


def test_freeze_prints_the_record_only_block(freezer, capsys):
    _, fz, runs, _ = freezer
    _artifacts(fz, runs)
    fz.freeze()
    out = capsys.readouterr().out
    printed = json.loads(out[out.index("{"):])
    assert printed["record_only"]["f1"]["gpu_mem_peak_mib"] == 4096.0
    assert [r["latency_ms"]["max"] for r in printed["record_only"]["f2"]] == [30.0, 30.0]
    assert printed["record_only"]["f2"][0]["feeds"]["benzinga"]["first"] == 60


def test_freeze_without_record_only_sidecars_writes_nulls(freezer):
    _, fz, runs, _ = freezer
    _artifacts(fz, runs, sidecars=False)
    fz.freeze()
    md = (freezer[3] / ".vibesys" / "tasks" / "news-signal" / "OBJECTIVE.md").read_text()
    assert "Bursts run 1: latency p50 10.0" in md and "model load (spawn to ready) null" in md


def test_write_new_never_overwrites(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("fz_w", ROOT / "scripts" / "freeze_baseline.py")
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)
    target = tmp_path / "x.json"
    paths = [fb.write_new(target, str(i)) for i in range(3)]
    assert paths[0] == target and len(set(paths)) == 3 and [p.read_text() for p in paths] == ["0", "1", "2"]
