"""Harness tests on the synthetic fixture (CPU). Everything real runs in subprocesses of the pass runner."""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hypothesis import given
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
        scored_pairs=800, ic=0.05, ic_se=0.01, hit_rate=0.52, send_lag_p99_ms=0.4,
    )  # fmt: skip
    return PassScore(**{**base, **kw})


def _gate(frozen=True, floor=0.03) -> dict:
    return {
        "max_missed_frac": 0.001, "max_duplicate_frac": 0.001, "max_invalid": 0,
        "baseline": {"frozen": frozen, "ic_floor": floor},
    }  # fmt: skip


# -- pins and gate file ---------------------------------------------------------------------------------


def test_infosys_pin_matches_tree():
    assert passrun.infosys_digest(ROOT) == passrun.load_gate()["infosys_sha256"]


def test_gate_toml_complete():
    gate = passrun.load_gate()
    assert isinstance(gate["h_minutes"], int) and gate["h_minutes"] > 0
    assert len(gate["infosys_sha256"]) == 64
    base = gate["baseline"]
    assert isinstance(base["frozen"], bool)
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
    dump = tmp_path / "env.json"
    (repo / "serving" / "__main__.py").write_text(
        f"import json, os\nopen({str(dump)!r}, 'w').write(json.dumps(sorted(os.environ)))\n"
    )
    monkeypatch.setenv("NEWS_SIGNAL_DATA", "/nowhere")
    monkeypatch.setenv("NEWS_SIGNAL_OTHER", "x")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(PassError):  # exits 0 without ever calling /ready
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")
    names = json.loads(dump.read_text())
    assert not [n for n in names if n.startswith("NEWS_SIGNAL_")]
    assert "HF_HUB_OFFLINE" in names


def test_prepare_failure_is_reported(repo):
    (repo / "serving" / "prepare.py").write_text("import sys\nprint('boom')\nsys.exit(4)\n")
    with pytest.raises(PassError, match="prepare exited with code 4: boom"):
        passrun.run_pass(Mode.SMOKE, "validation", root=repo, device="cpu")


@pytest.fixture(scope="module")
def smoke(tiny_model):
    return passrun.run_pass(Mode.SMOKE, "validation", device="cpu", model=str(tiny_model))


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


@given(
    ic=st.one_of(st.none(), st.floats(-0.2, 0.2)),
    missed=st.integers(0, 5), dup=st.integers(0, 5), invalid=st.integers(0, 2), valid=st.booleans(),
    frozen=st.booleans(),
)  # fmt: skip
def test_gate_passes_exactly_when_all_checks_hold(ic, missed, dup, invalid, valid, frozen):
    score = _score(
        ic=ic, missed=missed, duplicates=dup, invalid=invalid, valid=valid,
        invalid_reason=None if valid else "send lag p99 9.0 ms exceeds 5.0 ms",
    )  # fmt: skip
    expected = frozen and valid and ic is not None and ic >= 0.03 and missed <= 1 and dup <= 1 and invalid == 0
    assert (checker.evaluate_gate(score, _gate(frozen=frozen)) is None) == expected


def test_gate_names_the_invalid_pass():
    msg = checker.evaluate_gate(_score(valid=False, invalid_reason="send lag p99 9.0 ms exceeds 5.0 ms"), _gate())
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
    ok = _check_stream(_write(tmp_path, bench.records_for(_score(), gate)))
    assert ok["kind"] == "result" and ok["values"]["p99_latency_ms"] == 20.0
    for bad in (
        _score(missed=5), _score(duplicates=5), _score(invalid=1),
        _score(valid=False, invalid_reason="send lag p99 9.0 ms exceeds 5.0 ms"),
        _score(latency_ms={"p50": None, "p90": None, "p99": None, "max": None}, missed=1000, signaled=0),
    ):  # fmt: skip
        err = _check_stream(_write(tmp_path, bench.records_for(bad, gate)))
        assert err["kind"] == "error" and "error_kind" not in err
    named = bench.records_for(_score(missed=5), gate)[-1]
    assert named["message"].startswith("missed 5") and named["partial"]["value"] == 20.0
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


# -- baseline freeze arithmetic -------------------------------------------------------------------------


def _load_freeze():
    import importlib.util

    spec = importlib.util.spec_from_file_location("freeze_baseline", ROOT / "scripts" / "freeze_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@given(ic=st.floats(-0.5, 0.5), se=st.floats(0, 0.1), a=st.floats(0.1, 1e4), b=st.floats(0.1, 1e4))
def test_freeze_derivation(ic, se, a, b):
    floor, noise = _load_freeze().derive(ic, se, a, b)
    assert floor <= ic - 2 * se + 1e-12 and ic - 2 * se - floor < 1.1e-4
    assert noise >= 0.03 and noise + 1e-9 >= abs(a - b) / ((a + b) / 2)


def test_freeze_set_toml_edits_only_the_named_section(tmp_path):
    path = tmp_path / "g.toml"
    path.write_text("ic = 1\n\n[baseline]\nic = 0.0\nfrozen = false\n")
    freeze = _load_freeze()
    freeze.set_toml(path, "ic", "0.25", section="baseline")
    freeze.set_toml(path, "frozen", "true", section="baseline")
    assert path.read_text() == "ic = 1\n\n[baseline]\nic = 0.25\nfrozen = true\n"


def test_freeze_rewrites_task_files_idempotently(repo, tmp_path):
    import tomllib

    runs = tmp_path / "runs"
    runs.mkdir()
    score = {"stories": 1000, "scored_pairs": 700, "ic": 0.04125, "ic_se": 0.01, "missed": 0, "duplicates": 0,
             "invalid": 0, "valid": True}  # fmt: skip
    (runs / "f1.json").write_text(json.dumps({"score": score}))
    for i, p99 in enumerate((100.0, 104.0), start=1):
        values = {"p99_latency_ms": p99, "p50_latency_ms": 10.0, "stories": 300}
        rec = {"kind": "result", "label": "", "values": values}
        (runs / f"bench-{i}.jsonl").write_text(json.dumps(bench.HELLO) + "\n" + json.dumps(rec) + "\n")

    def freeze() -> dict:
        subprocess.run(
            [sys.executable, str(repo / "scripts" / "freeze_baseline.py"), "--stages", "freeze"],
            env={**os.environ, "NS_RUNS": str(runs)}, check=True, capture_output=True, timeout=BOUND_S,
        )  # fmt: skip
        task = repo / ".vibesys" / "tasks" / "news-signal"
        return {n: (task / n).read_bytes() for n in ("gate.toml", "objectives.toml", "OBJECTIVE.md")}

    first = freeze()
    gate = tomllib.loads(first["gate.toml"].decode())
    base = gate["baseline"]
    assert base["frozen"] is True and base["ic_floor"] == 0.0212 and base["p99_ms"] == [100.0, 104.0]
    assert base["validation_stories"] == 1000 and base["scored_pairs"] == 700
    assert tomllib.loads(first["objectives.toml"].decode())["pareto"]["relative_noise"] == 0.04
    assert b"IC 0.04" in first["OBJECTIVE.md"] and b"Pending" not in first["OBJECTIVE.md"]
    assert freeze() == first
