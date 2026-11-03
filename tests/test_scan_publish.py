import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "scan_publish", Path(__file__).resolve().parents[1] / "scripts" / "scan_publish.py"
)
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

# Fake credentials and denied names are assembled at runtime so this file itself passes the scan.
DENY = ["sh" + "li", "chel" + "an", "@uw" + ".edu"]
FAKE_KEY_ID = "P" + "K" + "ABCDEFGHJKLMNPQRST12"
FAKE_BASE62_40 = "aB3dE5gH7jK9mN1pQ3sT" + "5vW7yZ9bC2eF4hJ6kL8n"
HIGH_ENTROPY = "CwY1tY7nR4vB8Zm2KqL9x"[::-1]


def rules(text: str) -> list[str]:
    return [r for _, r in scan.scan_text(text, DENY)]


def test_stdlib_module_name_is_not_a_deny_hit():
    assert rules("import hash" + "lib") == []


def test_bare_token_and_path_segments_are_deny_hits():
    bare = "sh" + "li"
    assert rules(bare) == ["deny-pattern"]
    assert rules(f"run in rel/x/{bare}/y") == ["deny-pattern"]
    assert rules("host chel" + "an3 is up") == ["deny-pattern"]
    assert rules("mail bob@uw" + ".edu") == ["deny-pattern"]
    assert rules("Sh" + "Li") == ["deny-pattern"]  # case-insensitive


def test_absolute_paths():
    assert rules("see /Us" + "ers/someone/file") == ["absolute-path"]
    assert rules("see /feeds/x/ws and ~/Users/x and a/home/b") == []


def test_alpaca_key_id_shape():
    assert rules(f"id {FAKE_KEY_ID}") == ["secret-shaped"]
    assert rules("A" + "K" + "ABCDEFGHJKLMNPQRST12") == ["secret-shaped"]
    assert rules("PK" + "SHORT") == []


def test_forty_char_base62_secret_shape():
    assert len(FAKE_BASE62_40) == 40
    assert rules(f"value {FAKE_BASE62_40}") == ["secret-shaped"]
    assert rules("a" * 40) == []  # single case, no digit: not secret-like
    assert rules("0123456789abcdef0123456789abcdef01234567") == []  # a git sha


def test_high_entropy_assignment():
    assert rules(f'api_{"ke" + "y"} = "{HIGH_ENTROPY}"') == ["secret-shaped"]
    assert rules(f'{"SE" + "CRET"}: {HIGH_ENTROPY}') == ["secret-shaped"]
    assert rules(f'{"to" + "ken"} = "{"a" * 24}"') == []  # low entropy
    assert rules(f"name = {HIGH_ENTROPY}") == []  # not a secret-like name
