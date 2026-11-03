import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_cross_imports():
    offenders = []
    for package, other in (("serving", "infosys"), ("infosys", "serving")):
        for path in sorted((ROOT / package).rglob("*.py")):
            if other in _imported_roots(path):
                offenders.append(f"{path.relative_to(ROOT)} imports {other}")
    assert not offenders, offenders
