import argparse
import asyncio
import os
import sys

from serving.model import Classifier
from serving.service import run


def main() -> None:
    p = argparse.ArgumentParser(prog="serving")
    p.add_argument("--newsd", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True, choices=["cuda", "cpu"])
    p.add_argument("--prepared", required=True)  # native runtime artifacts live in <prepared>/native
    a = p.parse_args()
    from serving.native import select_binary

    exe = select_binary(a.prepared, a.model, a.device)
    if exe is not None:
        print("serving: runtime=native", file=sys.stderr, flush=True)
        os.execv(str(exe), [str(exe), "--newsd", a.newsd, "--model", a.model, "--device", a.device,
                            "--prepared", a.prepared])
    print("serving: runtime=python", file=sys.stderr, flush=True)
    clf = Classifier(a.model, a.device)
    clf.warmup()
    asyncio.run(run(a.newsd, clf.predict_ids, clf.tokenize))


if __name__ == "__main__":
    main()
