import argparse
import asyncio

from serving.model import Classifier
from serving.service import run


def main() -> None:
    p = argparse.ArgumentParser(prog="serving")
    p.add_argument("--newsd", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True, choices=["cuda", "cpu"])
    p.add_argument("--prepared", required=True)  # unused: the baseline prepares nothing
    a = p.parse_args()
    asyncio.run(run(a.newsd, Classifier(a.model, a.device).predict))


main()
