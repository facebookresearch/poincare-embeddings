#!/usr/bin/env python3

import argparse
import json

from hype.hypernymy_eval import main as hype_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    opt = parser.parse_args()

    results, summary = hype_eval(opt.file, cpu=False)

    print(json.dumps(results))


if __name__ == "__main__":
    main()  # pragma: no cover
