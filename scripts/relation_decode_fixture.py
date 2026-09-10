#!/usr/bin/env python3
"""Run fixed CPU Point 4 engineering fixtures; no checkpoints or scientific scores."""
import argparse
from pathlib import Path

from dm.eval.relation_decode_evidence import publish_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new JSON report path; existing paths are refused")
    args = parser.parse_args()
    publish_report(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
