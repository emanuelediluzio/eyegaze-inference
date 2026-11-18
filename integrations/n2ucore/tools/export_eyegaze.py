"""Export a trusted training checkpoint to the minimal node inference package."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from N2uCore.inference.gazeartifact import exportInference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--int8', action='store_true',
                        help='weight matrices as int8 with a scale per row: 87 MB instead of 324, same accuracy')
    args = parser.parse_args()
    report = exportInference(args.source, args.output, int8=args.int8)
    if report is None:
        parser.exit(1, "GazeDINO export failed; source or destination is invalid.\n")
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
