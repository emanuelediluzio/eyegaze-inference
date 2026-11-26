"""The R1 split of RT-GENE by person (RULE-R1.md): test or distillation, fixed before any image is looked at.

    python split_rtgene.py RTGENE_DIR

Only directory names are read. A person is the subject ID sNNN at the start of a folder name, so every folder of
the same subject (e.g. with and without the eye-tracking glasses) goes to the same group. Subjects are ordered by
sha256("eyegaze-r1:" + ID); the first half, rounded up, is the test group.
"""
import hashlib
import re
import sys
from pathlib import Path

SALT = "eyegaze-r1:"


def split(ids):
    ordered = sorted(set(ids), key=lambda s: hashlib.sha256((SALT + s).encode()).hexdigest())
    half = (len(ordered) + 1) // 2
    return sorted(ordered[:half]), sorted(ordered[half:])


def main():
    names = [p.name for p in Path(sys.argv[1]).iterdir() if p.is_dir()]
    ids = [m.group(1) for m in (re.match(r"(s\d{3})", n) for n in names) if m]
    if not ids:
        sys.exit("no sNNN folders")
    test, distill = split(ids)
    print("test:", " ".join(test))
    print("distillation:", " ".join(distill))


if __name__ == "__main__":
    main()
