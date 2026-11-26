"""The R1 split of RT-GENE by person (RULE-R1.md): test or distillation, fixed before any image is looked at.

    python split_rtgene.py RTGENE_DIR

Only names are read: the entries of RTGENE_DIR and the member names of any .zip in it (its central directory, no
member is extracted). A session is the sNNN at the start of a name (sNNN_glasses, sNNN_noglasses). A person is a
participant: the dataset's README_Dataset.md says session s014 is participant s002 recorded again and s015 is s005,
so those sessions go with their participant. People are ordered by sha256("eyegaze-r1:" + person); the first half,
rounded up, is the test group, with all their sessions.
"""
import hashlib
import re
import sys
import zipfile
from pathlib import Path

SALT = "eyegaze-r1:"
SAME_PERSON = {"s014": "s002", "s015": "s005"}     # README_Dataset.md, "Subjects / 3-Fold evaluation"


def split(ids):
    ordered = sorted(set(ids), key=lambda s: hashlib.sha256((SALT + s).encode()).hexdigest())
    half = (len(ordered) + 1) // 2
    return sorted(ordered[:half]), sorted(ordered[half:])


def names(folder):
    out = []
    for p in Path(folder).iterdir():
        out.append(p.name)
        if p.suffix.lower() == ".zip":
            out += [part for n in zipfile.ZipFile(p).namelist() for part in n.split("/")]
    return out


def main():
    sessions = sorted({m.group(1) for m in (re.match(r"(s\d{3})_", n) for n in names(sys.argv[1])) if m})
    if not sessions:
        sys.exit("no sNNN_ names")
    person = {s: SAME_PERSON.get(s, s) for s in sessions}
    test, distill = split(person.values())
    for title, group in [("test", test), ("distillation", distill)]:
        print(f"{title} ({len(group)} people):", " ".join(
            p + ("" if [s for s in sessions if person[s] == p] == [p] else
                 " (" + "+".join(s for s in sessions if person[s] == p) + ")") for p in group))
    print(f"{len(sessions)} sessions, {len(set(person.values()))} people")


if __name__ == "__main__":
    main()
