#!/usr/bin/env python3
"""Replace solvent 26 with the real coarse-grained water from Water_ions.

The benchmark CSV labels solvent 26 as Water.  If output/26_* was generated
from a non-water molecule, all pairs involving 26 need to be rebuilt.  This
script creates numbered water files in output/ and optionally cleans old
pair_*_26 simulation artifacts so phase_sim_run.sh will rebuild them.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path.cwd()
OUTPUT = ROOT / "output"
WATER = ROOT / "Water_ions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-pairs", action="store_true", help="remove old stage outputs for pair_*_26 directories")
    parser.add_argument("--pairs-dir", default="phase_pairs")
    return parser.parse_args()


def backup_once(path: Path) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".before_water26")
        if not backup.exists():
            shutil.copy2(path, backup)


def write_numbered_water_files() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name in ["26_ML.itp", "26.pdb", "26.gro", "26.top"]:
        backup_once(OUTPUT / name)

    (OUTPUT / "26_ML.itp").write_text(
        """[moleculetype]
26    1

[atoms]
1    W    1    26    W    1    0.000    72.000
""",
        encoding="utf-8",
    )

    w_pdb = WATER / "W.pdb"
    if not w_pdb.exists():
        raise SystemExit(f"Missing {w_pdb}")
    lines = []
    for raw in w_pdb.read_text(errors="replace").splitlines():
        if raw.startswith("TITLE"):
            lines.append("TITLE     Coarse-grained structure of 26 Water")
        elif raw.startswith("ATOM"):
            lines.append("ATOM      1 W    26  A   1       0.540  -0.157   0.003  1.00  0.00          CG")
        else:
            lines.append(raw)
    (OUTPUT / "26.pdb").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUTPUT / "26.gro").write_text(
        """Coarse-grained water 26
    1
    126      W    1   0.540  -0.157   0.003
   1.00000   1.00000   1.00000
""",
        encoding="utf-8",
    )

    (OUTPUT / "26.top").write_text(
        """; Topology file for solvent 26 Water

#include "forcefield_ML.itp"
#include "26_ML.itp"

[ system ]
26_Water

[ molecules ]
26 100
""",
        encoding="utf-8",
    )


def clean_pair_dirs(pairs_dir: Path) -> int:
    if not pairs_dir.exists():
        return 0
    patterns = [
        "packed*.pdb",
        "packed.gro",
        "editconf.log",
        "min.*",
        "nvt.*",
        "eq.*",
        "eq2.*",
        "*_grompp.log",
        "*.done",
        "*.failed",
        "*.mdp.sha256",
        "*.run.sha256",
        "*_prev.cpt",
        "step*.pdb",
        "#*#",
    ]
    count = 0
    for pair_dir in sorted(pairs_dir.glob("pair_*_26")) + sorted(pairs_dir.glob("pair_26_*")):
        local_pdb = pair_dir / "26.pdb"
        if local_pdb.exists():
            backup_once(local_pdb)
        shutil.copy2(OUTPUT / "26.pdb", local_pdb)
        for pattern in patterns:
            for path in pair_dir.glob(pattern):
                if path.is_file():
                    path.unlink()
        count += 1
    return count


def main() -> None:
    args = parse_args()
    write_numbered_water_files()
    cleaned = clean_pair_dirs(ROOT / args.pairs_dir) if args.clean_pairs else 0
    print("Replaced output/26_ML.itp and output/26.pdb with Water_ions water.")
    if args.clean_pairs:
        print(f"Cleaned {cleaned} pair directories involving solvent 26.")


if __name__ == "__main__":
    main()
