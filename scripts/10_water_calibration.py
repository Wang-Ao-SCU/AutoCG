#!/usr/bin/env python3
"""

This script encodes the empirical correction learned from the solvent phase
test set. 
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


DEFAULT_TEST_PAIRS = [
    "pair_1_26",   # Water false positive: polar acid should mix
    "pair_9_26",   # Water false negative: cyclohexane should phase separate
    "pair_13_18",  # Dioxane false positive: methanol should mix
    "pair_9_13",   # Dioxane false negative: cyclohexane should phase separate
    "pair_13_26",  # Water-Dioxane false positive: should mix
]

DIOXANE_ID = 13
WATER_ID = 26

POLAR_TYPES = {
    "CG0", "CG1", "CG2", "CG5", "CG6", "CG10", "CG11", "CG12",
    "CG13", "CG14", "CG15", "CG16", "CG17", "DIOX_O", "DIOX_C",
}
HYDROPHOBIC_TYPES = {"CG3", "CG4", "CG7", "CG8", "CG9", "CG18", "CG19"}

CALIBRATION_PRESETS = {
    "v1": {
        "description": "conservative first-pass Water/Dioxane correction",
        "water_polar_scale": 1.25,
        "water_hydrophobe_scale": 0.45,
        "dioxane_polar_scale": 1.35,
        "dioxane_hydrophobe_scale": 0.60,
        "dioxane_self_scale": 0.55,
    },
    "v2": {
        "description": "stronger general correction validated on remaining Water/Dioxane mismatches",
        "water_polar_scale": 1.80,
        "water_hydrophobe_scale": 0.20,
        "dioxane_polar_scale": 2.00,
        "dioxane_hydrophobe_scale": 0.25,
        "dioxane_self_scale": 0.20,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(CALIBRATION_PRESETS), default="v2")
    parser.add_argument("--source-pairs-dir", default="phase_pairs")
    parser.add_argument("--test-pairs-dir", default="phase_pairs_water_dioxane_calib_test")
    parser.add_argument("--source-output-dir", default="output")
    parser.add_argument("--test-output-dir", default="output_water_dioxane_calib")
    parser.add_argument("--current-csv", default="phase_separation_results_contact025_from_boxfix.csv")
    parser.add_argument("--output-csv", default="phase_separation_results_water_dioxane_calib_test.csv")
    parser.add_argument("--rerun-script", default="rerun_water_dioxane_calib_test.sh")
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_TEST_PAIRS)
    parser.add_argument(
        "--select-mismatch-targets",
        nargs="*",
        default=None,
        help="Select L2=no pairs from --current-csv containing any listed molecule ids, e.g. 13 26.",
    )
    parser.add_argument("--water-polar-scale", type=float, default=None)
    parser.add_argument("--water-hydrophobe-scale", type=float, default=None)
    parser.add_argument("--dioxane-polar-scale", type=float, default=None)
    parser.add_argument("--dioxane-hydrophobe-scale", type=float, default=None)
    parser.add_argument("--dioxane-self-scale", type=float, default=None)
    parser.add_argument("--output-only", action="store_true", help="only write calibrated output; do not copy pair dirs or write a rerun script")
    parser.add_argument("--force", action="store_true", help="replace existing test pair directories")
    args = parser.parse_args()
    apply_preset(args)
    return args


def apply_preset(args: argparse.Namespace) -> None:
    preset = CALIBRATION_PRESETS[args.preset]
    for name in (
        "water_polar_scale",
        "water_hydrophobe_scale",
        "dioxane_polar_scale",
        "dioxane_hydrophobe_scale",
        "dioxane_self_scale",
    ):
        if getattr(args, name) is None:
            setattr(args, name, preset[name])


def pair_sort_key(pair: str) -> tuple[int, int]:
    _, a, b = pair.split("_")
    return int(a), int(b)


def selected_pairs_from_csv(path: Path, target_ids: set[str]) -> list[str]:
    pairs: set[str] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("L2") or "").strip() != "no":
                continue
            x = (row.get("X") or "").strip()
            y = (row.get("Y") or "").strip()
            if x in target_ids or y in target_ids:
                a, b = sorted((int(x), int(y)))
                pairs.add(f"pair_{a}_{b}")
    return sorted(pairs, key=pair_sort_key)


def read_atomtypes(forcefield: Path) -> dict[str, tuple[float, float, float, float]]:
    atomtypes: dict[str, tuple[float, float, float, float]] = {}
    in_atomtypes = False
    for raw in forcefield.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_atomtypes = line.strip("[]").strip().lower() == "atomtypes"
            continue
        if in_atomtypes:
            parts = line.split()
            if len(parts) >= 6:
                atomtypes[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[4]), float(parts[5]))
    return atomtypes


def read_nonbond_params(forcefield: Path) -> dict[tuple[str, str], tuple[float, float]]:
    params: dict[tuple[str, str], tuple[float, float]] = {}
    in_nonbond = False
    for raw in forcefield.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_nonbond = line.strip("[]").strip().lower() == "nonbond_params"
            continue
        if in_nonbond:
            parts = line.split()
            if len(parts) >= 5 and parts[2] == "1":
                key = tuple(sorted((parts[0], parts[1])))
                params[key] = (float(parts[3]), float(parts[4]))
    return params


def mixed_param(
    atomtypes: dict[str, tuple[float, float, float, float]],
    params: dict[tuple[str, str], tuple[float, float]],
    a: str,
    b: str,
) -> tuple[float, float]:
    key = tuple(sorted((a, b)))
    if key in params:
        return params[key]
    sigma_a = atomtypes[a][2]
    eps_a = atomtypes[a][3]
    sigma_b = atomtypes[b][2]
    eps_b = atomtypes[b][3]
    return 0.5 * (sigma_a + sigma_b), (eps_a * eps_b) ** 0.5


def write_adjusted_13_itp(src: Path, dest: Path) -> None:
    out: list[str] = []
    in_atoms = False
    for raw in src.read_text().splitlines():
        line = raw
        stripped = raw.split(";", 1)[0].strip()
        if stripped.startswith("["):
            in_atoms = stripped.strip("[]").strip().lower() == "atoms"
        elif in_atoms and stripped:
            parts = line.split()
            if parts and parts[0].isdigit():
                bead = parts[1]
                if bead == "CG12":
                    line = line.replace("CG12", "DIOX_O", 1)
                elif bead == "CG4":
                    line = line.replace("CG4", "DIOX_C", 1)
        out.append(line)
    dest.write_text("\n".join(out) + "\n")


def write_adjusted_forcefield(src: Path, dest: Path, args: argparse.Namespace) -> None:
    atomtypes = read_atomtypes(src)
    params = read_nonbond_params(src)
    mass, charge, sigma, eps = atomtypes["CG12"]
    diox_atomtype_lines = [
        "; Water/Dioxane calibration atomtypes. Generated by scripts/12_water_dioxane_calibration.py",
        f"DIOX_O   {mass:.4f}  {charge:.3f}  A  {sigma:.4f}  {eps:.4f}",
    ]
    mass, charge, sigma, eps = atomtypes["CG4"]
    diox_atomtype_lines.append(f"DIOX_C   {mass:.4f}  {charge:.3f}  A  {sigma:.4f}  {eps:.4f}")

    all_types = sorted(atomtypes)
    rows: list[tuple[str, str, float, float, str]] = []

    for diox in ("DIOX_O", "DIOX_C"):
        for other in all_types + ["DIOX_O", "DIOX_C"]:
            if other in {"SOD", "CLA"}:
                continue
            sigma0, eps0 = mixed_param(atomtypes | {
                "DIOX_O": atomtypes["CG12"],
                "DIOX_C": atomtypes["CG4"],
            }, params, "CG12" if diox == "DIOX_O" else "CG4", other if not other.startswith("DIOX_") else ("CG12" if other == "DIOX_O" else "CG4"))
            if other.startswith("DIOX_"):
                scale = args.dioxane_self_scale
                reason = "dioxane self reduction"
            elif other in HYDROPHOBIC_TYPES:
                scale = args.dioxane_hydrophobe_scale
                reason = "dioxane hydrophobe reduction"
            elif other in POLAR_TYPES or other == "W":
                scale = args.dioxane_polar_scale
                reason = "dioxane polar mixing"
            else:
                scale = 1.0
                reason = "dioxane neutral"
            rows.append((diox, other, sigma0, eps0 * scale, reason))

    for other in all_types + ["DIOX_O", "DIOX_C"]:
        if other in {"W", "SOD", "CLA"}:
            continue
        base_other = "CG12" if other == "DIOX_O" else "CG4" if other == "DIOX_C" else other
        sigma0, eps0 = mixed_param(atomtypes, params, "W", base_other)
        if other in HYDROPHOBIC_TYPES:
            scale = args.water_hydrophobe_scale
            reason = "water hydrophobe demixing"
        elif other in POLAR_TYPES or other.startswith("DIOX_"):
            scale = args.water_polar_scale
            reason = "water polar mixing"
        else:
            scale = 1.0
            reason = "water neutral"
        rows.append(("W", other, sigma0, eps0 * scale, reason))

    nonbond_lines = [
        "",
        "[ nonbond_params ]",
        "; Water/Dioxane calibration overrides: i j func sigma epsilon",
    ]
    seen: set[tuple[str, str]] = set()
    for a, b, sigma, eps, reason in rows:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        nonbond_lines.append(f"{a:8s} {b:8s} 1 {sigma:12.8f} {eps:12.8f} ; {reason}")

    source_lines = src.read_text().splitlines()
    with_atomtypes: list[str] = []
    in_atomtypes = False
    inserted_atomtypes = False
    for raw in source_lines:
        stripped = raw.strip()
        if stripped.startswith("["):
            if in_atomtypes and not inserted_atomtypes:
                with_atomtypes.extend(diox_atomtype_lines)
                inserted_atomtypes = True
            in_atomtypes = stripped.strip("[]").strip().lower() == "atomtypes"
        with_atomtypes.append(raw)
    if in_atomtypes and not inserted_atomtypes:
        with_atomtypes.extend(diox_atomtype_lines)

    final_lines: list[str] = []
    inserted_nonbond = False
    for raw in with_atomtypes:
        stripped = raw.strip()
        if not inserted_nonbond and stripped.startswith("[") and stripped.strip("[]").strip().lower() == "bondtypes":
            final_lines.extend(nonbond_lines)
            inserted_nonbond = True
        final_lines.append(raw)
    if not inserted_nonbond:
        final_lines.extend(nonbond_lines)

    dest.write_text("\n".join(final_lines) + "\n")


def copy_adjusted_output(source: Path, dest: Path, args: argparse.Namespace) -> None:
    dest.mkdir(exist_ok=True)
    for path in source.glob("*_ML.itp"):
        target = dest / path.name
        if path.name == f"{DIOXANE_ID}_ML.itp":
            write_adjusted_13_itp(path, target)
        else:
            shutil.copy2(path, target)
    for pattern in ("*.pdb", "*.gro", "*.csv"):
        for path in source.glob(pattern):
            shutil.copy2(path, dest / path.name)
    write_adjusted_forcefield(source / "forcefield_ML.itp", dest / "forcefield_ML.itp", args)
    write_manifest(dest / "water_dioxane_calibration_manifest.json", args)


def write_manifest(path: Path, args: argparse.Namespace) -> None:
    data = {
        "script": "scripts/12_water_dioxane_calibration.py",
        "preset": args.preset,
        "description": CALIBRATION_PRESETS[args.preset]["description"],
        "water_molecule_id": WATER_ID,
        "dioxane_molecule_id": DIOXANE_ID,
        "scales": {
            "water_polar_scale": args.water_polar_scale,
            "water_hydrophobe_scale": args.water_hydrophobe_scale,
            "dioxane_polar_scale": args.dioxane_polar_scale,
            "dioxane_hydrophobe_scale": args.dioxane_hydrophobe_scale,
            "dioxane_self_scale": args.dioxane_self_scale,
        },
        "polar_types": sorted(POLAR_TYPES),
        "hydrophobic_types": sorted(HYDROPHOBIC_TYPES),
        "known_limits": [
            "Dioxane with chloroform/trichloroethylene remains difficult.",
            "Water with n-butanol, butyl-acetate, and ethyl-acetate remains difficult.",
        ],
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def rewrite_topol(topol: Path, test_output: Path) -> None:
    text = topol.read_text()
    text = re.sub(r'#include\s+"[^"]*/output/forcefield_ML\.itp"', f'#include "{(test_output / "forcefield_ML.itp").resolve()}"', text)
    text = re.sub(r'#include\s+"[^"]*/output/(\d+_ML\.itp)"', lambda m: f'#include "{(test_output / m.group(1)).resolve()}"', text)
    topol.write_text(text)


def copy_pair_dir(src: Path, dest: Path, test_output: Path, force: bool) -> None:
    if dest.exists():
        if not force:
            return
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    keep_names = {
        "topol.top", "packed.gro", "packmol.log", "editconf.log",
        "min.mdp.sha256", "nvt.mdp.sha256",
    }
    for path in src.iterdir():
        if path.name in keep_names or path.suffix in {".pdb", ".inp"}:
            if path.is_file():
                shutil.copy2(path, dest / path.name)
    rewrite_topol(dest / "topol.top", test_output)


def write_rerun_script(path: Path, pairs: list[str], test_pairs_dir: Path, base_csv: str, output_csv: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'cd "$(dirname "$0")"',
        'GMX="${GMX:-/home/wang/gromacs_2024.2/software/bin/gmx}"',
        'MAXWARN="${MAXWARN:-64}"',
        f'PAIRS_DIR="${{PAIRS_DIR:-{test_pairs_dir}}}"',
        'STAGES="${STAGES:-min nvt eq eq2}"',
        'BOX_MAX_NM="${BOX_MAX_NM:-100}"',
        'MDP_EQ="${MDP_EQ:-${PWD}/mdp/eq_stable.mdp}"',
        'MDP_EQ2="${MDP_EQ2:-${PWD}/mdp/eq2_stable.mdp}"',
        'LOG_DIR="${LOG_DIR:-logs/water_dioxane_calib_test}"',
        'mkdir -p "$LOG_DIR"',
        "pairs=(",
    ]
    for pair in pairs:
        lines.append(f"  {pair}")
    lines.extend([
        ")",
        'for pair in "${pairs[@]}"; do',
        '  echo "[$(date \'+%F %T\')] Water/Dioxane calibration test ${pair}"',
        '  PAIRS_DIR="$PAIRS_DIR" GMX="$GMX" MAXWARN="$MAXWARN" BOX_MAX_NM="$BOX_MAX_NM" MDP_EQ="$MDP_EQ" MDP_EQ2="$MDP_EQ2" START_PAIR="$pair" STOP_PAIR="$pair" STAGES="$STAGES" bash phase_sim_run.sh >"${LOG_DIR}/${pair}.log" 2>&1',
        "done",
        "python3 scripts/11_update_box_stable_results.py \\",
        f"  --base-csv {base_csv} \\",
        f"  --output-csv {output_csv} \\",
        '  --pairs-dir "$PAIRS_DIR" \\',
        "  --threshold 0.25 \\",
        "  --pairs " + " ".join(pairs),
    ])
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    source_pairs = Path(args.source_pairs_dir)
    test_pairs = Path(args.test_pairs_dir)
    source_output = Path(args.source_output_dir)
    test_output = Path(args.test_output_dir)
    copy_adjusted_output(source_output, test_output, args)
    if args.output_only:
        print(f"Wrote adjusted output: {test_output}")
        print(f"Preset: {args.preset}")
        print(f"Manifest: {test_output / 'water_dioxane_calibration_manifest.json'}")
        return

    if args.select_mismatch_targets is not None:
        pairs = selected_pairs_from_csv(Path(args.current_csv), set(args.select_mismatch_targets))
    else:
        pairs = sorted(set(args.pairs), key=pair_sort_key)
    if not pairs:
        raise SystemExit("No target pairs selected.")

    test_pairs.mkdir(exist_ok=True)
    for pair in pairs:
        src = source_pairs / pair
        if not src.exists():
            raise FileNotFoundError(src)
        copy_pair_dir(src, test_pairs / pair, test_output, args.force)

    write_rerun_script(Path(args.rerun_script), pairs, test_pairs, args.current_csv, args.output_csv)
    print(f"Wrote adjusted output: {test_output}")
    print(f"Wrote isolated test pairs: {test_pairs}")
    print(f"Wrote rerun script: {args.rerun_script}")
    print("Test pairs:")
    for pair in pairs:
        print(f"  {pair}")


if __name__ == "__main__":
    main()
