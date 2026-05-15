#!/usr/bin/env python3
"""Apply group-aware Water/Dioxane empirical corrections to AutoCG output.

This script turns the Water/Dioxane lessons from the solvent phase benchmark
into a reusable force-field output-layer correction. It is not tied to the
phase-separation workflow: it only reads an AutoCG ``output`` directory and
writes a calibrated output directory.

Core experience encoded here:

* Water should interact more strongly with polar/hetero-rich coarse groups and
  more weakly with hydrophobic groups.
* Dioxane should not reuse generic CG12/CG4 interactions blindly. It gets
  molecule-specific bead types so corrections are local to dioxane chemistry.
* Dioxane self attraction and Dioxane-hydrophobe attraction are reduced, while
  Dioxane-polar attraction is strengthened.

Preset ``v2`` is the stronger calibrated default. Preset ``v1`` keeps the
earlier conservative coefficients.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


WATER_ID = 26
DIOXANE_ID = 13

POLAR_TYPES = {
    "CG0", "CG1", "CG2", "CG5", "CG6", "CG10", "CG11", "CG12",
    "CG13", "CG14", "CG15", "CG16", "CG17", "DIOX_O", "DIOX_C",
}
HYDROPHOBIC_TYPES = {"CG3", "CG4", "CG7", "CG8", "CG9", "CG18", "CG19"}
ION_TYPES = {"SOD", "CLA"}

PRESETS = {
    "v1": {
        "description": "conservative Water/Dioxane group correction",
        "water_polar_scale": 1.25,
        "water_hydrophobe_scale": 0.45,
        "dioxane_polar_scale": 1.35,
        "dioxane_hydrophobe_scale": 0.60,
        "dioxane_self_scale": 0.55,
    },
    "v2": {
        "description": "stronger default Water/Dioxane group correction",
        "water_polar_scale": 1.80,
        "water_hydrophobe_scale": 0.20,
        "dioxane_polar_scale": 2.00,
        "dioxane_hydrophobe_scale": 0.25,
        "dioxane_self_scale": 0.20,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-output-dir", default="output")
    parser.add_argument("--output-dir", default="output_water_dioxane_experience")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="v2")
    parser.add_argument("--water-polar-scale", type=float)
    parser.add_argument("--water-hydrophobe-scale", type=float)
    parser.add_argument("--dioxane-polar-scale", type=float)
    parser.add_argument("--dioxane-hydrophobe-scale", type=float)
    parser.add_argument("--dioxane-self-scale", type=float)
    parser.add_argument("--force", action="store_true", help="replace --output-dir if it exists")
    args = parser.parse_args()
    apply_preset(args)
    return args


def apply_preset(args: argparse.Namespace) -> None:
    preset = PRESETS[args.preset]
    for name in (
        "water_polar_scale",
        "water_hydrophobe_scale",
        "dioxane_polar_scale",
        "dioxane_hydrophobe_scale",
        "dioxane_self_scale",
    ):
        if getattr(args, name) is None:
            setattr(args, name, preset[name])


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
                params[tuple(sorted((parts[0], parts[1])))] = (float(parts[3]), float(parts[4]))
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


def dioxane_base_type(bead: str) -> str:
    if bead == "DIOX_O":
        return "CG12"
    if bead == "DIOX_C":
        return "CG4"
    return bead


def write_dioxane_itp(src: Path, dest: Path) -> None:
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
                if parts[1] == "CG12":
                    line = line.replace("CG12", "DIOX_O", 1)
                elif parts[1] == "CG4":
                    line = line.replace("CG4", "DIOX_C", 1)
        out.append(line)
    dest.write_text("\n".join(out) + "\n")


def build_override_rows(
    atomtypes: dict[str, tuple[float, float, float, float]],
    params: dict[tuple[str, str], tuple[float, float]],
    args: argparse.Namespace,
) -> list[tuple[str, str, float, float, str]]:
    extended_atomtypes = atomtypes | {
        "DIOX_O": atomtypes["CG12"],
        "DIOX_C": atomtypes["CG4"],
    }
    rows: list[tuple[str, str, float, float, str]] = []
    all_types = sorted(atomtypes)

    for diox in ("DIOX_O", "DIOX_C"):
        for other in all_types + ["DIOX_O", "DIOX_C"]:
            if other in ION_TYPES:
                continue
            sigma0, eps0 = mixed_param(
                extended_atomtypes,
                params,
                dioxane_base_type(diox),
                dioxane_base_type(other),
            )
            if other.startswith("DIOX_"):
                scale = args.dioxane_self_scale
                reason = "experience: reduce dioxane self clustering"
            elif other in HYDROPHOBIC_TYPES:
                scale = args.dioxane_hydrophobe_scale
                reason = "experience: reduce dioxane hydrophobe attraction"
            elif other in POLAR_TYPES or other == "W":
                scale = args.dioxane_polar_scale
                reason = "experience: strengthen dioxane polar mixing"
            else:
                scale = 1.0
                reason = "experience: dioxane neutral"
            rows.append((diox, other, sigma0, eps0 * scale, reason))

    for other in all_types + ["DIOX_O", "DIOX_C"]:
        if other in {"W", *ION_TYPES}:
            continue
        sigma0, eps0 = mixed_param(atomtypes, params, "W", dioxane_base_type(other))
        if other in HYDROPHOBIC_TYPES:
            scale = args.water_hydrophobe_scale
            reason = "experience: reduce water hydrophobe attraction"
        elif other in POLAR_TYPES or other.startswith("DIOX_"):
            scale = args.water_polar_scale
            reason = "experience: strengthen water polar mixing"
        else:
            scale = 1.0
            reason = "experience: water neutral"
        rows.append(("W", other, sigma0, eps0 * scale, reason))

    deduped: list[tuple[str, str, float, float, str]] = []
    seen: set[tuple[str, str]] = set()
    for a, b, sigma, eps, reason in rows:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((a, b, sigma, eps, reason))
    return deduped


def insert_lines(src: Path, dest: Path, args: argparse.Namespace) -> None:
    atomtypes = read_atomtypes(src)
    params = read_nonbond_params(src)
    rows = build_override_rows(atomtypes, params, args)

    diox_atomtypes = [
        "; Water/Dioxane experience atomtypes",
        f"DIOX_O   {atomtypes['CG12'][0]:.4f}  {atomtypes['CG12'][1]:.3f}  A  {atomtypes['CG12'][2]:.4f}  {atomtypes['CG12'][3]:.4f}",
        f"DIOX_C   {atomtypes['CG4'][0]:.4f}  {atomtypes['CG4'][1]:.3f}  A  {atomtypes['CG4'][2]:.4f}  {atomtypes['CG4'][3]:.4f}",
    ]
    nonbond_lines = [
        "",
        "[ nonbond_params ]",
        "; Water/Dioxane group-experience overrides: i j func sigma epsilon",
    ]
    for a, b, sigma, eps, reason in rows:
        nonbond_lines.append(f"{a:8s} {b:8s} 1 {sigma:12.8f} {eps:12.8f} ; {reason}")

    source_lines = src.read_text().splitlines()
    with_atomtypes: list[str] = []
    in_atomtypes = False
    inserted_atomtypes = False
    for raw in source_lines:
        stripped = raw.strip()
        if stripped.startswith("["):
            if in_atomtypes and not inserted_atomtypes:
                with_atomtypes.extend(diox_atomtypes)
                inserted_atomtypes = True
            in_atomtypes = stripped.strip("[]").strip().lower() == "atomtypes"
        with_atomtypes.append(raw)
    if in_atomtypes and not inserted_atomtypes:
        with_atomtypes.extend(diox_atomtypes)

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


def write_manifest(path: Path, args: argparse.Namespace) -> None:
    manifest = {
        "script": "scripts/12_water_dioxane_forcefield_experience.py",
        "purpose": "general force-field output-layer correction for Water/Dioxane-related group interactions",
        "preset": args.preset,
        "description": PRESETS[args.preset]["description"],
        "scales": {
            "water_polar_scale": args.water_polar_scale,
            "water_hydrophobe_scale": args.water_hydrophobe_scale,
            "dioxane_polar_scale": args.dioxane_polar_scale,
            "dioxane_hydrophobe_scale": args.dioxane_hydrophobe_scale,
            "dioxane_self_scale": args.dioxane_self_scale,
        },
        "polar_types": sorted(POLAR_TYPES),
        "hydrophobic_types": sorted(HYDROPHOBIC_TYPES),
        "local_dioxane_types": {"CG12": "DIOX_O", "CG4": "DIOX_C"},
        "known_limits": [
            "Dioxane with chloroform/trichloroethylene may still need halogen-specific correction.",
            "Water with n-butanol, butyl-acetate, and ethyl-acetate may need oxygenated-organic-specific correction.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def copy_output(src_dir: Path, out_dir: Path, force: bool) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists; use --force to replace it")
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)


def main() -> None:
    args = parse_args()
    src_dir = Path(args.input_output_dir)
    out_dir = Path(args.output_dir)
    copy_output(src_dir, out_dir, args.force)

    ff = out_dir / "forcefield_ML.itp"
    diox = out_dir / f"{DIOXANE_ID}_ML.itp"
    if not ff.exists():
        raise FileNotFoundError(ff)
    if diox.exists():
        write_dioxane_itp(diox, diox)
    insert_lines(ff, ff, args)
    write_manifest(out_dir / "water_dioxane_experience_manifest.json", args)

    print(f"Wrote Water/Dioxane experience output: {out_dir}")
    print(f"Preset: {args.preset}")
    print(f"Manifest: {out_dir / 'water_dioxane_experience_manifest.json'}")


if __name__ == "__main__":
    main()
