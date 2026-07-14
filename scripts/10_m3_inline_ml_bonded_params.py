#!/usr/bin/env python3
from pathlib import Path
import sys


OUTPUT_DIR = Path("./output")
M3_DIR = OUTPUT_DIR / "M3"
FORCEFIELD = OUTPUT_DIR / "forcefield_ML.itp"


def section_name(line):
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped.strip("[]").strip().lower()
    return None


def split_data_comment(line):
    if ";" in line:
        data, comment = line.split(";", 1)
        return data.rstrip(), ";" + comment.rstrip("\n")
    return line.rstrip("\n"), ""


def canonical_pair(a, b):
    return tuple(sorted((a, b)))


def canonical_angle(a, b, c):
    forward = (a, b, c)
    reverse = (c, b, a)
    return min(forward, reverse)


def canonical_dihedral(a, b, c, d):
    forward = (a, b, c, d)
    reverse = (d, c, b, a)
    return min(forward, reverse)


def parse_forcefield_types(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing force-field file: {path}")

    bonded = {"bonds": {}, "angles": {}, "dihedrals": {}}
    current_section = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = section_name(line)
        if name is not None:
            current_section = name
            continue

        data, comment = split_data_comment(line)
        parts = data.split()
        if not parts:
            continue

        if current_section == "bondtypes" and len(parts) >= 5:
            key = canonical_pair(parts[0], parts[1])
            bonded["bonds"][key] = {
                "func": parts[2],
                "params": parts[3:],
                "comment": comment,
            }
        elif current_section == "angletypes" and len(parts) >= 6:
            key = canonical_angle(parts[0], parts[1], parts[2])
            bonded["angles"][key] = {
                "func": parts[3],
                "params": parts[4:],
                "comment": comment,
            }
        elif current_section == "dihedraltypes" and len(parts) >= 7:
            key = canonical_dihedral(parts[0], parts[1], parts[2], parts[3])
            bonded["dihedrals"][key] = {
                "func": parts[4],
                "params": parts[5:],
                "comment": comment,
            }

    return bonded


def parse_cg_atom_types(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing CG molecule ITP: {path}")

    atoms = {}
    current_section = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = section_name(line)
        if name is not None:
            current_section = name
            continue

        data, _ = split_data_comment(line)
        parts = data.split()
        if current_section == "atoms" and len(parts) >= 2:
            atoms[parts[0]] = parts[1]

    if not atoms:
        raise ValueError(f"No atoms found in CG molecule ITP: {path}")
    return atoms


def format_bonded_line(atom_ids, type_def, comment):
    prefix = "".join(f"{atom_id:>5}" for atom_id in atom_ids)
    params = " ".join(type_def["params"])
    suffix = f" {type_def['func']:>4} {params}"
    comments = []
    if type_def["comment"]:
        comments.append(type_def["comment"].lstrip(";").strip())
    if comment:
        comments.append(comment.lstrip(";").strip())
    if comments:
        suffix += " ; " + " | ".join(comments)
    return prefix + suffix + "\n"


def should_inline(section, parts):
    min_lengths = {"bonds": 3, "angles": 4, "dihedrals": 5}
    full_lengths = {"bonds": 5, "angles": 6, "dihedrals": 7}
    return len(parts) >= min_lengths[section] and len(parts) < full_lengths[section]


def lookup_key(section, atom_ids, cg_atoms):
    try:
        cg_types = [cg_atoms[atom_id] for atom_id in atom_ids]
    except KeyError as exc:
        raise KeyError(f"Atom id {exc.args[0]} is missing from source CG atom map")

    if section == "bonds":
        return canonical_pair(cg_types[0], cg_types[1]), cg_types
    if section == "angles":
        return canonical_angle(cg_types[0], cg_types[1], cg_types[2]), cg_types
    return canonical_dihedral(cg_types[0], cg_types[1], cg_types[2], cg_types[3]), cg_types


def inline_file(m3_ml_itp, bonded_types):
    mol_id = m3_ml_itp.name.removesuffix("_m3_ML.itp")
    cg_ml_itp = OUTPUT_DIR / f"{mol_id}_ML.itp"
    cg_atoms = parse_cg_atom_types(cg_ml_itp)

    current_section = None
    output_lines = []
    stats = {"bonds": 0, "angles": 0, "dihedrals": 0}
    missing = []

    for line_no, line in enumerate(m3_ml_itp.read_text(encoding="utf-8", errors="replace").splitlines(True), start=1):
        name = section_name(line)
        if name is not None:
            current_section = name
            output_lines.append(line)
            continue

        data, comment = split_data_comment(line)
        parts = data.split()
        if current_section not in stats or not parts or parts[0].startswith(";"):
            output_lines.append(line)
            continue

        section = current_section
        if not should_inline(section, parts):
            output_lines.append(line)
            continue

        atom_count = {"bonds": 2, "angles": 3, "dihedrals": 4}[section]
        atom_ids = parts[:atom_count]
        key, cg_types = lookup_key(section, atom_ids, cg_atoms)
        type_def = bonded_types[section].get(key)
        if type_def is None:
            missing.append((line_no, section, "-".join(cg_types), line.rstrip("\n")))
            output_lines.append(line)
            continue

        output_lines.append(format_bonded_line(atom_ids, type_def, comment))
        stats[section] += 1

    m3_ml_itp.write_text("".join(output_lines), encoding="utf-8")
    return stats, missing


def main():
    if not M3_DIR.exists():
        raise FileNotFoundError(f"Missing M3 output directory: {M3_DIR}")

    bonded_types = parse_forcefield_types(FORCEFIELD)
    targets = sorted(M3_DIR.glob("*_m3_ML.itp"))
    if not targets:
        raise FileNotFoundError(f"No M3 ML molecule ITP files found in {M3_DIR}")

    total = {"bonds": 0, "angles": 0, "dihedrals": 0}
    all_missing = []
    for target in targets:
        stats, missing = inline_file(target, bonded_types)
        for section, count in stats.items():
            total[section] += count
        all_missing.extend((target, *item) for item in missing)

    print(
        "Inlined M3 bonded parameters: "
        f"bonds={total['bonds']}, angles={total['angles']}, dihedrals={total['dihedrals']}"
    )

    if all_missing:
        for target, line_no, section, cg_key, line in all_missing:
            print(
                f"Missing {section} type in {FORCEFIELD}: {target}:{line_no} "
                f"CG={cg_key} line='{line}'",
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
