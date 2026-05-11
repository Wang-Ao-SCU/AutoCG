#!/usr/bin/env python3
from pathlib import Path


OUTPUT_DIR = Path("./output")
M3_DIR = OUTPUT_DIR / "M3"


def section_name(line):
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped.strip("[]").strip().lower()
    return None


def load_m3_atoms(m3_itp):
    atoms = {}
    current_section = None
    for line in m3_itp.read_text(encoding="utf-8", errors="replace").splitlines():
        name = section_name(line)
        if name is not None:
            current_section = name
            continue
        stripped = line.strip()
        if current_section != "atoms" or not stripped or stripped.startswith(";"):
            continue
        data = stripped.split(";", 1)[0].strip()
        parts = data.split()
        if len(parts) >= 7:
            atoms[parts[0]] = {
                "bead_type": parts[1],
                "atom_name": parts[4],
                "charge": parts[6],
            }
    return atoms


def replace_atom_fields(line, m3_atoms):
    if ";" in line:
        data, comment = line.split(";", 1)
        comment = ";" + comment.rstrip("\n")
    else:
        data = line.rstrip("\n")
        comment = ""
    parts = data.split()
    if len(parts) < 7:
        return line
    atom_id = parts[0]
    if atom_id not in m3_atoms:
        return line
    m3_atom = m3_atoms[atom_id]
    parts[1] = m3_atom["bead_type"]
    parts[4] = m3_atom["atom_name"]
    parts[6] = m3_atom["charge"]
    new_line = (
        f"{parts[0]:>5} {parts[1]:>4} {parts[2]:>4} {parts[3]:>4} "
        f"{parts[4]:>4} {parts[5]:>4}"
    )
    if len(parts) > 6:
        new_line += "".join(f" {part:>9}" for part in parts[6:])
    if comment:
        new_line += comment
    return new_line + "\n"


def build_m3_ml_itp(m3_itp):
    stem = m3_itp.stem
    if not stem.endswith("_m3"):
        return False
    mol_id = stem[:-3]
    ml_itp = OUTPUT_DIR / f"{mol_id}_ML.itp"
    if not ml_itp.exists():
        return False

    m3_atoms = load_m3_atoms(m3_itp)
    if not m3_atoms:
        return False

    current_section = None
    output_lines = []
    for line in ml_itp.read_text(encoding="utf-8", errors="replace").splitlines(True):
        name = section_name(line)
        if name is not None:
            current_section = name
            output_lines.append(line)
            continue
        stripped = line.strip()
        if current_section == "atoms" and stripped and not stripped.startswith(";"):
            output_lines.append(replace_atom_fields(line, m3_atoms))
        else:
            output_lines.append(line)

    out_path = M3_DIR / f"{mol_id}_m3_ML.itp"
    out_path.write_text("".join(output_lines), encoding="utf-8")
    return True


def main():
    if not M3_DIR.exists():
        return
    for m3_itp in sorted(M3_DIR.glob("*_m3.itp")):
        build_m3_ml_itp(m3_itp)


if __name__ == "__main__":
    main()
