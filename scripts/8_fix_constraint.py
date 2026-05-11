import os
import glob
import re
import shutil

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process_itp_file(filepath):
    backup_path = filepath + '.bak'
    shutil.copy(filepath, backup_path)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    new_lines = []
    constraints_to_move = []
    current_section = None
    FORCE_CONSTANT = '10000'
    processed_content_lines = []
    bonds_section_end_index = -1
    in_constraints_section = False
    count_moved = 0

    def section_name_from_line(line):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            return stripped[1:-1].strip().lower()
        return None

    for i, line in enumerate(lines):
        stripped = line.strip()
        section_name = section_name_from_line(line)
        if section_name is not None:
            current_section = section_name
            in_constraints_section = section_name == 'constraints'
            processed_content_lines.append(line)
            continue
        if current_section == 'bonds' and stripped != '':
            pass
        if in_constraints_section and stripped and (not stripped.startswith(';')):
            parts = stripped.split()
            if len(parts) >= 4 and is_float(parts[3]):
                dist = float(parts[3])
                atom_i = parts[0]
                atom_j = parts[1]
                bond_line = f'   {atom_i:>4} {atom_j:>4}    1     {parts[3]}   {FORCE_CONSTANT} ; Converted from constraint\n'
                constraints_to_move.append(bond_line)
                count_moved += 1
                processed_content_lines.append(f'; [Removed] Constraint {atom_i}-{atom_j} len={dist} moved to bonds\n')
            else:
                processed_content_lines.append(line)
        else:
            processed_content_lines.append(line)
    if count_moved == 0:
        return
    final_output = []
    bonds_inserted = False
    in_bonds = False
    has_bonds_section = False
    for line in lines:
        if section_name_from_line(line) == 'bonds':
            has_bonds_section = True
            break
    if not has_bonds_section:
        final_output = processed_content_lines + ['\n[ bonds ]\n'] + constraints_to_move
    else:
        for line in processed_content_lines:
            final_output.append(line)
            stripped = line.strip()
            current_sect = section_name_from_line(line)
            if current_sect is not None:
                if current_sect == 'bonds':
                    in_bonds = True
                elif in_bonds:
                    if not bonds_inserted:
                        final_output.insert(-1, '\n; --- MDI Bridge Bonds (Auto-converted) ---\n')
                        for bl in constraints_to_move:
                            final_output.insert(-1, bl)
                        bonds_inserted = True
                    in_bonds = False
        if in_bonds and (not bonds_inserted):
            final_output.append('\n; --- MDI Bridge Bonds (Auto-converted) ---\n')
            final_output.extend(constraints_to_move)
    with open(filepath, 'w') as f:
        f.writelines(final_output)

def main():
    files = glob.glob('./output/*_ML.itp')
    files = list(set(files))
    if not files:
        return
    for file_path in files:
        process_itp_file(file_path)
if __name__ == '__main__':
    main()
