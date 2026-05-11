import os
import re

def parse_forcefield_itp(forcefield_file):
    forcefield_data = {'bondtypes': {}, 'angletypes': {}, 'dihedraltypes': {}}
    current_section = None
    with open(forcefield_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('['):
                if 'bondtypes' in line:
                    current_section = 'bondtypes'
                elif 'angletypes' in line:
                    current_section = 'angletypes'
                elif 'dihedraltypes' in line:
                    current_section = 'dihedraltypes'
                else:
                    current_section = None
                continue
            if current_section in ['bondtypes', 'angletypes']:
                data_part = line.split(';')[0].strip()
                if not data_part:
                    continue
                parts = data_part.split()
                if len(parts) >= 3:
                    if current_section == 'bondtypes':
                        atom1, atom2, func = (parts[0], parts[1], parts[2])
                        key = f'{atom1}-{atom2}'
                        forcefield_data['bondtypes'][key] = func
                    elif current_section == 'angletypes':
                        if len(parts) >= 4:
                            atom1, atom2, atom3, func = (parts[0], parts[1], parts[2], parts[3])
                            key = f'{atom1}-{atom2}-{atom3}'
                            forcefield_data['angletypes'][key] = func
    return forcefield_data

def get_atom_type_mapping(itp_file):
    atom_mapping = {}
    in_atoms_section = False
    with open(itp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('['):
                if 'atoms' in line.lower():
                    in_atoms_section = True
                else:
                    in_atoms_section = False
                continue
            if in_atoms_section:
                parts = line.split()
                if len(parts) >= 2:
                    atom_id = parts[0]
                    atom_type = parts[1]
                    atom_mapping[atom_id] = atom_type
    return atom_mapping

def update_itp_file(itp_file, forcefield_data, atom_mapping):
    updated_lines = []
    current_section = None
    bonds_updated = 0
    angles_updated = 0
    with open(itp_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith('['):
            current_section = line
            updated_lines.append(line + '\n')
            i += 1
            continue
        if current_section and 'bonds' in current_section and (not line.startswith(';')) and line.strip():
            parts = line.split()
            if len(parts) >= 3:
                atom1, atom2, current_func = (parts[0], parts[1], parts[2])
                type1 = atom_mapping.get(atom1)
                type2 = atom_mapping.get(atom2)
                if type1 and type2:
                    key1 = f'{type1}-{type2}'
                    key2 = f'{type2}-{type1}'
                    new_func = None
                    if key1 in forcefield_data['bondtypes']:
                        new_func = forcefield_data['bondtypes'][key1]
                    elif key2 in forcefield_data['bondtypes']:
                        new_func = forcefield_data['bondtypes'][key2]
                    if new_func and new_func != current_func:
                        parts[2] = new_func
                        new_line = ' '.join(parts) + ' ' + ' '.join(parts[3:]) if len(parts) > 3 else ' '.join(parts)
                        updated_lines.append(new_line + '\n')
                        bonds_updated += 1
                    else:
                        updated_lines.append(line + '\n')
                else:
                    updated_lines.append(line + '\n')
            else:
                updated_lines.append(line + '\n')
        elif current_section and 'angles' in current_section and (not line.startswith(';')) and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                atom1, atom2, atom3, current_func = (parts[0], parts[1], parts[2], parts[3])
                type1 = atom_mapping.get(atom1)
                type2 = atom_mapping.get(atom2)
                type3 = atom_mapping.get(atom3)
                if type1 and type2 and type3:
                    key = f'{type1}-{type2}-{type3}'
                    if key in forcefield_data['angletypes']:
                        new_func = forcefield_data['angletypes'][key]
                        if new_func != current_func:
                            parts[3] = new_func
                            new_line = ' '.join(parts) + ' ' + ' '.join(parts[4:]) if len(parts) > 4 else ' '.join(parts)
                            updated_lines.append(new_line + '\n')
                            angles_updated += 1
                        else:
                            updated_lines.append(line + '\n')
                    else:
                        updated_lines.append(line + '\n')
                else:
                    updated_lines.append(line + '\n')
            else:
                updated_lines.append(line + '\n')
        else:
            updated_lines.append(line + '\n')
        i += 1
    with open(itp_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    return (bonds_updated, angles_updated)

def main():
    directory = './output/'
    forcefield_file = os.path.join(directory, 'forcefield.itp')
    if not os.path.exists(forcefield_file):
        return
    forcefield_data = parse_forcefield_itp(forcefield_file)
    ml_itp_files = []
    for file in os.listdir(directory):
        if file.endswith('.itp') and '_ML' in file and (file != 'forcefield.itp'):
            ml_itp_files.append(os.path.join(directory, file))
    if not ml_itp_files:
        return
    for file in ml_itp_files:
        pass
    total_bonds_updated = 0
    total_angles_updated = 0
    for itp_file in ml_itp_files:
        atom_mapping = get_atom_type_mapping(itp_file)
        bonds_updated, angles_updated = update_itp_file(itp_file, forcefield_data, atom_mapping)
        total_bonds_updated += bonds_updated
        total_angles_updated += angles_updated
if __name__ == '__main__':
    main()
