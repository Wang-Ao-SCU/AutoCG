import pandas as pd
import sys
BOND_COLS = {'atom1': 'bead_type1', 'atom2': 'bead_type2', 'r0': 'predicted_r', 'k': 'predicted_k'}
ANGLE_COLS = {'side1': 'bead_type1', 'center': 'bead_type2', 'side2': 'bead_type3', 'theta0': 'predicted_r', 'k': 'predicted_r'}

def validate_columns(df, required_cols, filename):
    missing = [col for col in required_cols.values() if col not in df.columns]
    if missing:
        sys.exit(1)

def load_predictions(bonds_csv, angles_csv):
    try:
        bonds_df = pd.read_csv(bonds_csv)
    except FileNotFoundError:
        sys.exit(1)
    try:
        angles_df = pd.read_csv(angles_csv)
    except FileNotFoundError:
        sys.exit(1)
    possible_i_names = ['atom_i', 'atom1', 'i', 'ai']
    possible_j_names = ['atom_j', 'atom2', 'j', 'aj']
    if BOND_COLS['atom1'] not in bonds_df.columns:
        for name in possible_i_names:
            if name in bonds_df.columns:
                BOND_COLS['atom1'] = name
                break
    if BOND_COLS['atom2'] not in bonds_df.columns:
        for name in possible_j_names:
            if name in bonds_df.columns:
                BOND_COLS['atom2'] = name
                break
    validate_columns(bonds_df, BOND_COLS, bonds_csv)
    if ANGLE_COLS['center'] not in angles_df.columns:
        for name in ['atom_j', 'atom2', 'center', 'j']:
            if name in angles_df.columns:
                ANGLE_COLS['center'] = name
                break
    validate_columns(angles_df, ANGLE_COLS, angles_csv)
    bond_lookup = {}
    for idx, row in bonds_df.iterrows():
        try:
            a1 = str(row[BOND_COLS['atom1']]).strip()
            a2 = str(row[BOND_COLS['atom2']]).strip()
            key = tuple(sorted([a1, a2]))
            r0 = float(row[BOND_COLS['r0']])
            k_val = float(row[BOND_COLS['k']])
            bond_lookup[key] = (r0, k_val)
        except ValueError as e:
            pass
    angle_lookup = {}
    for idx, row in angles_df.iterrows():
        try:
            a_center = str(row[ANGLE_COLS['center']]).strip()
            a_side1 = str(row[ANGLE_COLS['side1']]).strip()
            a_side2 = str(row[ANGLE_COLS['side2']]).strip()
            sides = sorted([a_side1, a_side2])
            key = (sides[0], a_center, sides[1])
            theta0 = float(row[ANGLE_COLS['theta0']])
            k_val = float(row[ANGLE_COLS['k']])
            angle_lookup[key] = (theta0, k_val)
        except ValueError as e:
            pass
    return (bond_lookup, angle_lookup)

def process_itp(input_file, output_file, bond_lookup, angle_lookup):
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
    new_lines = []
    current_section = None
    updated_bonds = 0
    updated_angles = 0
    skip_markers = ['; updated by script (SMILES match)', '; updated by script']
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            current_section = stripped_line[1:-1].strip()
            new_lines.append(line)
            continue
        if not stripped_line or stripped_line.startswith(';'):
            new_lines.append(line)
            continue
        if any((marker in line for marker in skip_markers)):
            new_lines.append(line)
            continue
        if current_section == 'bondtypes':
            parts = stripped_line.split()
            if len(parts) >= 5:
                atom_i = parts[0]
                atom_j = parts[1]
                func = parts[2]
                key = tuple(sorted([atom_i, atom_j]))
                if key in bond_lookup:
                    new_r0, new_k = bond_lookup[key]
                    new_line = f'{atom_i:<7} {atom_j:<7} {func:>4} {new_r0:>10.4f} {new_k:>10.1f} ; updated by ML (SMILES match)\n'
                    new_lines.append(new_line)
                    updated_bonds += 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        elif current_section == 'angletypes':
            parts = stripped_line.split()
            if len(parts) >= 6:
                atom_i = parts[0]
                atom_j = parts[1]
                atom_k = parts[2]
                func = parts[3]
                sides = sorted([atom_i, atom_k])
                key = (sides[0], atom_j, sides[1])
                if key in angle_lookup:
                    new_theta, new_k = angle_lookup[key]
                    new_line = f'{atom_i:<7} {atom_j:<7} {atom_k:<7} {func:>4} {new_theta:>10.3f} {new_k:>10.1f} ; updated by ML (SMILES match)\n'
                    new_lines.append(new_line)
                    updated_angles += 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    with open(output_file, 'w') as f_out:
        f_out.writelines(new_lines)
if __name__ == '__main__':
    itp_file = './output/forcefield.itp'
    bonds_csv = './output/bonds_with_predictions.csv'
    angles_csv = './output/angles_with_predictions.csv'
    output_itp = './output/forcefield_ML.itp'
    bonds_map, angles_map = load_predictions(bonds_csv, angles_csv)
    process_itp(itp_file, output_itp, bonds_map, angles_map)
