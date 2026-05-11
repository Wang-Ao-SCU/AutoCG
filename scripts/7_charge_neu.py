import os
import glob

def neutralize_itp_files(file_pattern='./output/*ML.itp'):
    files = glob.glob(file_pattern)
    if not files:
        return
    for filename in files:
        with open(filename, 'r') as f:
            lines = f.readlines()
        new_lines = []
        in_atoms_section = False
        atom_line_indices = []
        charges = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                section_name = stripped[1:-1].strip()
                in_atoms_section = section_name == 'atoms'
                new_lines.append(line)
                continue
            if in_atoms_section and stripped and (not stripped.startswith(';')):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        q = float(parts[6])
                        charges.append(q)
                        atom_line_indices.append(i)
                    except ValueError:
                        pass
                new_lines.append(line)
            else:
                new_lines.append(line)
        if not charges:
            continue
        net_charge = sum(charges)
        if abs(net_charge) < 1e-05:
            continue
        indices_to_modify = []
        if net_charge < 0:
            indices_to_modify = [idx for idx, q in enumerate(charges) if q < 0]
        else:
            indices_to_modify = [idx for idx, q in enumerate(charges) if q > 0]
        if not indices_to_modify:
            continue
        adjustment = net_charge / len(indices_to_modify)
        for charge_idx in indices_to_modify:
            line_idx = atom_line_indices[charge_idx]
            original_line = lines[line_idx]
            parts = original_line.split()
            old_charge = float(parts[6])
            new_charge = old_charge - adjustment
            parts[6] = f'{new_charge:.6f}'
            new_line = '\t'.join(parts) + '\n'
            new_lines[line_idx] = new_line
        output_filename = filename
        with open(output_filename, 'w') as f:
            f.writelines(new_lines)
if __name__ == '__main__':
    neutralize_itp_files()
