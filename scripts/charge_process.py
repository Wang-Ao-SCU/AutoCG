import sys
import os
import argparse

def extract_charges(source_file):
    charges = []
    if not os.path.exists(source_file):
        return []
    with open(source_file, 'r') as f:
        lines = f.readlines()
    in_atom_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('@<TRIPOS>ATOM'):
            in_atom_section = True
            continue
        elif stripped.startswith('@<TRIPOS>'):
            in_atom_section = False
            continue
        if in_atom_section and stripped:
            parts = stripped.split()
            if len(parts) >= 9:
                try:
                    charges.append(float(parts[8]))
                except ValueError:
                    pass
    return charges

def inject_charges_exact_width(template_file, charge_list, output_file):
    with open(template_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    atom_counter = 0
    in_atom_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('@<TRIPOS>ATOM'):
            in_atom_section = True
            new_lines.append(line)
            continue
        elif stripped.startswith('@<TRIPOS>'):
            in_atom_section = False
            new_lines.append(line)
            continue
        if not in_atom_section or not stripped:
            new_lines.append(line)
            continue
        if len(line.rstrip()) > 50:
            if atom_counter < len(charge_list):
                new_val = charge_list[atom_counter]
                atom_counter += 1
                original_content = line.rstrip('\r\n')
                line_ending = line[len(original_content):]
                prefix = original_content[:-10]
                new_suffix = f'{new_val:>10.4f}'
                new_line = prefix + new_suffix + line_ending
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    with open(output_file, 'w') as out:
        out.writelines(new_lines)

def main():
    parser = argparse.ArgumentParser(description='Exact Mol2 Charge Replacer')
    parser.add_argument('-c', '--charges', nargs='+', required=True, help='Source files (X)')
    parser.add_argument('-f', '--formats', nargs='+', required=True, help='Template files (Y)')
    parser.add_argument('-o', '--outputs', nargs='+', required=True, help='Output files (Z)')
    args = parser.parse_args()
    if not len(args.charges) == len(args.formats) == len(args.outputs):
        sys.exit(1)
    for i in range(len(args.charges)):
        src = args.charges[i]
        tpl = args.formats[i]
        out = args.outputs[i]
        q_list = extract_charges(src)
        if not q_list:
            continue
        inject_charges_exact_width(tpl, q_list, out)
if __name__ == '__main__':
    main()
