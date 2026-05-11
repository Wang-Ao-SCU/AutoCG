def calculate_net_charge(itp_filename):
    total_charge = 0.0
    in_atoms_section = False
    atom_count = 0
    try:
        with open(itp_filename, 'r') as f:
            for line in f:
                clean_line = line.strip().split(';')[0].strip()
                if not clean_line:
                    continue
                if clean_line.startswith('['):
                    section_name = clean_line.strip('[] ').lower()
                    if section_name == 'atoms':
                        in_atoms_section = True
                        continue
                    else:
                        if in_atoms_section:
                            break
                        continue
                if in_atoms_section:
                    parts = clean_line.split()
                    if len(parts) >= 7:
                        try:
                            charge = float(parts[6])
                            total_charge += charge
                            atom_count += 1
                        except ValueError:
                            continue
        return (total_charge, atom_count)
    except FileNotFoundError:
        return (None, 0)
filename = './output/4_ML.itp'
net_charge, count = calculate_net_charge(filename)
if count > 0:
    pass
