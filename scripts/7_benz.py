import pandas as pd
import os
import glob

def main():
    base_dir = './output'
    csv_path = os.path.join(base_dir, 'global_smiles_to_cgtype.csv')
    ff_path = os.path.join(base_dir, 'forcefield.itp')
    target_smiles = 'c1ccccc1'
    target_cg = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[df['SMILES'] == target_smiles]
        if not row.empty:
            target_cg = row.iloc[0]['CG_Type']
        else:
            return
    else:
        return
    if os.path.exists(ff_path):
        with open(ff_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        in_nonbond = False
        modified_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                if 'nonbond_params' in stripped:
                    in_nonbond = True
                else:
                    in_nonbond = False
            if in_nonbond and (not stripped.startswith(';')) and (len(stripped) > 0):
                parts = line.split()
                if len(parts) >= 5:
                    t1, t2 = (parts[0], parts[1])
                    if t1 == target_cg and t2 == target_cg:
                        parts[3] = '0.34'
                        parts[4] = '1.51'
                        new_line = f'  {t1:<8s} {t2:<8s} {parts[2]:<5s} {parts[3]:<12s} {parts[4]:<12s}'
                        if ';' in line:
                            comment = line[line.find(';'):].strip()
                            new_line += '  ' + comment
                        new_line += '\n'
                        new_lines.append(new_line)
                        modified_count += 1
                        continue
                    elif t1 == target_cg and t2 == 'W' or (t1 == 'W' and t2 == target_cg):
                        parts[3] = '0.387'
                        parts[4] = '1.36'
                        new_line = f'  {t1:<8s} {t2:<8s} {parts[2]:<5s} {parts[3]:<12s} {parts[4]:<12s}'
                        if ';' in line:
                            comment = line[line.find(';'):].strip()
                            new_line += '  ' + comment
                        new_line += '\n'
                        new_lines.append(new_line)
                        modified_count += 1
                        continue
            new_lines.append(line)
        with open(ff_path, 'w') as f:
            f.writelines(new_lines)
    ml_files = glob.glob(os.path.join(base_dir, '*_ML.itp'))
    for ml_file in ml_files:
        with open(ml_file, 'r') as f:
            lines = f.readlines()
        atom_map = {}
        new_ml_lines = []
        current_section = None
        file_modified = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                current_section = stripped.strip('[]').strip()
                new_ml_lines.append(line)
                continue
            if not stripped or stripped.startswith(';'):
                new_ml_lines.append(line)
                continue
            if current_section == 'atoms':
                parts = line.split()
                if len(parts) >= 2:
                    atom_id = parts[0]
                    atom_type = parts[1]
                    atom_map[atom_id] = atom_type
                new_ml_lines.append(line)
            elif current_section == 'constraints':
                parts = line.split()
                if len(parts) >= 4:
                    ai = parts[0]
                    aj = parts[1]
                    type_i = atom_map.get(ai)
                    type_j = atom_map.get(aj)
                    if type_i == target_cg and type_j == target_cg:
                        original_length = parts[3]
                        parts[3] = '0.29'
                        new_line = f'    {parts[0]:<4s} {parts[1]:<4s} {parts[2]:<4s} {parts[3]:<10s}'
                        if ';' in line:
                            comment = line[line.find(';'):].strip()
                            new_line += comment
                        new_line += '\n'
                        new_ml_lines.append(new_line)
                        file_modified = True
                    else:
                        new_ml_lines.append(line)
                else:
                    new_ml_lines.append(line)
            else:
                new_ml_lines.append(line)
        if file_modified:
            with open(ml_file, 'w') as f:
                f.writelines(new_ml_lines)
if __name__ == '__main__':
    main()
