import pandas as pd
import argparse
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Add the `[ nonbond_params ]` section to the global ff.itp using `GBR_predictions.csv` (with automatic deduplication)')
    parser.add_argument('--csv', required=True, help='Path to the machine-learning prediction parameter file (for example, `GBR_predictions.csv`)')
    parser.add_argument('--input-itp', required=True, help='Path to the input global force-field file (for example, `all_molecules_ff.itp`)')
    parser.add_argument('--output-itp', required=True, help='Path to the output force-field file after adding `[ nonbond_params ]`')
    return parser.parse_args()

def extract_cg_number(cg_str):
    match = re.match('^CG(\\d+)$', cg_str.strip())
    if not match:
        raise ValueError(f'Invalid bead-type format: {cg_str}, expected format: `CG` followed by digits(for example `CG0` or `CG1`)')
    return int(match.group(1))

def process_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV file not found: {csv_path}')
    df = pd.read_csv(csv_path)
    required_cols = ['bead_type1', 'bead_type2', 'predicted_sigma', 'predicted_epsilon']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'CSV file is missing required columns: {', '.join(missing_cols)}')
    nonbond_dict = {}
    for _, row in df.iterrows():
        t1_str = str(row['bead_type1'])
        t2_str = str(row['bead_type2'])
        t1_num = extract_cg_number(t1_str)
        t2_num = extract_cg_number(t2_str)
        if t1_num <= t2_num:
            t_min_str = t1_str
            t_max_str = t2_str
        else:
            t_min_str = t2_str
            t_max_str = t1_str
        key = (t_min_str, t_max_str)
        if key not in nonbond_dict:
            nonbond_dict[key] = (row['predicted_sigma'], row['predicted_epsilon'])

    def sort_key(item):
        key = item[0]
        num1 = extract_cg_number(key[0])
        num2 = extract_cg_number(key[1])
        return (num1, num2)
    sorted_nonbond = sorted(nonbond_dict.items(), key=sort_key)
    return sorted_nonbond

def insert_nonbond_params(input_itp_path, output_itp_path, sorted_nonbond):
    if not os.path.exists(input_itp_path):
        raise FileNotFoundError(f'Input ITP file not found: {input_itp_path}')
    with open(input_itp_path, 'r', encoding='utf-8') as f:
        itp_lines = f.readlines()
    insert_index = None
    atomtypes_found = False
    for idx, line in enumerate(itp_lines):
        line_stripped = line.strip()
        if line_stripped == '[ atomtypes ]':
            atomtypes_found = True
        elif atomtypes_found and line_stripped == '[ bondtypes ]':
            insert_index = idx
            break
    if not atomtypes_found:
        raise ValueError('inputITPfiledoes not contain an `[ atomtypes ]` section, so insertion is not possible')
    if insert_index is None:
        raise ValueError('inputITPfiledoes not contain a `[ bondtypes ]` section, so the insertion point cannot be determined')
    nonbond_lines = ['\n[ nonbond_params ]\n', ';   i        j        func    sigma      epsilon  ; Nonbonded interaction parameters (Lennard-Jones 12-6)\n', ';----------|----------|-------|----------|----------\n']
    for (t_min_str, t_max_str), (sigma, epsilon) in sorted_nonbond:
        type1 = t_min_str
        type2 = t_max_str
        nonbond_line = f'  {type1:6s}  {type2:6s}    1    {sigma:10.6f}    {epsilon:10.6f}\n'
        nonbond_lines.append(nonbond_line)
    itp_lines = itp_lines[:insert_index] + nonbond_lines + itp_lines[insert_index:]
    with open(output_itp_path, 'w', encoding='utf-8') as f:
        f.writelines(itp_lines)

def main():
    args = parse_args()
    try:
        sorted_nonbond = process_csv(args.csv)
        insert_nonbond_params(args.input_itp, args.output_itp, sorted_nonbond)
    except Exception as e:
        exit(1)
if __name__ == '__main__':
    main()
