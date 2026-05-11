import pandas as pd
csv_path = './ML_ions/OCO_pre.csv'
df = pd.read_csv(csv_path)
itp_path = './output/forcefield_new.itp'
with open(itp_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
atomtypes_index = -1
for i, line in enumerate(lines):
    if '[ atomtypes ]' in line:
        atomtypes_index = i
        break
if atomtypes_index != -1:
    insert_pos = atomtypes_index + 2
    while insert_pos < len(lines) and (lines[insert_pos].strip() == '' or lines[insert_pos].strip().startswith(';')):
        insert_pos += 1
    lines.insert(insert_pos, 'CLA       35.4500    -1.000     A   0.4700   3.5000\n')
    lines.insert(insert_pos, 'SOD       22.9900    1.000     A   0.4700   3.5000\n')
nonbond_index = -1
for i, line in enumerate(lines):
    if '[ nonbond_params ]' in line:
        nonbond_index = i
        break
insert_pos = nonbond_index + 1
while insert_pos < len(lines) and (not lines[insert_pos].strip().startswith('[')):
    insert_pos += 1
for _, row in df.iterrows():
    bead_type = row['bead_type']
    sigma = row['predicted_sigma']
    epsilon = row['predicted_epsilon']
    lines.insert(insert_pos, f'  CLA     {bead_type}       1      {sigma:.10f}      {epsilon:.10f}\n')
    insert_pos += 1
lines.insert(insert_pos, '  CLA     CLA       1      0.354      1.1800000000\n')
nonbond_index = -1
for i, line in enumerate(lines):
    if '[ nonbond_params ]' in line:
        nonbond_index = i
        break
insert_pos = nonbond_index + 1
while insert_pos < len(lines) and (not lines[insert_pos].strip().startswith('[')):
    insert_pos += 1
for _, row in df.iterrows():
    bead_type = row['bead_type']
    sigma = row['predicted_sigma']
    epsilon = row['predicted_epsilon']
    lines.insert(insert_pos, f'  SOD     {bead_type}       1      {sigma:.10f}      {epsilon:.10f}\n')
    insert_pos += 1
lines.insert(insert_pos, '  SOD     SOD       1      0.354      1.1800000000\n')
lines.insert(insert_pos, '  W     SOD       1      0.385      11.460000000\n')
lines.insert(insert_pos, '  W     CLA       1      0.385      11.460000000\n')
lines.insert(insert_pos, '  SOD     CLA       1      0.354      1.1800000000\n')
output_path = './output/forcefield_new.itp'
with open(output_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
