import os
import shutil

def process_itp_files(source_dir):
    target_dir = os.path.join(source_dir, 'nocharge')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith('.itp'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(source_path, target_path)
            with open(target_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            in_atoms_section = False
            modified_lines = []
            for line in lines:
                if line.strip().startswith('[atoms]'):
                    in_atoms_section = True
                    modified_lines.append(line)
                    continue
                if in_atoms_section and line.strip().startswith('['):
                    in_atoms_section = False
                    modified_lines.append(line)
                    continue
                if in_atoms_section:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        parts[6] = '0.000'
                        modified_lines.append('\t'.join(parts) + '\n')
                    else:
                        modified_lines.append(line)
                else:
                    modified_lines.append(line)
            with open(target_path, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)
if __name__ == '__main__':
    ITP_DIRECTORY = './output'
    process_itp_files(ITP_DIRECTORY)
