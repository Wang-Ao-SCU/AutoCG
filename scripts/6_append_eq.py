import os
import argparse
import pandas as pd

class FFEquilibriumUpdater:

    def __init__(self, ff_itp_path, model_type, bond_csv_path=None, angle_csv_path=None, dihedral_csv_path=None):
        if model_type not in ['ridge', 'gbr']:
            raise ValueError(f"Invalid model type: {model_type}, only 'ridge' or 'gbr' are supported")
        self.ff_itp_path = ff_itp_path
        self.model_type = model_type.lower()
        self.bond_data = self._safe_load_data(bond_csv_path, self._load_bond_data, 'bond parameters')
        self.angle_data = self._safe_load_data(angle_csv_path, self._load_angle_data, 'angle parameters')
        self.dihedral_data = self._safe_load_data(dihedral_csv_path, self._load_dihedral_data, 'dihedral parameters')
        self.param_cols = {'bond_r': f'{self.model_type}_r_pred', 'angle_theta0': f'{self.model_type}_theta0_pred', 'dihedral_phi0': f'{self.model_type}_phi0_pred'}

    def _safe_load_data(self, csv_path, loader_func, data_name):
        if not csv_path:
            return None
        try:
            return loader_func(csv_path)
        except Exception as e:
            return None

    def _load_bond_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'File not found: {csv_path}')
        df = pd.read_csv(csv_path)
        required_cols = ['bead_type1', 'bead_type2', f'{self.model_type}_r_pred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        df['type_pair'] = df.apply(lambda x: tuple(sorted([x['bead_type1'], x['bead_type2']])), axis=1)
        return dict(zip(df['type_pair'], df[f'{self.model_type}_r_pred']))

    def _load_angle_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'File not found: {csv_path}')
        df = pd.read_csv(csv_path)
        required_cols = ['bead_type1', 'bead_type2', 'bead_type3', f'{self.model_type}_theta0_pred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        df['type_triple'] = df.apply(lambda x: (x['bead_type1'], x['bead_type2'], x['bead_type3']), axis=1)
        return dict(zip(df['type_triple'], df[f'{self.model_type}_theta0_pred']))

    def _load_dihedral_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'File not found: {csv_path}')
        df = pd.read_csv(csv_path)
        required_cols = ['bead_type1', 'bead_type2', 'bead_type3', 'bead_type4', f'{self.model_type}_phi0_pred']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        df['type_quadruple'] = df.apply(lambda x: (x['bead_type1'], x['bead_type2'], x['bead_type3'], x['bead_type4']), axis=1)
        return dict(zip(df['type_quadruple'], df[f'{self.model_type}_phi0_pred']))

    def _process_itp_section(self, lines, start_marker, end_markers, update_func):
        processed_lines = []
        in_target_section = False
        for line in lines:
            if start_marker.strip() in line.strip():
                in_target_section = True
                processed_lines.append(line)
                continue
            if in_target_section and any((marker.strip() in line.strip() for marker in end_markers)):
                in_target_section = False
                processed_lines.append(line)
                continue
            if in_target_section:
                if line.strip().startswith(';') or not line.strip():
                    processed_lines.append(line)
                else:
                    processed_lines.append(update_func(line))
            else:
                processed_lines.append(line)
        return processed_lines

    def _update_bond_equilibrium(self, line):
        parts = line.strip().split()
        if len(parts) < 4:
            return line
        bead_type1, bead_type2 = (parts[0], parts[1])
        type_pair = tuple(sorted([bead_type1, bead_type2]))
        new_r = self.bond_data.get(type_pair, parts[3])
        try:
            new_r = f'{float(new_r):.3f}'
        except ValueError:
            new_r = parts[3]
        comment = ' '.join(parts[5:]) if len(parts) > 5 else ''
        updated_parts = [bead_type1.ljust(4), bead_type2.ljust(4), parts[2], new_r.ljust(6)]
        if len(parts) >= 5:
            updated_parts.append(parts[4].ljust(6))
        if comment:
            updated_parts.append(comment)
        return '  '.join(updated_parts) + '\n'

    def _update_angle_equilibrium(self, line):
        parts = line.strip().split()
        if len(parts) < 5:
            return line
        bead_type1, bead_type2, bead_type3 = (parts[0], parts[1], parts[2])
        type_triple = (bead_type1, bead_type2, bead_type3)
        new_theta0 = self.angle_data.get(type_triple, parts[4])
        try:
            new_theta0 = f'{float(new_theta0):.1f}'
        except ValueError:
            new_theta0 = parts[4]
        comment = ' '.join(parts[6:]) if len(parts) > 6 else ''
        updated_parts = [bead_type1.ljust(4), bead_type2.ljust(4), bead_type3.ljust(4), parts[3], new_theta0.ljust(6)]
        if len(parts) >= 6:
            updated_parts.append(parts[5].ljust(6))
        if comment:
            updated_parts.append(comment)
        return '  '.join(updated_parts) + '\n'

    def _update_dihedral_equilibrium(self, line):
        parts = line.strip().split()
        if len(parts) < 6:
            return line
        bead_type1, bead_type2, bead_type3, bead_type4 = (parts[0], parts[1], parts[2], parts[3])
        type_quadruple = (bead_type1, bead_type2, bead_type3, bead_type4)
        new_phi0 = self.dihedral_data.get(type_quadruple, parts[5])
        try:
            new_phi0 = f'{float(new_phi0):.1f}'
        except ValueError:
            new_phi0 = parts[5]
        comment = ' '.join(parts[7:]) if len(parts) > 7 else ''
        updated_parts = [bead_type1.ljust(4), bead_type2.ljust(4), bead_type3.ljust(4), bead_type4.ljust(4), parts[4], new_phi0.ljust(6)]
        if len(parts) >= 7:
            updated_parts.append(parts[6].ljust(6))
        if len(parts) >= 8:
            updated_parts.append(parts[7])
        if comment:
            updated_parts.append(comment)
        return '  '.join(updated_parts) + '\n'

    def update_equilibrium_parameters(self, output_path=None):
        if not output_path:
            dirname, filename = os.path.split(self.ff_itp_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(dirname, f'{name}_eq_updated{ext}')
        if not os.path.exists(self.ff_itp_path):
            raise FileNotFoundError(f'force-fieldITPFile not found: {self.ff_itp_path}')
        with open(self.ff_itp_path, 'r') as f:
            lines = f.readlines()
        if self.bond_data is not None:
            lines = self._process_itp_section(lines, start_marker='[ bondtypes ]', end_markers=['[ angletypes ]', '[ dihedraltypes ]', '[ pairs ]', '[ exclusions ]'], update_func=self._update_bond_equilibrium)
        if self.angle_data is not None:
            lines = self._process_itp_section(lines, start_marker='[ angletypes ]', end_markers=['[ bondtypes ]', '[ dihedraltypes ]', '[ pairs ]', '[ exclusions ]'], update_func=self._update_angle_equilibrium)
        if self.dihedral_data is not None:
            lines = self._process_itp_section(lines, start_marker='[ dihedraltypes ]', end_markers=['[ bondtypes ]', '[ angletypes ]', '[ pairs ]', '[ exclusions ]'], update_func=self._update_dihedral_equilibrium)
        with open(output_path, 'w') as f:
            f.writelines(lines)
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Update force-field equilibrium parameters (bond lengths, angles, and dihedrals) from machine-learning predictions')
    parser.add_argument('-ff', '--forcefield', required=True, help='Path to the global force-field ITP file to be updated(for example all_molecules_ff.itp)')
    parser.add_argument('-b', '--bond-csv', help='Path to the CSV file containing predicted bond parameters(for example all_bondtypes_with_predictions.csv, optional)')
    parser.add_argument('-a', '--angle-csv', help='Path to the CSV file containing predicted angle parameters(for example all_angletypes_with_predictions.csv, optional)')
    parser.add_argument('-d', '--dihedral-csv', help='dihedralPath to the CSV file containing predicted angle parameters(for example all_dihedraltypes_with_predictions.csv, optional)')
    parser.add_argument('-m', '--model', required=True, choices=['ridge', 'gbr'], help='机器学习modeltype, 用于选择predicted values(ridge 或 gbr)')
    parser.add_argument('-o', '--output', help='Output path for the updated ITP file (optional)')
    args = parser.parse_args()
    try:
        updater = FFEquilibriumUpdater(ff_itp_path=args.forcefield, bond_csv_path=args.bond_csv, angle_csv_path=args.angle_csv, dihedral_csv_path=args.dihedral_csv, model_type=args.model)
        updater.update_equilibrium_parameters(output_path=args.output)
    except Exception as e:
        exit(1)
if __name__ == '__main__':
    main()
