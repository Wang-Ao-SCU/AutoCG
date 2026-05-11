import pandas as pd
import glob
import itertools
import os

def main():
    GLOBAL_FF = 'output/forcefield_new.itp'
    CG_MAP_CSV = 'output/global_smiles_to_cgtype.csv'
    DIH_DATA_CSV = 'dihedral_data/dihedral_data.csv'
    ML_ITP_PATTERN = 'output/*_ML.itp'
    try:
        cg_df = pd.read_csv(CG_MAP_CSV)
        cg_to_smiles = dict(zip(cg_df['CG_Type'].str.strip(), cg_df['SMILES'].str.strip()))
        smiles_to_cg = {}
        for c, s in cg_to_smiles.items():
            smiles_to_cg.setdefault(s, []).append(c)
    except Exception as e:
        return
    try:
        dih_df = pd.read_csv(DIH_DATA_CSV)
    except Exception as e:
        return
    dih_rules = {}
    unique_rules = []
    for idx, row in dih_df.iterrows():
        try:
            s1 = str(row['smiles_1']).strip()
            s2 = str(row['smiles_2']).strip()
            s3 = str(row['smiles_3']).strip()
            s4 = str(row['smiles_4']).strip()
            param_str = str(row['Param']).strip()
            parts = param_str.split()
            if not parts:
                continue
            func = parts[0]
            params = ' '.join(parts[1:])
            dih_rules[s1, s2, s3, s4] = (func, params)
            dih_rules[s4, s3, s2, s1] = (func, params)
            unique_rules.append({'s': [s1, s2, s3, s4], 'func': func, 'params': params})
        except Exception as e:
            pass
    if not os.path.exists(GLOBAL_FF):
        pass
    else:
        new_ff_entries = []
        seen_ff_types = set()
        for rule in unique_rules:
            s_list = rule['s']
            func = rule['func']
            params = rule['params']
            c1_opts = smiles_to_cg.get(s_list[0], [])
            c2_opts = smiles_to_cg.get(s_list[1], [])
            c3_opts = smiles_to_cg.get(s_list[2], [])
            c4_opts = smiles_to_cg.get(s_list[3], [])
            if not (c1_opts and c2_opts and c3_opts and c4_opts):
                continue
            for c1, c2, c3, c4 in itertools.product(c1_opts, c2_opts, c3_opts, c4_opts):
                combo = (c1, c2, c3, c4)
                rev_combo = (c4, c3, c2, c1)
                if combo in seen_ff_types or rev_combo in seen_ff_types:
                    continue
                seen_ff_types.add(combo)
                new_ff_entries.append(f'{c1}\t{c2}\t{c3}\t{c4}\t{func}\t{params}\n')
        if new_ff_entries:
            with open(GLOBAL_FF, 'a') as f:
                f.write('\n[ dihedraltypes ]\n')
                f.write('; i      j      k      l     func     params (Added by script)\n')
                for entry in new_ff_entries:
                    f.write(entry)
    itp_files = glob.glob(ML_ITP_PATTERN)
    for fname in itp_files:
        atoms = {}
        bonds = []
        with open(fname, 'r') as f:
            lines = f.readlines()
        mode = None
        for line in lines:
            sline = line.strip()
            if not sline or sline.startswith(';'):
                continue
            if sline.startswith('['):
                mode = sline.strip('[] ').strip()
                continue
            parts = sline.split()
            if mode == 'atoms':
                try:
                    nr = int(parts[0])
                    atype = parts[1]
                    atoms[nr] = atype
                except:
                    pass
            elif mode == 'bonds':
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    bonds.append((i, j))
                except:
                    pass
        adj = {}
        for i, j in bonds:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        found_dihedrals = []
        for j in adj:
            for k in adj[j]:
                if k <= j:
                    continue
                i_nodes = [x for x in adj[j] if x != k]
                l_nodes = [x for x in adj[k] if x != j]
                for i in i_nodes:
                    for l in l_nodes:
                        if i == l:
                            continue
                        ti, tj, tk, tl = (atoms.get(i), atoms.get(j), atoms.get(k), atoms.get(l))
                        if not all([ti, tj, tk, tl]):
                            continue
                        si = cg_to_smiles.get(ti)
                        sj = cg_to_smiles.get(tj)
                        sk = cg_to_smiles.get(tk)
                        sl = cg_to_smiles.get(tl)
                        if not all([si, sj, sk, sl]):
                            continue
                        rule = dih_rules.get((si, sj, sk, sl))
                        if rule:
                            func, _ = rule
                            found_dihedrals.append(f'{i}\t{j}\t{k}\t{l}\t{func}\n')
        if found_dihedrals:
            with open(fname, 'a') as f:
                f.write('\n[ dihedrals ]\n')
                f.write('; i      j      k      l     func\n')
                for entry in found_dihedrals:
                    f.write(entry)
if __name__ == '__main__':
    main()
