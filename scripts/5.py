import re
import warnings
import itertools
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def clean_smiles(smi: str) -> str:
    if not isinstance(smi, str):
        return ''
    return smi.strip().replace(' ', '').replace('\t', '')

def get_mol_safe(smi: str):
    cleaned = clean_smiles(smi)
    mol = Chem.MolFromSmiles(cleaned)
    if mol:
        return mol
    mol = Chem.MolFromSmiles(cleaned, sanitize=False)
    if mol:
        try:
            mol.UpdatePropertyCache(strict=False)
        except:
            pass
        return mol
    return None

def match_smi(target_smi: str, ref_smi: str) -> bool:
    s1 = clean_smiles(target_smi)
    s2 = clean_smiles(ref_smi)
    if s1 == s2:
        return True
    mol1 = get_mol_safe(s1)
    mol2 = get_mol_safe(s2)
    if mol1 is not None and mol2 is not None:
        try:
            can1 = Chem.MolToSmiles(mol1, canonical=True, isomericSmiles=False)
            can2 = Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=False)
            if can1 == can2:
                return True
        except:
            pass
        try:
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
            if mol1.GetNumAtoms() == mol2.GetNumAtoms() and fp1 == fp2:
                return True
        except:
            pass
    return False

def get_triplet_permutations(triplet: tuple) -> list:
    return list(set(itertools.permutations(triplet)))

def load_param_db():
    try:
        bond_df = pd.read_csv('top_data/bond_data.csv')
        angle_df = pd.read_csv('top_data/angle_data.csv')
    except FileNotFoundError as e:
        return ({}, {}, {})
    bond_db = {}
    for _, r in bond_df.iterrows():
        key = (clean_smiles(r.smiles_1), clean_smiles(r.smiles_2))
        bond_db[key] = (int(r.type), float(r.r), float(r.k))
        bond_db[key[1], key[0]] = (int(r.type), float(r.r), float(r.k))
    angle_db = {}
    for _, r in angle_df.iterrows():
        key = (clean_smiles(r.smiles_1), clean_smiles(r.smiles_2), clean_smiles(r.smiles_3))
        vals = (int(r.type), float(r.r), float(r.k))
        angle_db[key] = vals
        angle_db[key[2], key[1], key[0]] = vals
    dihedral_db = {}
    return (bond_db, angle_db, dihedral_db)

def load_mapping():
    try:
        bond_df = pd.read_csv('output/bonds.csv')
        angle_df = pd.read_csv('output/angles.csv')
    except FileNotFoundError as e:
        return ({}, {}, {})
    bond_map = {}
    for _, r in bond_df.iterrows():
        key = (str(r.bead_type1).strip(), str(r.bead_type2).strip())
        bond_map[key] = (clean_smiles(r.smiles1), clean_smiles(r.smiles2))
        bond_map[key[1], key[0]] = (clean_smiles(r.smiles2), clean_smiles(r.smiles1))
    angle_map = {}
    for _, r in angle_df.iterrows():
        key = (str(r.bead_type1).strip(), str(r.bead_type2).strip(), str(r.bead_type3).strip())
        smi_tuple = (clean_smiles(r.smiles1), clean_smiles(r.smiles2), clean_smiles(r.smiles3))
        angle_map[key] = smi_tuple
        rev_key = (key[2], key[1], key[0])
        rev_smi = (smi_tuple[2], smi_tuple[1], smi_tuple[0])
        angle_map[rev_key] = rev_smi
    return (bond_map, angle_map, {})

def update_itp(template='output/forcefield.itp', outfile='output/forcefield_new.itp'):
    bond_db, angle_db, _ = load_param_db()
    bond_map, angle_map, _ = load_mapping()
    if not Path(template).exists():
        return
    txt = Path(template).read_text(encoding='utf8')
    stats = {'bonds': 0, 'angles': 0}

    def repl_bond(m):
        bt1, bt2, func, r, k = m.group(1, 2, 3, 4, 5)
        comment = m.group(6).strip() if m.group(6) else ''
        key = (bt1, bt2)
        is_silicon = 'CG16' in key
        if key not in bond_map:
            return m.group(0)
        smi1, smi2 = bond_map[key]
        matched = None
        for (s1, s2), (t, rr, kk) in bond_db.items():
            if match_smi(smi1, s1) and match_smi(smi2, s2):
                matched = (t, rr, kk)
                break
        if matched is None:
            if is_silicon:
                pass
            return m.group(0)
        t, rr, kk = matched
        new_comment = f'; updated by script (SMILES match)'
        if comment:
            if 'updated by script' not in comment:
                new_comment = f'; {comment} {new_comment}'
            else:
                new_comment = f'; {comment}'
        stats['bonds'] += 1
        return f'{bt1:6} {bt2:6} {t:6} {rr:10.4f} {kk:10.1f} {new_comment}'
    txt = re.sub('^([A-Z0-9]{3,5})\\s+([A-Z0-9]{3,5})\\s+(\\d+)\\s+(\\d+(\\.\\d+)?)\\s+(\\d+(\\.\\d+)?)(.*)$', repl_bond, txt, flags=re.MULTILINE)

    def repl_angle(m):
        bt1, bt2, bt3, func, theta, k = m.group(1, 2, 3, 4, 5, 6)
        comment = m.group(7).strip() if m.group(7) else ''
        key = (bt1, bt2, bt3)
        is_silicon = 'CG16' in key
        if key not in angle_map:
            return m.group(0)
        smi1, smi2, smi3 = angle_map[key]
        matched = None
        for (s1, s2, s3), (t, rr, kk) in angle_db.items():
            if match_smi(smi1, s1) and match_smi(smi2, s2) and match_smi(smi3, s3):
                matched = (t, rr, kk)
                break
        if matched is None:
            if is_silicon:
                pass
            return m.group(0)
        t, rr, kk = matched
        new_comment = f'; updated by script'
        if comment:
            if 'updated by script' not in comment:
                new_comment = f'; {comment} {new_comment}'
            else:
                new_comment = f'; {comment}'
        stats['angles'] += 1
        return f'{bt1:6} {bt2:6} {bt3:6} {t:6} {rr:10.2f} {kk:10.1f} {new_comment}'
    txt = re.sub('^([A-Z0-9]{3,5})\\s+([A-Z0-9]{3,5})\\s+([A-Z0-9]{3,5})\\s+(\\d+)\\s+(\\d+(\\.\\d+)?)\\s+(\\d+(\\.\\d+)?)(.*)$', repl_angle, txt, flags=re.MULTILINE)
    Path(outfile).write_text(txt, encoding='utf8')
if __name__ == '__main__':
    update_itp()
