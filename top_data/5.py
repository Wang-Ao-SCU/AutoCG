#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatically replace bond/angle/dihedral parameters in the force-field template based on SMILES fragment matching,
and append the comment `; updated by script` to each modified line.
python update_ff.py
Output: forcefield_new.itp
"""
import re
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# -------------------- Utility functions --------------------
def smi2fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return fp, mol.GetNumAtoms()

def match_smi(target_smi: str, ref_smi: str):
    fp1, n1 = smi2fp(target_smi)
    fp2, n2 = smi2fp(ref_smi)
    return n1 == n2 and fp1 == fp2

# -------------------- Read parameter tables and mapping tables --------------------
def load_param_db():
    bond_df = pd.read_csv("top_data/bond_data.csv")
    angle_df = pd.read_csv("top_data/angle_data.csv")
    dihedral_df = pd.read_csv("top_data/dihedral_data.csv")

    bond_db = {}
    for _, r in bond_df.iterrows():
        key = (r.smiles_1.strip(), r.smiles_2.strip())
        bond_db[key] = (int(r.type), float(r.r), float(r.k))

    angle_db = {}
    for _, r in angle_df.iterrows():
        key = (r.smiles_1.strip(), r.smiles_2.strip(), r.smiles_3.strip())
        angle_db[key] = (int(r.type), float(r.r), float(r.k))

    dihedral_db = {}
    for _, r in dihedral_df.iterrows():
        key = (
            r.smiles_1.strip(),
            r.smiles_2.strip(),
            r.smiles_3.strip(),
            r.smiles_4.strip(),
        )
        dihedral_db[key] = (int(r.type), float(r.PHI0), float(r.CP), int(r.MULT))

    return bond_db, angle_db, dihedral_db


def load_mapping():
    bond_df = pd.read_csv("output/bonds.csv")
    angle_df = pd.read_csv("output/angles.csv")
    dihedral_df = pd.read_csv("output/dihedrals.csv")

    bond_map = {}
    for _, r in bond_df.iterrows():
        bond_map[(r.bead_type1, r.bead_type2)] = (r.smiles1, r.smiles2)

    angle_map = {}
    for _, r in angle_df.iterrows():
        angle_map[(r.bead_type1, r.bead_type2, r.bead_type3)] = (
            r.smiles1,
            r.smiles2,
            r.smiles3,
        )

    dihedral_map = {}
    for _, r in dihedral_df.iterrows():
        dihedral_map[
            (r.bead_type1, r.bead_type2, r.bead_type3, r.bead_type4)
        ] = (r.smiles1, r.smiles2, r.smiles3, r.smiles4)

    return bond_map, angle_map, dihedral_map


# -------------------- Core logic: replace and annotate --------------------
def update_itp(template="output/forcefield.itp", outfile="output/forcefield_new.itp"):
    bond_db, angle_db, _ = load_param_db()
    bond_map, angle_map, _ = load_mapping()

    txt = Path(template).read_text(encoding="utf8")

    # 3.1 Replace [ bondtypes ]
    def repl_bond(m):
        bt1, bt2, func, r, k = m.group(1, 2, 3, 4, 5)
        comment = m.group(6).strip()  # Original content after the semicolon
        key = (bt1, bt2)
        if key not in bond_map:
            warnings.warn(f"Bond mapping missing for {key}")
            return m.group(0)
        smi1, smi2 = bond_map[key]
        matched = None
        for (s1, s2), (t, rr, kk) in bond_db.items():
            if (match_smi(smi1, s1) and match_smi(smi2, s2)) or (
                match_smi(smi1, s2) and match_smi(smi2, s1)
            ):
                matched = (t, rr, kk)
                break
        if matched is None:
            warnings.warn(f"Bond param missing for SMILES {(smi1, smi2)}")
            return m.group(0)
        t, rr, kk = matched
        new_comment = f"; updated by script"
        if comment:
            new_comment = f"; {comment} {new_comment}"
        return f"{bt1:6} {bt2:6} {t:6} {rr:10.4f} {kk:10.1f} {new_comment}"

    txt = re.sub(
        r"^([A-Z0-9]{3})\s+([A-Z0-9]{3})\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)(.*)$",
        repl_bond,
        txt,
        flags=re.MULTILINE,
    )

    # 3.2 Replace [ angletypes ]
    def repl_angle(m):
        bt1, bt2, bt3, func, theta, k = m.group(1, 2, 3, 4, 5, 6)
        comment = m.group(7).strip()
        key = (bt1, bt2, bt3)
        if key not in angle_map:
            warnings.warn(f"Angle mapping missing for {key}")
            return m.group(0)
        smi1, smi2, smi3 = angle_map[key]
        matched = None
        for (s1, s2, s3), (t, rr, kk) in angle_db.items():
            if (
                match_smi(smi1, s1)
                and match_smi(smi2, s2)
                and match_smi(smi3, s3)
            ) or (
                match_smi(smi1, s3)
                and match_smi(smi2, s2)
                and match_smi(smi3, s1)
            ):
                matched = (t, rr, kk)
                break
        if matched is None:
            warnings.warn(f"Angle param missing for SMILES {(smi1, smi2, smi3)}")
            return m.group(0)
        t, rr, kk = matched
        new_comment = f"; updated by script"
        if comment:
            new_comment = f"; {comment} {new_comment}"
        return f"{bt1:6} {bt2:6} {bt3:6} {t:6} {rr:10.2f} {kk:10.1f} {new_comment}"

    txt = re.sub(
        r"^([A-Z0-9]{3})\s+([A-Z0-9]{3})\s+([A-Z0-9]{3})\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)(.*)$",
        repl_angle,
        txt,
        flags=re.MULTILINE,
    )

    # 3.3 If dihedral processing is needed, add a repl_dihedral block similar to the ones above. Omitted here.

    Path(outfile).write_text(txt, encoding="utf8")
    print(f"Updated force field written to {outfile}")


# -------------------- main --------------------
if __name__ == "__main__":
    update_itp()
