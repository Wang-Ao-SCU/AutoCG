                     

import os
import numpy as np
import itertools
import requests
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdchem
from rdkit.Chem import rdMolDescriptors
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

import sys

sys.setrecursionlimit(200000)  

import re
import math
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import collections
import random
import matplotlib.pyplot as plt
from operator import itemgetter
import argparse
import tempfile
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import traceback, sys
sys.excepthook = traceback.print_exception



                                                                          
global_atomtypes = {}
global_bondtypes = {}
global_angletypes = {}
global_dihedraltypes = {}

ANGLE_FUNC_TYPE = 2

                                                                               
global_smiles_to_cgtype = {}
global_cgtype_counter = 0

                                                   
                                                     
delta_Gs = {
    0:{
        'standard':{
            'T': [-14.8,-15.2,-12.1,-9.8,-8.8,-7.2,-6.1,-4.9,-2.9,-3.1,0.3,2.3,3.6,4.5,6.4,6.7,7.8,12.0],
            'S': [-12.0,-11.8,-9.8,-7.7,-6.9,-5.2,-4.2,-3.6,-0.9,-1.8,2.1,3.6,5.3,6.3,8.4,9.2,9.9,14.2],
            'R': [-9.2,-9.1,-7.4,-5.1,-3.8,-2.0,-1.1,0.0,2.2,1.8,5.6,8.1,10.1,11.2,13.4,13.8,14.8,18.9]
        },
        'halogen':{
            'T': [2.7,5.4,5.2,7.6],
            'S': [4.3,8.0,7.2,9.4],
            'R': [8.7,13.9,12.7,14.3]
        }
    },
    4:{
        'standard':{
            'T': [-5.23,-5.77,-3.77,-0.35,0.44,2.18,2.90,4.08,6.03,5.41,8.92,10.64,11.84,12.56,14.35,14.74,15.74,19.10],
            'S': [-3.89,-4.15,-1.84,-0.08,0.78,2.39,3.31,4.20,6.66,5.84,9.78,11.56,12.84,13.99,16.20,16.56,17.32,20.91],
            'R': [-4.27,-4.01,-1.64,0.26,1.66,3.55,4.53,5.43,7.93,7.43,11.49,13.79,15.63,16.77,18.90,19.61,20.59,24.01]
        }
    },
    3:{
        'standard':{
            'T': [-7.73,-8.25,-6.19,-2.63,-1.85,0.12,0.80,2.12,4.12,3.66,7.13,8.93,10.11,10.85,12.71,12.92,13.99,17.66],
            'S': [-5.45,-5.57,-3.22,-1.59,-0.61,0.99,1.87,2.82,5.28,4.46,8.36,10.01,11.38,12.53,14.61,15.07,15.78,19.58],
            'R': [-4.68,-4.33,-2.10,-0.27,1.13,3.05,4.04,4.93,7.32,6.90,10.69,12.92,14.77,15.83,18.11,18.61,19.73,23.19]
        }
    },
    2:{
        'standard':{
            'T': [-10.22,-10.74,-8.61,-4.98,-4.09,-2.10,-1.33,0.09,1.97,1.62,5.16,6.99,8.26,9.04,10.98,11.23,16.19],
            'S': [-7.64,-7.69,-5.34,-3.62,-2.57,-0.92,-0.10,0.90,3.47,2.51,6.57,8.29,9.61,10.85,12.95,13.49,14.21,18.13],
            'R': [-6.43,-6.10,-3.79,-1.94,-0.50,1.42,2.40,3.30,5.83,5.27,9.22,11.67,13.39,14.43,16.75,17.37,18.41,22.03]
        }}}

m3_beads = {
    'standard': ['P6','P5','P4','P3','P2','P1','N6','N5','N4','N3','N2','N1','C6','C5','C4','C3','C2','C1'],
    'halogen': ['X4','X3','X2','X1']
    }

preset_beads = {
    'CC':'TC2',
    'CCC':'SC2',
    'CC(=O)O':'SN5',
    'CCC(=O)O':'N4',
    'COC=O':'N6',
    'COC(C)=O':'N4'
    }

                                                                
                                                                                                 

BOND_PARAMS = {
    'S-S': {0.36: 5000.0, 0.378: 5000.0, 0.321: 25000.0, 0.331: 5000.0, 0.3: 5000.0,
            0.37: 5000.0, 0.281: 25000.0, 0.314: 25000.0, 0.32: 7500.0, 0.38: 5000.0,
            0.33: 17000.0, 0.405: 5000.0, 0.395: 5000.0, 0.39: 5000.0, 0.385: 5000.0,
            0.35: 5000.0, 0.375: 3500.0, 0.376: 7000.0, 0.34: 7000.0},
    'T-T': {0.32: 25000.0, 0.261: 25000.0, 0.376: 25000.0, 0.25: 25000.0, 0.401: 25000.0,
            0.449: 10000.0, 0.251: 10000.0},
    'T-S': {0.364: 25000.0, 0.408: 25000.0, 0.272: 25000.0, 0.31: 7000.0, 0.3: 5000.0,
            0.253: 5000.0, 0.387: 25000.0, 0.34: 5000.0, 0.29: 5000.0, 0.32: 5000.0,
            0.33: 10000.0, 0.286: 20000.0, 0.371: 20000.0, 0.244: 20000.0,
            0.355: 5000.0, 0.36: 5000.0},
    'R-T': {0.389: 5000.0},
    'R-R': {0.38: 5000.0, 0.475: 3800.0, 0.47: 3800.0, 0.468: 3800.0, 0.49: 3800.0,
            0.46: 7000.0, 0.45: 7000.0, 0.455: 7000.0},
    'R-S': {0.385: 7000.0, 0.38: 7000.0, 0.405: 7000.0}
}

ANGLE_PARAMS = {
    'T-T-S': {180.0: 250.0, 138.0: 250.0, 71.0: 250.0, 122.0: 50.0},
    'T-S-S': {155.0: 100.0, 148.0: 100.0},
    'T-S-T': {135.0: 30.0},
    'S-S-S': {150.6: 100.0, 130.0: 25.0, 150.0: 100.0, 135.0: 15.0},
    'T-T-R': {160.0: 180.0},
    'R-R-R': {180.0: 35.0, 100.0: 10.0}
}

DIHEDRAL_PARAMS = {
    'S-S-S-T': {180.0: 100.0},
    'S-S-T-T': {180.0: 100.0},
    'S-T-T-S': {180.0: 75.0, 0.0: 50.0},
    'S-T-T-T': {180.0: 200.0, 0.0: 100.0},
    'T-T-T-T': {180.0: 200.0, 0.0: 100.0},
    'T-S-S-T': {180.0: 100.0},
    'R-T-T-T': {180.0: 50.0},
    'T-T-S-S': {180.0: 50.0},
    'S-T-S-S': {0.0: 50.0},
    'T-T-T-S': {180.0: 20.0},
    'T-R-R-T': {0.0: 1.8},
    'T-T-S-T': {-45.0: 200.0},
    'S-S-S-S': {180.0: 1.96, 0.0: 0.18}
}

def find_closest_key(param_dict, value):
    """Find the closest key in a dictionary to a given value."""
    if not param_dict:
        return None
    closest = min(param_dict.keys(), key=lambda k: abs(k - value))
    return closest

def get_bead_size_label(bead_type):
    """Get size label (T/S/R) from bead type string."""
    if bead_type.startswith('T'):
        return 'T'
    elif bead_type.startswith('S'):
        return 'S'
    elif bead_type.startswith('R'):
        return 'R'
    else:
                                                                   
        return 'R'

def get_bead_size_from_atoms(bead_atoms):
    """
    Determine bead size (T/S/R) based on number of heavy atoms in the bead.
    T (Tiny): 1-2 heavy atoms
    S (Small/Standard): 3-4 heavy atoms
    R (Ring/Large): 5+ heavy atoms
    """
    n_atoms = len(bead_atoms)
    if n_atoms <= 2:
        return 'T'
    elif n_atoms < 4:
        return 'S'
    else:
        return 'R'

def lookup_bond_force_constant(bond_length, bead_type1, bead_type2):
    """Look up bond force constant from MAD database based on bond length and bead types."""
    size1 = get_bead_size_from_atoms(bead_type1)
    size2 = get_bead_size_from_atoms(bead_type2)

    key = f"{size1}-{size2}"
    reverse_key = f"{size2}-{size1}"

    params = BOND_PARAMS.get(key) or BOND_PARAMS.get(reverse_key)

    if params:
        closest_length = find_closest_key(params, bond_length)
        return params[closest_length]
    else:
                          
        return 1250.0

def lookup_angle_force_constant(angle, bead_type1, bead_type2, bead_type3):
    """Look up angle force constant from MAD database based on angle value and bead types."""
    size1 = get_bead_size_from_atoms(bead_type1)
    size2 = get_bead_size_from_atoms(bead_type2)
    size3 = get_bead_size_from_atoms(bead_type3)

    key = f"{size1}-{size2}-{size3}"
    reverse_key = f"{size3}-{size2}-{size1}"

    params = ANGLE_PARAMS.get(key) or ANGLE_PARAMS.get(reverse_key)

    if params:
        closest_angle = find_closest_key(params, angle)
        return params[closest_angle]
    else:
                          
        return 25.0

def lookup_dihedral_force_constant(dihedral, bead_type1, bead_type2, bead_type3, bead_type4):
    """Look up dihedral force constant from MAD database based on dihedral value and bead types."""
    size1 = get_bead_size_from_atoms(bead_type1)
    size2 = get_bead_size_from_atoms(bead_type2)
    size3 = get_bead_size_from_atoms(bead_type3)
    size4 = get_bead_size_from_atoms(bead_type4)

    key = f"{size1}-{size2}-{size3}-{size4}"
    reverse_key = f"{size4}-{size3}-{size2}-{size1}"

    params = DIHEDRAL_PARAMS.get(key) or DIHEDRAL_PARAMS.get(reverse_key)

    if params:
        closest_dihedral = find_closest_key(params, dihedral)
        return params[closest_dihedral]
    else:
                          
        return 10


def _parse_mol2_charge(parts):
    """Best-effort parse for MOL2 atom charge field."""
    if len(parts) > 8:
        try:
            return float(parts[8])
        except ValueError:
            pass
    for token in reversed(parts):
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0

def normalize_mol2_file(mol2_file):
    """
    Normalize MOL2 section tags and ATOM records to improve parser robustness.
    Returns a temporary normalized MOL2 filepath.
    """
    with open(mol2_file, 'r') as f:
        lines = f.readlines()

    normalized_lines = []
    in_atom_section = False

    for raw in lines:
        line = raw.rstrip('\n').rstrip('\r')
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith('@<TRIPOS>'):
            normalized_lines.append(upper + '\n')
            in_atom_section = (upper == '@<TRIPOS>ATOM')
            continue

        if in_atom_section and stripped:
            parts = stripped.split()
            if len(parts) >= 6:
                try:
                    atom_id = int(parts[0])
                    atom_name = parts[1]
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atom_type = parts[5]
                    subst_id = int(parts[6]) if len(parts) > 6 and re.fullmatch(r'-?\d+', parts[6]) else 1
                    subst_name = parts[7] if len(parts) > 7 else 'MOL'
                    charge = _parse_mol2_charge(parts)
                    normalized_lines.append(
                        f"{atom_id:7d} {atom_name:<8s} {x:10.4f} {y:10.4f} {z:10.4f} "
                        f"{atom_type:<8s} {subst_id:4d} {subst_name:<8s} {charge:10.4f}\n"
                    )
                    continue
                except (ValueError, IndexError):
                    pass

        normalized_lines.append(line + '\n')

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.mol2', delete=False)
    try:
        tmp.writelines(normalized_lines)
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()

def _coord_from_mol(mol, atom_idx):
    """Fallback coordinate from RDKit conformer."""
    try:
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(atom_idx)
        return np.array([float(pos.x), float(pos.y), float(pos.z)])
    except Exception:
        return np.array([0.0, 0.0, 0.0])

def read_mol2_file(mol2_file):
    """
    Read mol2 file and extract molecule information, coordinates, and charges.
    Returns RDKit molecule object, atom charges dictionary, and coordinates.
    """
    print("ok")

                                                             
    parse_file = mol2_file
    temp_normalized_file = None
    mol = Chem.MolFromMol2File(parse_file, removeHs=False)
    if mol is None:
        temp_normalized_file = normalize_mol2_file(mol2_file)
        parse_file = temp_normalized_file
        print("done")
        mol = Chem.MolFromMol2File(parse_file, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read mol2 file (even after normalization): {mol2_file}")

                                                 
    atom_charges = {}
    atom_coords = {}

    with open(parse_file, 'r') as f:
        lines = f.readlines()
        in_atom_section = False

        for line in lines:
            line_upper = line.strip().upper()
            if line_upper.startswith('@<TRIPOS>ATOM'):
                in_atom_section = True
                continue
            elif line_upper.startswith('@<TRIPOS>') and in_atom_section:
                in_atom_section = False
                continue

            if in_atom_section:
                parts = line.split()
                if len(parts) >= 6:
                    atom_id = int(parts[0]) - 1                        
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    charge = _parse_mol2_charge(parts)
                    atom_charges[atom_id] = charge
                    atom_coords[atom_id] = np.array([x, y, z])

                                           
    mol_no_h = Chem.RemoveHs(mol)

                                                                   
    heavy_atom_map = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:                
            heavy_atom_map.append(atom.GetIdx())

                                                    
    heavy_atom_charges = {i: atom_charges.get(orig_idx, 0.0) for i, orig_idx in enumerate(heavy_atom_map)}
    heavy_atom_coords = {i: atom_coords.get(orig_idx, _coord_from_mol(mol, orig_idx))
                         for i, orig_idx in enumerate(heavy_atom_map)}

    if temp_normalized_file and os.path.exists(temp_normalized_file):
        try:
            os.remove(temp_normalized_file)
        except OSError:
            pass

    return mol_no_h, heavy_atom_charges, heavy_atom_coords, mol, heavy_atom_map, atom_charges

def calculate_bead_charges(beads, atom_charges, mol_with_h=None, heavy_atom_map=None, all_atom_charges=None):
    """
    Calculate charge for each bead by summing charges of constituent atoms.
    Now includes charges from hydrogen atoms bonded to heavy atoms in each bead.

    Parameters:
    - beads: List of beads (each bead contains heavy atom indices in mol_no_h)
    - atom_charges: Dict of charges for heavy atoms only (indexed by mol_no_h indices) - DEPRECATED, use all_atom_charges
    - mol_with_h: RDKit molecule with hydrogens (required for H charge inclusion)
    - heavy_atom_map: Mapping from heavy atom indices (mol_no_h) to original indices (mol_with_h) (required)
    - all_atom_charges: Dict of ALL atom charges (indexed by mol_with_h indices, includes H) (required)
    """
    bead_charges = []
    for bead in beads:
        total_charge = 0.0

        for heavy_idx in bead:
                                              
            orig_idx = heavy_atom_map[heavy_idx]

                                          
            heavy_charge = all_atom_charges.get(orig_idx, 0.0)
            total_charge += heavy_charge

                                                                              
            if mol_with_h is not None:
                atom = mol_with_h.GetAtomWithIdx(orig_idx)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:            
                        h_idx = neighbor.GetIdx()
                        h_charge = all_atom_charges.get(h_idx, 0.0)
                        total_charge += h_charge

        bead_charges.append(total_charge)

    return bead_charges

def write_map_file(map_filename, beads, mol_name, mol_with_h, heavy_atom_map):
    """
    Write a mapping file that links CG bead indices to original atom indices.
    Includes both heavy atoms and hydrogen atoms bonded to them.
    Format: bead_index atom_index1 atom_index2 ... (space-separated)

    Parameters:
    - beads: List of beads, where each bead contains heavy atom indices (0-indexed in mol_no_h)
    - mol_with_h: RDKit molecule with hydrogens
    - heavy_atom_map: Mapping from heavy atom indices to original indices in mol_with_h
    """
    with open(map_filename, 'w') as f:
        f.write(f"# Coarse-grained mapping for {mol_name}\n")
        f.write(f"# Format: bead_index atom_index1 atom_index2 ... (includes heavy atoms and bonded hydrogens)\n")
        f.write(f"# Total beads: {len(beads)}\n\n")

        for bead_idx, bead_atoms in enumerate(beads):
                                                                              
            all_atom_indices = []

                                             
            for heavy_idx in bead_atoms:
                                                           
                orig_idx = heavy_atom_map[heavy_idx]
                all_atom_indices.append(orig_idx)

                                                               
                atom = mol_with_h.GetAtomWithIdx(orig_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                                                   
                    if neighbor.GetAtomicNum() == 1:
                        all_atom_indices.append(neighbor_idx)

                                        
            atom_list = ' '.join(str(idx) for idx in sorted(all_atom_indices))
            f.write(f"{bead_idx} {atom_list}\n")

    print("runing")

def read_DG_data(DGfile):
                                                    
    DG_data = {}
    if os.path.exists(DGfile):
        with open(DGfile) as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) >= 2:
                    smi, DG = parts[0], parts[1]
                    src = parts[2] if len(parts) > 2 else 'unknown'
                    DG_data[smi] = {'DG':float(DG),'src':src}

    return DG_data

def resolve_dg_file(script_path):
    """
    Locate the fragment free-energy table.  Some distributions keep it under
    ML1/, while older code expected it next to AutoCG.py.
    """
    candidates = [
        os.path.join(script_path, 'fragments-exp.txt'),
        os.path.join(script_path, 'ML1', 'fragments-exp.txt'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Cannot find fragments-exp.txt. Checked: " + ", ".join(candidates)
    )

def require_output_file(path, description):
    """Raise a clear error if an expected output file is missing or empty."""
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {description}: {path}")
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"Empty {description}: {path}")

def required_molecule_outputs(mol_name, output_dir):
    m3_dir = os.path.join(output_dir, "M3")
    return [
        (os.path.join(output_dir, f"{mol_name}.gro"), "CG GRO"),
        (os.path.join(output_dir, f"{mol_name}.pdb"), "CG PDB"),
        (os.path.join(output_dir, f"{mol_name}.itp"), "CG ITP"),
        (os.path.join(output_dir, f"{mol_name}_ML.itp"), "ML ITP"),
        (os.path.join(output_dir, f"{mol_name}.top"), "CG topology"),
        (os.path.join(output_dir, f"{mol_name}.map"), "CG map"),
        (os.path.join(m3_dir, f"{mol_name}_m3.gro"), "M3 GRO"),
        (os.path.join(m3_dir, f"{mol_name}_m3.pdb"), "M3 PDB"),
        (os.path.join(m3_dir, f"{mol_name}_m3.itp"), "M3 ITP"),
        (os.path.join(m3_dir, f"{mol_name}_m3.top"), "M3 topology"),
    ]

def validate_molecule_outputs(mol_name, output_dir):
    for path, description in required_molecule_outputs(mol_name, output_dir):
        require_output_file(path, description)

def include_weights(A,w):
                                                  
    A_weighted = np.copy(A)
    for i,weight in enumerate(w):
        A_weighted[i,i] = weight

    return A_weighted

def get_weights(groups,w_init,path_matrix):
                                                               
    w = []
    for node in groups:
        avgmass = get_avgmass(node,w_init)
        wi = avgmass * (get_size(node,path_matrix))
        w.append(wi)

    return w

def Scale_Scores(A):

                                                                         
    for n in range(1,10):
        vals,vecs = np.linalg.eig(A)
        maxval = np.argmax(vals)
        scores = vecs[:,maxval]
                                                                                                                       
        if all(scores<0) or all(scores>0):
            scores=np.absolute(scores)
            if np.absolute(math.log10(scores[np.argmin(scores)]))-np.absolute(math.log10(scores[np.argmax(scores)])) <10:
                                                                     
                break

                                                                                  
        np.fill_diagonal(A, A.diagonal() / 2)

    return(scores)

def rank_nodes(A):

    scores = Scale_Scores(A)
    if args.v:print("",
                    "    Bead Centrality Scores: ",
                    "",
                    "    ",scores,
                    "")

                                                         
    min_score = np.amin(scores)
    if min_score == 0 or np.isnan(min_score) or np.isinf(min_score):
                                                     
        scores = scores + 1e-10
        min_score = np.amin(scores)

    scores = scores/min_score
    ranked = np.argsort(scores)

    ties = []
    sublist = []

                                                         
    score_prev = scores[ranked[0]]
    for i in ranked:
        score_i = scores[i]
        if np.isclose(score_i,score_prev):
            sublist.append(i)
        else:
            ties.append(sublist)
            sublist = [i]
        score_prev = score_i
    ties.append(sublist)

    if args.v:print("upd")
    if args.v:print("warn")
    if args.v:print("oops")
    if args.v:print("skip")
    if args.v:print("fine")
    return scores,ties

def lone_atom(ties,A,A_init,scores,ring_beads,matched_maps,comp,exclusion_list):
                                                                
    groups = []
    temp_exclusions = []
    n = 0
    for rank in ties:
        for node in rank:
            if len(comp[node]) == 1:
                test_group = [comp[node][0]]
                temp_exclusions.append(test_group[0])

                                              
                connects = A[node]
                                                                                               
                bonded = [i for i in np.nonzero(connects)[0] if not any(i in m for m in matched_maps)]
                
                                                
                if len(bonded) == 0:
                    continue

                                              
                bonded_scores = np.asarray([scores[bonded[k]] for k in range(len(bonded))])
                
                                                              
                                                                                  
                sorted_indices = np.argsort(bonded_scores)

                                                                                   
                valid_sorted_indices = []
                for idx in sorted_indices:
                    nbor_node = bonded[idx]
                                                                          
                    if not any(x in exclusion_list for x in comp[nbor_node]):
                        valid_sorted_indices.append(idx)
                
                                                              
                
                                                                                                      
                                                                             
                if len(valid_sorted_indices) == 0:
                                                                                          
                                                 
                    best_idx = sorted_indices[0]
                    target_nbor_node = bonded[best_idx]
                    
                                                                                                           
                                                                                                                     
                    force_merged_group = [comp[node][0]] + comp[target_nbor_node]
                    groups.append(force_merged_group)
                    
                                                               
                    continue

                                                                                                   
                bonded_sorted = valid_sorted_indices 
                
                                       

                                                                                    
                aa_connects = A_init[comp[node][0]]
                aa_bonded = [i for i in np.nonzero(aa_connects)[0]]

                                                                                   
                stolen_from = []
                
                                                                       
                best_valid_idx = bonded_sorted[0]
                score_prev = bonded_scores[best_valid_idx]
                
                for idx in bonded_sorted:
                                                    
                    nbor_node = bonded[idx]
                    scorej = scores[nbor_node]
                    
                    if np.isclose(scorej,score_prev):
                        stolen_from.append(nbor_node)
                                                                                    
                        if len(comp[nbor_node]) == 2 and len(np.nonzero(A[nbor_node])[0]) == 1:
                            test_group.extend(comp[nbor_node])
                        elif any(np.size(np.intersect1d(comp[nbor_node],ring)) != 0 for ring in ring_beads):
                            test_group.extend(comp[nbor_node])
                        else:
                            for a in aa_bonded:
                                if a in comp[nbor_node]:
                                    test_group.append(a)
                groups.append(test_group)
                n = len(groups) - 1
                
                                                  
                for k in stolen_from:
                    test_group_k = comp[k][:]
                    for b in comp[k]:
                        if b in groups[n]:
                            test_group_k.remove(b)
                    if test_group_k != []:
                        groups.append(test_group_k)

    exclusion_list.extend(temp_exclusions)
    temp_groups = groups[:]

    for a in range(3):
        new_nodes = []
        for group in temp_groups:                                                                    
            for i in range(len(new_nodes)):
                if np.size(np.intersect1d(group,new_nodes[i])) != 0:
                    new_nodes[i] = [int(p) for p in np.unique(np.concatenate((group,new_nodes[i]),axis=None)).tolist()]
                    break
            else:
                new_nodes.append([int(p) for p in group])
        temp_groups = new_nodes[:]

                                 
    for bead in comp:
        if not any(any(atom in group for group in groups) for atom in bead):
            new_nodes.append(bead)

                        
    new_ring_beads = []
    for ring in ring_beads:
                                
        for i,group in enumerate(new_nodes):
            if any(atom in ring for atom in group) and i not in new_ring_beads:
                new_ring_beads.append(i)

    return new_nodes,ring_beads,exclusion_list


def spectral_grouping(ties,A,scores,ring_beads,comp,path_matrix,max_size,matched_maps):
                                                                        
    groups = []

            
    if args.v:
        print("ok")
        print("done")

                                                          
    for rank in ties:
        new_groups = []
        for node in rank:
                                                                
            if any(node in a for a in groups) or any(node in a for a in ring_beads) or any(node in m for m in matched_maps):
                continue
                                                                                    
            test_group = [node]
            connects = A[node]
            bonded = [i for i in np.nonzero(connects)[0] if not (any(i in a for a in groups) or any(i in a for a in matched_maps))]
            bonded_scores = np.asarray([scores[bonded[k]] for k in range(len(bonded))])
            bonded_sorted = np.argsort(bonded_scores)
                                                   
            for j in bonded_sorted:
                                           
                if test_group == [node]:
                    if any(bonded[j] in a for a in ring_beads):
                        continue
                    if get_size(comp[node] + comp[bonded[j]],path_matrix) <= max_size:                                      
                        scorej = scores[bonded[j]]
                        test_group.append(bonded[j])
                    else:
                        break
                elif np.isclose(scorej,scores[bonded[j]]):                                          
                    test_group.append(bonded[j])
                else:
                    break
            new_groups.append(test_group)
        new_nodes = []
        for group in new_groups:                                                                    
            for i in range(len(new_nodes)):
                if np.size(np.intersect1d(group,new_nodes[i])) != 0:
                    new_nodes[i] = np.unique(np.concatenate((group,new_nodes[i]),axis=None)).tolist()
                    break
            else:
                new_nodes.append(group)

                                                                   
        for k in new_nodes[:]:
            compk = []
            for atom in k:
                compk.extend(comp[atom])

            if get_size(compk,path_matrix) > max_size:
                new_nodes.remove(k)
                for x in k:
                    new_nodes.append([x])
        groups = groups + new_nodes
    groups,ring_beads,matched_maps = process_rings(ring_beads,matched_maps,groups)                              

    return groups,ring_beads,matched_maps

def process_rings(ring_beads,matched_maps,groups):

                                                             
    for bead in ring_beads:
        if not any(any(a in group for a in bead) for group in groups):
            groups.append(bead)

    for match in matched_maps:
        groups.append(match)
                                                                 
    for i in range(len(groups)):
        for bead in ring_beads:
            if np.size(np.intersect1d(bead,groups[i])) != 0:
                groups[i] = np.unique(np.concatenate((groups[i],bead),axis=None)).tolist()

                                                                                 
    new_groups = []
    for l in range(len(groups)):
        if any(any(atom in bead for bead in new_groups) for atom in groups[l]):
            continue
        new_group = groups[l][:]
        for m in range(len(groups)):
            if np.size(np.intersect1d(new_group,groups[m])) != 0:
                new_group = np.unique(np.concatenate((new_group,groups[m]),axis=None)).tolist()
        new_groups.append(new_group)

    groups = new_groups

    new_ring_beads = []
    new_matched_maps = []

                                                      
    for k,group in enumerate(groups):
        for j in range(len(ring_beads)):
            if any(a in ring_beads[j] for a in group):
                new_ring_beads.append([k])
                break
        for p in range(len(matched_maps)):
            if any(a in matched_maps[p] for a in group):
                new_matched_maps.append([k])
                break

    return groups,new_ring_beads,new_matched_maps

def new_connectivity(groups,oldA):
                                  
    newA = np.zeros((len(groups),len(groups)),dtype=int)
    for i,gi in enumerate(groups):
        for j,gj in enumerate(groups[i+1:]):
            for k in gi:
                for l in gj:
                    if oldA[k,l] == 1:
                        newA[i,i+j+1] = 1
                        newA[i+j+1,i] = 1
                if newA[i,i+j+1] == 1:
                    break

    return newA

def iteration(results,itr,A_init,w_init,ring_beads,path_matrix,matched_maps):
    results_dict = dict.fromkeys(['A','comp'])

                                       
    if itr == 0:
        oldA = np.copy(A_init)
        comp = [[i] for i in range(len(w_init))]
        w = w_init[:]
    else:
        oldA = results[itr-1]['A']
        comp = results[itr-1]['comp']
        w = get_weights(comp,w_init,path_matrix)
    A_weighted = include_weights(oldA,w)

    if args.v: print("runing")
    if args.v: print("upd")

                            
    scores,ties = rank_nodes(A_weighted)
    groups,ring_beads,matched_maps = spectral_grouping(ties,oldA,scores,ring_beads,comp,path_matrix,3,matched_maps)
    results_dict['A'] = new_connectivity(groups,oldA)

                                              
    if itr == 0:
        results_dict['comp'] = groups[:]

    else:
        comp = []
        for gj in groups:
            bead_comp = list(itertools.chain.from_iterable([results[itr-1]['comp'][x] for x in gj]))
            comp.append(bead_comp)

        results_dict['comp'] = comp[:]


    return results_dict,ring_beads,matched_maps

def group_rings(A, ring_atoms, matched_maps, moli):
                                             
    new_groups = []
                                                                    
    edge_frags = collections.OrderedDict()
    edge_frags["[R1][R1][R1][R1][R1][R1]"] =  [[0,1],[2,3],[4,5]]
    edge_frags["[R1][R1][R1][R1][R1]"] = [[0,1,2],[2,3]]
    edge_frags["[R1][R1][R1][R1]"] = [[0,1],[2,3]]
    edge_frags["[R1][R1][R1]"] =  [[0,1,2]]
    edge_frags["[R1][R1]"] = [[0,1]]
    edge_frags["[R2][R1][R2]"] = [[0,1,2]]
    
                      
    for substruct in edge_frags:
        matches = moli.GetSubstructMatches(Chem.MolFromSmarts(substruct))
        for match in matches:
            all_shared = False
            for system in ring_atoms:
                if all(m in system for m in match):
                    all_shared = True
                    break
            if not all_shared:
                continue
            if substruct == "[R2][R1][R2]":
                overlap = False
                for matchj in matches:
                    if match != matchj and list(set(match).intersection(matchj)):
                        overlap = True
                        break
                if overlap:
                    continue
            for bead in edge_frags[substruct]:
                test_bead = [match[x] for x in bead]
                if not any(any(y in ngroup for ngroup in new_groups) for y in test_bead):
                    new_groups.append(test_bead)
    
    if args.v:
        print("warn")
    
                                  
    unmapped = []
    for ring in ring_atoms:
        for a in ring:
            if not any(a in group for group in new_groups):
                unmapped.append(a)
    

    if unmapped:
        if args.v:
            print("oops")
            print("skip")
        

        unm_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli, unmapped, canonical=False).upper()
        unm_mol = Chem.MolFromSmiles(unm_smi)
        if unm_mol is None:
            print("fine")
            return ring_beads, new_groups, A
        unmapped_frags = Chem.GetMolFrags(unm_mol)
        if args.v:
            print("ok")
        

        fragment_queue = []
        for frag in unmapped_frags:
            if args.v:
                print("done")

            indices = [unmapped[k] for k in frag]
 
            frag_smi = Chem.rdmolfiles.MolFragmentToSmiles(moli, unmapped).split(".")[0].upper()
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                print("runing")
                continue
            A_frag = np.asarray(Chem.GetAdjacencyMatrix(frag_mol), dtype='f')
            assign_atom_maps(frag_mol) 
            

            core_map = re.findall(r"\:([^\]]*)\]", frag_smi)
            if len(core_map) != len(indices):
                core_map.insert(0, "0") 
            core_map = [int(x) for x in core_map]
            
 
            fragment_queue.append({
                "frag_indices": indices,
                "frag_mol": frag_mol,
                "A_frag": A_frag,
                "core_map": core_map
            })
        

        while fragment_queue:
          
            frag_data = fragment_queue.pop(0)  
            indices = frag_data["frag_indices"]
            frag_mol = frag_data["frag_mol"]
            A_frag = frag_data["A_frag"]
            core_map = frag_data["core_map"]
            

            frag_ring_atoms = get_ring_atoms(frag_mol)
            new_beads = []
            
       
            if frag_ring_atoms:
         
                for ring in frag_ring_atoms:
                    
                    ring_matched = False
                    for substruct, bead_map in edge_frags.items():
                        substruct_mol = Chem.MolFromSmarts(substruct)
                        if substruct_mol is None:
                            continue
                        ring_matches = frag_mol.GetSubstructMatches(substruct_mol)
                        for match in ring_matches:
                            if all(m in ring for m in match):
                                for bead in bead_map:
                                    test_bead = [match[x] for x in bead]
                                    new_beads.append(test_bead)
                                ring_matched = True
                                break
                        if ring_matched:
                            break
                  
                    if not ring_matched and not any(any(a in b for a in ring) for b in new_beads):
                        new_beads.append(ring)
            else:
                new_beads = []
            
          
            if sum([len(b) for b in new_beads]) < frag_mol.GetNumAtoms():
                path_frag = floyd_warshall(csgraph=A_frag, directed=False)
                w_frag = [1.0 for _ in frag_mol.GetAtoms()]
                A_fragw = include_weights(A_frag, w_frag)
                scores, ties = rank_nodes(A_fragw)
                comp = [[i] for i in range(frag_mol.GetNumAtoms())]
             
                spectral_beads = spectral_grouping(ties, A_frag, scores, new_beads, comp, path_frag, 2, matched_maps)[0]
                new_beads.extend(spectral_beads)
            
           
            for bead in new_beads:
                
                match_flag = False
                for m in matched_maps:
                    if sorted(bead) == sorted(m):
                        match_flag = True
                        break
                if not match_flag:
                 
                    mapped_bead = [core_map[x] for x in bead]
                    new_groups.append(mapped_bead)
    

    ring_beads = new_groups[:]
                        
    new_groups += matched_maps
    for i in range(A.shape[0]):
        if not any(i in a for a in new_groups):
            new_groups.append([i])
    
    return ring_beads, new_groups, A

def postprocessing(results,ring_atoms,n_iter,A_init,w_init,path_matrix,matched_maps):
                                                     
    last_iter = results[n_iter -1]
    exclusion_list = []
    postprocess = 1
    while postprocess:
        min_size = 1000
        avg_size = 0
        count = 0.0
        for i,bead in enumerate(last_iter['comp']):
            size = len(bead)
            if size < min_size:
                min_size = size
            if i not in ring_atoms:
                avg_size += size
                count += 1.0
        avg_size = avg_size / count

        if min_size == 1:
            postprocess = 1
        else:
            postprocess = 0

        if postprocess:
                                                                      
            results_dict,ring_atoms,exclusion_list= path_contraction(last_iter,postprocess,A_init,w_init,ring_atoms,matched_maps,path_matrix,exclusion_list)
        else:
            results_dict = last_iter.copy()
        last_iter = results_dict.copy()

    return results_dict


def path_contraction(last_iter,postprocess,A_init,w_init,ring_beads,matched_maps,path_matrix,exclusion_list):
                               
    results_dict = dict.fromkeys(['A','comp'])

    oldA = last_iter['A']
    comp = last_iter['comp']
    w = get_weights(comp,w_init,path_matrix)

    A_weighted = include_weights(oldA,w)

    scores,ties = rank_nodes(A_weighted)
    groups,ring_beads,exclusion_list = lone_atom(ties,oldA,A_init,scores,ring_beads,matched_maps,comp,exclusion_list)
    results_dict['A'] = new_connectivity(groups,A_init)

    results_dict['comp'] = groups[:]
    return results_dict,ring_beads,exclusion_list


def get_size(comp,path_matrix):

                                             
    longpath = 0
    for i in comp:
        for j in comp:
            path = path_matrix[i,j]
            if path > longpath:
                    longpath = path

    return longpath

def get_avgmass(comp,masses):
                                               

    avgmass = sum([masses[i] for i in comp])/len(comp)
    return avgmass

def get_paths(A_atom,mol):

                                                  
    dist_matrix,preds = floyd_warshall(csgraph=A_atom,directed=False,return_predecessors=True)
    n_atoms = len(mol.GetAtoms())

                                                      
    row_weights = []
    for at in mol.GetAtoms():
        if at.GetAtomicNum <= 10:
            row_weights.append(1)
        else:
            row_weights.append(2)

    path_matrix = np.zeros((dist_matrix.shape()))

                                                     
    for i in range(n_atoms-1):
        for j in range(i,n_atoms):
            min_path = 0
            node = j
            while node != i:
                min_path += row_weights[node]
                node = preds[i,node]
            min_path += row_weights[i]
            path_matrix[i,j] = min_path

    return path_matrix


def assign_atom_maps(mol_dict):

                                                                                          
    for atom in mol_dict.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol_dict

def mapping(mol,ring_atoms,matched_maps,n_iter,mol_dict):
                               
                                     
    A_atom = np.asarray(Chem.GetAdjacencyMatrix(mol),dtype='f')
    path_matrix = floyd_warshall(csgraph=A_atom,directed=False)
    w_init = [0.5*atom.GetMass() for atom in mol.GetAtoms()]                                                                                                                                 
                                              

    assign_atom_maps(mol_dict)                                                                          
    ring_beads,comp,A_init = group_rings(A_atom,ring_atoms,matched_maps,mol_dict)
                                      

                                    
    results = []
    for itr in range(n_iter):

        if args.v:print("upd")
        if args.v:print("warn")
        if args.v:print("oops")

                                                                                                             
                             

        results_dict,ring_beads,matched_maps = iteration(results,itr,A_init,w_init,ring_beads,path_matrix,matched_maps)
        results.append(results_dict)
        if args.v:print("skip")
        if args.v:print("fine")

    if args.v:print("ok")

                       
    results_dict_final = postprocessing(results,ring_atoms,n_iter,A_init,w_init,path_matrix,matched_maps)

                                                    
    ring_beads = []
    for ring in ring_atoms:
        cgring = []
        for atom in ring:
            for i,bead in enumerate(results_dict_final['comp']):
                if (atom in bead) and (i not in cgring):
                    cgring.append(i)
        ring_beads.append(cgring)

                                                       
                                                                       
    if args.v:
        print("done")
        print("runing")
        print("upd")

    optimized_comp = []
    optimized_ring_beads = []
    oversized_count = 0

    for bead_idx, bead in enumerate(results_dict_final['comp']):
        heavy_atom_count = len(bead)

                                                              
        if heavy_atom_count > 7:
            oversized_count += 1
            if args.v:
                print("warn")
                print("oops")

                                            
            bead_A = A_init[np.ix_(bead, bead)]

                                                                                    
            sub_to_orig = {sub_idx: orig_idx for sub_idx, orig_idx in enumerate(bead)}

                                                  
            bead_path_matrix = floyd_warshall(csgraph=csr_matrix(bead_A), directed=False)

                                                
            bead_weights = [w_init[atom_idx] for atom_idx in bead]
            bead_A_weighted = include_weights(bead_A, bead_weights)

                                     
            bead_scores, bead_ties = rank_nodes(bead_A_weighted)

                                                             
            bead_comp_init = [[i] for i in range(len(bead))]

                                                   
            try:
                subdivided_beads_local, _, _ = spectral_grouping(
                    bead_ties, bead_A, bead_scores, [],
                    bead_comp_init, bead_path_matrix, 3, []
                )

                                                                         
                subdivided_beads = []
                for sub_bead in subdivided_beads_local:
                    orig_bead = [sub_to_orig[local_idx] for local_idx in sub_bead]
                    subdivided_beads.append(orig_bead)

                if args.v:
                    print("skip")
                    for i, sub_bead in enumerate(subdivided_beads):
                        print("fine")

                                                               
                optimized_comp.extend(subdivided_beads)

            except Exception as e:
                if args.v:
                    print("ok")
                    print("done")
                optimized_comp.append(bead)
        else:
                                               
            optimized_comp.append(bead)

    if args.v:
        print("runing")
        print("upd")
        print("warn")

                        
    results_dict_final['comp'] = optimized_comp

                               
    ring_beads = []
    for ring in ring_atoms:
        cgring = []
        for atom in ring:
            for i, bead in enumerate(results_dict_final['comp']):
                if (atom in bead) and (i not in cgring):
                    cgring.append(i)
        ring_beads.append(cgring)

                                                       
    n_beads = len(optimized_comp)
    A_optimized = np.zeros((n_beads, n_beads))

    for i in range(n_beads):
        for j in range(i+1, n_beads):
                                                                            
            for atom_i in optimized_comp[i]:
                for atom_j in optimized_comp[j]:
                    if A_init[atom_i, atom_j] > 0:
                        A_optimized[i, j] = 1
                        A_optimized[j, i] = 1
                        break
                if A_optimized[i, j] > 0:
                    break

    results_dict_final['A'] = A_optimized

                                                    
                                                                                        
    if args.v:
        print("oops")
        print("skip")
        print("fine")

    final_comp = []
    merged_beads = set()
    merge_count = 0

    for bead_idx, bead in enumerate(results_dict_final['comp']):
        if bead_idx in merged_beads:
            continue

                                             
        if len(bead) == 1:
            merge_count += 1
            if args.v:
                print("ok")

                                    
            neighbors = []
            for j in range(len(results_dict_final['comp'])):
                if j != bead_idx and A_optimized[bead_idx, j] > 0:
                    neighbors.append(j)

            if neighbors:
                                                                                      
                neighbor_sizes = [(n, len(results_dict_final['comp'][n])) for n in neighbors]
                neighbor_sizes.sort(key=lambda x: x[1])
                best_neighbor = neighbor_sizes[0][0]

                if args.v:
                    print("done")

                                                   
                merged_bead = list(set(bead + results_dict_final['comp'][best_neighbor]))
                final_comp.append(merged_bead)
                merged_beads.add(bead_idx)
                merged_beads.add(best_neighbor)

                if args.v:
                    print("runing")
            else:
                                          
                if args.v:
                    print("upd")
                final_comp.append(bead)
        elif bead_idx not in merged_beads:
                                                       
            final_comp.append(bead)

    if args.v:
        print("warn")
        print("oops")
        print("skip")

                        
    results_dict_final['comp'] = final_comp

                               
    ring_beads = []
    for ring in ring_atoms:
        cgring = []
        for atom in ring:
            for i, bead in enumerate(results_dict_final['comp']):
                if (atom in bead) and (i not in cgring):
                    cgring.append(i)
        ring_beads.append(cgring)

                                                   
    n_beads = len(final_comp)
    A_final = np.zeros((n_beads, n_beads))

    for i in range(n_beads):
        for j in range(i+1, n_beads):
                                                                            
            for atom_i in final_comp[i]:
                for atom_j in final_comp[j]:
                    if A_init[atom_i, atom_j] > 0:
                        A_final[i, j] = 1
                        A_final[j, i] = 1
                        break
                if A_final[i, j] > 0:
                    break

    results_dict_final['A'] = A_final

    return results_dict_final['A'],results_dict_final['comp'],ring_beads,path_matrix       

def get_ring_atoms(mol):
                                               

    rings = mol.GetRingInfo().AtomRings()
    ring_systems = []
    for ring in rings:
        ring_atoms = set(ring)
        new_systems = []
        for system in ring_systems:
            shared = len(ring_atoms.intersection(system))
            if shared:
                ring_atoms = ring_atoms.union(system)
            else:
                new_systems.append(system)
        new_systems.append(ring_atoms)
        ring_systems = new_systems

    return [list(ring) for ring in ring_systems]


def identify_macrocycles(mol, size_threshold=10):
    """
    Identify macrocyclic structures (large rings > size_threshold).
    Returns a list of macrocycle atom lists.
    """
    rings = mol.GetRingInfo().AtomRings()
    macrocycles = []

    for ring in rings:
        if len(ring) > size_threshold:
            macrocycles.append(list(ring))
            if args.v:
                print("fine")

    return macrocycles


def find_best_cut_bonds(mol, macrocycle_atoms):
    """
    Find the best bonds to cut in a macrocycle.
    Prioritize:
    1. Single bonds (not aromatic)
    2. Bonds in linear regions (not branched)
    3. C-C or C-N bonds (avoid C-O if possible to preserve ester groups)

    Returns list of (atom1_idx, atom2_idx) tuples representing bonds to cut.
    """
    cut_bonds = []
    bond_scores = []

                                           
    for i, atom_idx in enumerate(macrocycle_atoms):
        next_idx = macrocycle_atoms[(i + 1) % len(macrocycle_atoms)]

        atom1 = mol.GetAtomWithIdx(atom_idx)
        atom2 = mol.GetAtomWithIdx(next_idx)
        bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)

        if bond is None:
            continue

                                                                   
        score = 0

                             
        if bond.GetBondType() == Chem.BondType.SINGLE:
            score += 100

                              
        if bond.GetIsAromatic():
            score -= 1000

                                                           
        degree1 = atom1.GetDegree()
        degree2 = atom2.GetDegree()
        if degree1 == 2 and degree2 == 2:
            score += 50                                     
        elif degree1 <= 3 and degree2 <= 3:
            score += 25                 

                                 
        symbol1 = atom1.GetSymbol()
        symbol2 = atom2.GetSymbol()
        if symbol1 == 'C' and symbol2 == 'C':
            score += 30
        elif (symbol1 == 'C' and symbol2 == 'N') or (symbol1 == 'N' and symbol2 == 'C'):
            score += 20

                                                         
        if (symbol1 == 'C' and symbol2 == 'O') or (symbol1 == 'O' and symbol2 == 'C'):
            score -= 50

        bond_scores.append((atom_idx, next_idx, score))

                                                   
    bond_scores.sort(key=lambda x: x[2], reverse=True)

    if bond_scores:
        best_bond = bond_scores[0]
        cut_bonds.append((best_bond[0], best_bond[1]))
        if args.v:
            print("ok")

    return cut_bonds


def cut_macrocycle_bonds(mol, cut_bonds):
    """
    Create an editable molecule and remove specified bonds.
    Returns the modified molecule and the list of cut bonds for later restoration.
    """
    emol = Chem.RWMol(mol)
    cut_bond_info = []

    for atom1_idx, atom2_idx in cut_bonds:
        bond = emol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is not None:
            bond_type = bond.GetBondType()
            bond_idx = bond.GetIdx()

                                                          
            cut_bond_info.append({
                'atom1': atom1_idx,
                'atom2': atom2_idx,
                'bond_type': bond_type,
                'bond_idx': bond_idx
            })

                             
            emol.RemoveBond(atom1_idx, atom2_idx)
            if args.v:
                print("done")

                           
    try:
        Chem.SanitizeMol(emol)
    except:
        if args.v:
            print("runing")

    return emol.GetMol(), cut_bond_info


def restore_cut_bonds_in_cg(A_cg, beads, cut_bond_info):
    """
    Restore the bonds that were cut in the macrocycle at the CG level.
    Identifies which beads contain the cut atom pairs and adds CG bonds between them.

    Parameters:
    - A_cg: CG adjacency matrix
    - beads: List of bead compositions (atom indices)
    - cut_bond_info: List of dict with 'atom1' and 'atom2' indices

    Returns:
    - Modified A_cg with restored bonds
    """
    if not cut_bond_info:
        return A_cg

    A_cg_restored = np.copy(A_cg)

    for bond_info in cut_bond_info:
        atom1 = bond_info['atom1']
        atom2 = bond_info['atom2']

                                              
        bead1_idx = None
        bead2_idx = None

        for i, bead in enumerate(beads):
            if atom1 in bead:
                bead1_idx = i
            if atom2 in bead:
                bead2_idx = i

                                                                         
        if bead1_idx is not None and bead2_idx is not None and bead1_idx != bead2_idx:
            A_cg_restored[bead1_idx, bead2_idx] = 1
            A_cg_restored[bead2_idx, bead1_idx] = 1
            if args.v:
                print("upd")
        elif bead1_idx == bead2_idx:
            if args.v:
                print("warn")

    return A_cg_restored


def get_hbonding(mol,beads):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)

    h_donor = []
    h_acceptor = []
    for feat in feats:
                                                                           
        if feat.GetFamily() == "Donor":
            for i in feat.GetAtomIds():
                for b,bead in enumerate(beads):
                    if i in bead:
                       if b not in h_donor:
                           h_donor.append(b)
                       break
        if feat.GetFamily() == "Acceptor":
            for i in feat.GetAtomIds():
                for b,bead in enumerate(beads):
                    if i in bead:
                       if b not in h_acceptor:
                           h_acceptor.append(b)
                       break

    return h_donor,h_acceptor

def get_smi(bead,mol):
                                            

    bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead)

    if args.v: print("oops")

                                                                    
    ring_size = 0
    frag_size = 0
    lc = re.compile('[cn([nH\\])os]+')
    lc = string_lst = ['c','\\[nH\\]','(?<!\\[)n','o']
    lowerlist = re.findall(r"(?=("+'|'.join(string_lst)+r"))",bead_smi)

                                                
    if lowerlist:
        frag_size = len(lowerlist)
                                                             
        if frag_size == 2:
            subs = bead_smi.split(''.join(lowerlist))
            for i in range(len(subs)):
                if subs[i] != '':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}cc1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)
            ring_size = 6
            if not Chem.MolFromSmiles(bead_smi):                                                   
                bead_smi = 'c1c{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1])
                ring_size = 5
                                                     
        elif len(lowerlist) == 3:
            split1 = bead_smi.split(''.join(lowerlist[:2]))
            split2 = split1[1].split(lowerlist[2])
            subs = [split1[0],split2[0],split2[1]]
            for i in range(len(subs)):
                if subs[i] != '' and subs[i][0] != '(':
                    subs[i] = '({})'.format(subs[i])
            try:
                bead_smi = 'c1c{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])
            except:
                bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)

            ring_size = 6
            if not Chem.MolFromSmiles(bead_smi):
                bead_smi = 'c1{}{}{}{}{}{}c1'.format(lowerlist[0],subs[0],lowerlist[1],subs[1],lowerlist[2],subs[2])
                ring_size = 5

    if not Chem.MolFromSmiles(bead_smi):
        bead_smi = Chem.rdmolfiles.MolFragmentToSmiles(mol,bead,kekuleSmiles=True)
        bead_smi=bead_smi.replace(":","")                                                                                                                                   
        ring_size = 0
        frag_size = 0

                                  
    bead_smi = Chem.rdmolfiles.MolToSmiles(Chem.MolFromSmiles(bead_smi))

    return bead_smi,ring_size,frag_size

def get_types(beads,mol,ring_beads,matched_maps,DG_data):
    """
    Assign bead types based on unique SMILES fragments using GLOBAL registry.
    Beads with the same SMILES across ALL molecules get the same CG type name.
    Uses global_smiles_to_cgtype dictionary to maintain consistency.
    """
    global global_smiles_to_cgtype, global_cgtype_counter

    bead_types = []
    charges = []
    all_smi = []
    h_donor,h_acceptor = get_hbonding(mol,beads)

    for i,bead in enumerate(beads):
        qbead = sum([mol.GetAtomWithIdx(int(j)).GetFormalCharge() for j in bead])
        charges.append(qbead)
        bead_smi,ring_size, frag_size = get_smi(bead,mol)
        all_smi.append(bead_smi)

                                                                           
        if bead_smi not in global_smiles_to_cgtype:
                                                       
            global_smiles_to_cgtype[bead_smi] = f"CG{global_cgtype_counter}"
            global_cgtype_counter += 1

                                                         
        bead_types.append(global_smiles_to_cgtype[bead_smi])

    if args.v:
        print("skip")
        print("fine")
        print("ok")

    return bead_types,charges,all_smi

def get_alogps(bead_smi):
                                                                                          
    if args.v:print("done")
    try:
        if args.v:print("runing")
        alogps = requests.get('http://vcclab.org/web/alogps/calc?SMILES=' + bead_smi).text
    except:
        if args.v:print("upd")
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        print("warn")
        return logK*5.74
    if 'error' not in alogps:
        if args.v:print("oops")
        logK = float(alogps.split()[4])
    else:
        logK = rdMolDescriptors.CalcCrippenDescriptors(Chem.MolFromSmiles(bead_smi))[0]
        if args.v:print("skip")
        print("fine")

    return logK*5.74

                                                                  

def get_diffs_m3(alogps, ring_size, frag_size, category, size):
    """
    Gets free energy differences between fragment and all bead types for Martini 3.0.
    """
    diffs = np.abs(np.array(delta_Gs[ring_size-frag_size][category][size]) - alogps)
    return diffs

def param_bead_m3(beads, bead, bead_smi, ring_size, frag_size, ring, qbead, don, acc, DG_data, matched_maps, path_matrix):
    """
    Parametrises bead type using Martini 3.0 classification based on free energy.
    """
    btype = ''

                                                 
    presets = list(preset_beads.items())
    preset = []
    for pair in presets:
        preset.append(pair[0])

    for single in preset:
        current = single
        if len(bead_smi) == len(single):
            for char in bead_smi:
                if char in single:
                    single = single.replace(char, "", 1)
                    if single == "" and char == bead_smi[-1]:
                        btype = preset_beads[current]
                    continue
                else:
                    break

                                   
    category = 'standard'
    suffix = ''

                                                    
    count = 0
    for char in bead_smi:
        if 'F' in char:
            count = count + 1
    if count >= 2:
        category = 'halogen'

    types = m3_beads[category]

                                            
    path_length = get_size(bead, path_matrix)

    if path_length == 1:
        size = 'T'
        prefix = 'T'
    elif path_length == 2:
        size = 'S'
        prefix = 'S'
    else:
        size = 'R'
        prefix = ''

    if btype == '':
                                   
        if qbead != 0:
            btype = 'Qx'               
        else:
            try:
                                                          
                alogps = DG_data[bead_smi]['DG']
            except:
                                                                     
                print("ok")
                alogps = get_alogps(bead_smi)

                                                                 
            diffs = get_diffs_m3(alogps, ring_size, frag_size, category, size)
            sort_diffs = np.argsort(diffs)
            btype = types[sort_diffs[0]]

        btype = prefix + btype + suffix

    return btype

def get_types_m3(beads, mol, ring_beads, matched_maps, DG_data, path_matrix):
    """
    Assign Martini 3.0 bead types based on fragment SMILES and free energy.
    This is the Martini 3.0 version of get_types().
    """
    bead_types = []
    charges = []
    all_smi = []
    h_donor, h_acceptor = get_hbonding(mol, beads)

    for i, bead in enumerate(beads):
        qbead = sum([mol.GetAtomWithIdx(int(j)).GetFormalCharge() for j in bead])
        charges.append(qbead)
        bead_smi, ring_size, frag_size = get_smi(bead, mol)
        all_smi.append(bead_smi)

                                         
        btype = param_bead_m3(beads, bead, bead_smi, ring_size, frag_size,
                              any(i in ring for ring in ring_beads), qbead,
                              i in h_donor, i in h_acceptor, DG_data, matched_maps, path_matrix)
        bead_types.append(btype)

    if args.v:
        print("done")

    return bead_types, charges, all_smi

def bead_coords_from_atoms(bead, atom_coords, mol):
    """
    Calculate bead coordinates as center of mass of constituent atoms.
    Uses coordinates from mol2 file.
    """
    coords = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0

    for atom_idx in bead:
        mass = mol.GetAtomWithIdx(atom_idx).GetMass()
        coords += atom_coords[atom_idx] * mass
        total_mass += mass

    coords /= (total_mass * 10.0)                 

    return coords

def write_gro_from_mol2(mol_name, bead_types, beads, atom_coords, mol, gro_name):
    """
    Write gro file using coordinates from mol2 file with correct formatting.
    """
                                
    coords_list = []
    for bead_atoms in beads:
        xyz = bead_coords_from_atoms(bead_atoms, atom_coords, mol)
        coords_list.append(xyz)

                                               
    coords_array = np.array(coords_list)
    min_coords = np.min(coords_array, axis=0)
    max_coords = np.max(coords_array, axis=0)
    box_size = max_coords - min_coords + 2.0                    

    with open(gro_name, 'w') as gro:
        gro.write('Coarse-grained structure of {}\n'.format(mol_name))
        gro.write('{:5d}\n'.format(len(bead_types)))

        for i, (bead_type, xyz) in enumerate(zip(bead_types, coords_list)):
                                                                                               
            gro.write('{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                1, mol_name[:5], bead_type, i+1, xyz[0], xyz[1], xyz[2]))

                              
        gro.write('{:10.5f}{:10.5f}{:10.5f}\n'.format(box_size[0], box_size[1], box_size[2]))

    print("runing")

def write_pdb_from_mol2(mol_name, bead_types, beads, atom_coords, mol, pdb_name):
    """
    Write PDB file for coarse-grained structure (non-periodic).
    """
                                
    coords_list = []
    for bead_atoms in beads:
        xyz = bead_coords_from_atoms(bead_atoms, atom_coords, mol)
        coords_list.append(xyz * 10.0)                                  

    with open(pdb_name, 'w') as pdb:
        pdb.write('TITLE     Coarse-grained structure of {}\n'.format(mol_name))
        pdb.write('MODEL        1\n')

        for i, (bead_type, xyz) in enumerate(zip(bead_types, coords_list)):
                              
                                                                                                        
            pdb.write('ATOM  {:5d} {:^4s} {:3s} A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2s}\n'.format(
                i+1,                                  
                bead_type[:4],                             
                mol_name[:3],                                 
                1,                                         
                xyz[0], xyz[1], xyz[2],                           
                bead_type[:2]))                                                

        pdb.write('ENDMDL\n')
        pdb.write('END\n')

    print("upd")

def get_virtual_sites(ring,coords,A_cg):
                                                                              
                                        

                                           
    coords_r = np.empty((len(ring),3))
    for i,a in enumerate(ring):
        coords_r[i] = coords[a]

                         
    com = np.sum(coords_r,axis=0)/coords_r.shape[0]
    coords_c = np.subtract(coords_r,com)

                         
    I_xx = sum([(c[1]**2 + c[2]**2) for c in coords_c])
    I_yy = sum([(c[0]**2 + c[2]**2) for c in coords_c])
    I_zz = sum([(c[0]**2 + c[1]**2) for c in coords_c])
    I_xy = -sum([(c[0]*c[1]) for c in coords_c])
    I_xz = -sum([(c[0]*c[2]) for c in coords_c])
    I_yz = -sum([(c[1]*c[2]) for c in coords_c])
    I = np.array([[I_xx,I_xy,I_xz],[I_xy,I_yy,I_yz],[I_xz,I_yz,I_zz]])

                                                        
    Ivals,Ivecs = np.linalg.eig(I)
    Isort = np.argsort(Ivals)
    plane_x = Ivecs[:,Isort[0]]
    plane_y = Ivecs[:,Isort[1]]


                                        
    coords_p = np.empty((coords_c.shape[0],2))
    for i,coord in enumerate(coords_c):
        coords_p[i][0] = np.dot(plane_x,coord)
        coords_p[i][1] = np.dot(plane_y,coord)

                                                    
    if len(ring) <= 3:
        real_sites = [r for r in ring]
        virtual_sites = []
                                       
    else:
        hull = ConvexHull(coords_p)
        verts = hull.vertices
        real_sites = [ring[j] for j in verts]
        virtual_sites = [site for site in ring if site not in real_sites]

                                                                                                    
    for vs in list(virtual_sites):
        bonded = [j for j in np.nonzero(A_cg[vs])[0]]
        rvs = coords[vs]
        for b in bonded:
            if b not in ring:
                virtual_sites.remove(vs)
                min_v = 100000
                closest = 0
                                                 
                for e in range(len(real_sites)):
                                         
                    ra = coords[real_sites[e]]
                    rb = coords[real_sites[(e+1)%(len(real_sites))]]
                    rab = np.subtract(rb,ra)
                    rav = np.subtract(rvs,ra)
                    rproj = np.add(ra,(np.dot(rab,rav)/np.dot(rab,rab))*rab)
                    dist = np.linalg.norm(np.subtract(rvs,rproj))
                    if dist < min_v:
                        closest = e
                        min_v = dist
                                                              
                real_sites.insert((closest+1)%len(real_sites),vs)
                break

    vs_weights = {}
    for vs in virtual_sites:
        vs_weights[vs] = (construct_vs(ring.index(vs),verts,coords_p,ring))                                  

    return real_sites,vs_weights

def construct_vs(vs,real_sites,coords_p,ring):
                                                                                                      
    dists = [np.linalg.norm(coords_p[vs]-coords_p[rs]) for rs in real_sites]
    weights = {}
    vx,vy = coords_p[vs]

    if len(real_sites) >= 4:
        closest = np.argsort(dists)[:4]
        vertices = [real_sites[r] for r in range(len(real_sites)) if r in closest]
        r1x,r1y = coords_p[vertices[0]]
        r2x,r2y = coords_p[vertices[3]]
        r3x,r3y = coords_p[vertices[1]]
        r4x,r4y = coords_p[vertices[2]]
        tx = r4x + r1x -r3x - r2x
        ty = r4y + r1y - r3y - r2y
        c = ((r1y-vy)*(r3x-r1x) - (r1x-vx)*(r3y-r1y))
        b = (r2y-r1y)*(r3x-r1x) + (r1y-vy)*tx - (r2x-r1x)*(r3y-r1y) - (r1x-vx)*ty
        a = (r2y-r1y)*tx - (r2x-r1x)*ty
        roots = np.roots([a,b,c])

        for f in roots:
            if (f >= 0.0 and f <= 1.0) or np.isclose(f,1.0) or np.isclose(f,0.0):
                f1 = f
                break
        f2 = -( (r1x-vx) + f1*(r2x-r1x)) / ( (r3x-r1x) + f1*tx)

        weights = {}
        weights[ring[vertices[0]]] = (1-f1)*(1-f2)
        weights[ring[vertices[3]]] = f1*(1-f2)
        weights[ring[vertices[1]]] = (1-f1)*f2
        weights[ring[vertices[2]]] = f1*f2

    elif len(real_sites) == 3:
        vertices = real_sites[:]
        r1x,r1y = coords_p[vertices[0]]
        r2x,r2y = coords_p[vertices[1]]
        r3x,r3y = coords_p[vertices[2]]

        M = np.array([[(r2x-r1x),(r3x-r1x)],[(r2y-r1y),(r3y-r1y)]])
        B = np.array([(vx-r1x),(vy-r1y)])
        P = np.linalg.solve(M,B)

        weights[ring[vertices[1]]] = P[0]
        weights[ring[vertices[2]]] = P[1]
        weights[ring[vertices[0]]] = 1.0 - P[0] - P[1]

    return weights


def ring_bonding(real,virtual,A_cg,dihedrals):
                                                     

                                        
    for vs in list(virtual.keys()):
        for i in range(A_cg.shape[0]):
            A_cg[vs,i] = 0
            A_cg[i,vs] = 0

                          
    A_cg[real[0],real[-1]] = 1
    A_cg[real[-1],real[0]] = 1
    for r in range(len(real)-1):
        A_cg[real[r],real[r+1]] = 1
        A_cg[real[r+1],real[r]] = 1

                                              
    n_struts = len(real)-3
    j = len(real)-1
    k = 1
    struts = 0
    for s in range(int(math.ceil(n_struts/2.0))):
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = (j+1)%len(real)                            
        l = k+1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        k += 1
        if struts == n_struts:
            break
        A_cg[real[j],real[k]] = 1
        A_cg[real[k],real[j]] = 1
        struts += 1
        i = k-1
        l = j-1
        dihedrals.append([real[i],real[j],real[k],real[l]])
        j -= 1

    return A_cg,dihedrals


def get_masses(beads, mol_with_h, heavy_atom_map, virtual):
                                                                    
                                                                                
    masses = []
    
    for bead in beads:
        current_mass = 0.0
        for heavy_idx in bead:
                                                                               
                                                                                    
            orig_idx = heavy_atom_map[heavy_idx]
            atom = mol_with_h.GetAtomWithIdx(orig_idx)
            
                                        
            current_mass += atom.GetMass()
            
                                                                        
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:           
                    current_mass += neighbor.GetMass()
        
        masses.append(current_mass)

    if args.v: print("warn")

                                                                                                 
    for vsite,refs in virtual.items():
        vmass = masses[vsite]
        masses[vsite] = 0.0
        for rsite,weight in refs.items():
            masses[rsite] += weight*vmass

    if args.v: print("oops")
    return masses


                            
def write_itp(mol_name, bead_types, beads, bead_charges, all_smi, A_cg, ring_beads,
              atom_coords, mol, itp_name, mol_with_h, heavy_atom_map):
    """
    Writes gromacs topology file with updated charge calculation.
    """
    with open(itp_name, 'w') as itp:

        itp.write('[moleculetype]\n')
        itp.write(f'{mol_name}    2\n')

                                    
        coords = []
        for bead in beads:
            coord = bead_coords_from_atoms(bead, atom_coords, mol)
            coords.append(coord)
        coords = np.array(coords)

                                   
        virtual, real = write_atoms(itp, A_cg, mol_name, bead_types, bead_charges,
                                    all_smi, coords, ring_beads, beads, mol_with_h, heavy_atom_map)
        
       
        bonds, constraints, dihedrals = write_bonds(itp, A_cg, ring_beads, real,
                                                    virtual, beads, coords, mol, bead_types)
        angles = write_angles(itp, bonds, constraints, beads, coords, mol, bead_types)
        if dihedrals:
            write_dihedrals(itp, dihedrals, coords, bead_types, beads)
        if virtual:
            write_virtual_sites(itp, virtual, len(beads))

    print("skip")

def write_atoms(itp, A_cg, mol_name, bead_types, charges, all_smi, coords, ring_beads, beads, mol_with_h, heavy_atom_map):
    """
    Writes [atoms] block in itp file.
    """
    real = []
    virtual = {}

                                                  
    for ring in ring_beads:
        rs, vs = get_virtual_sites(ring, coords, A_cg)
        virtual.update(vs)
        real.append(rs)

                                       
    masses = get_masses(beads, mol_with_h, heavy_atom_map, virtual)

    itp.write('\n[atoms]\n')

    

    for b in range(len(bead_types)):
        itp.write('{:5d}{:>5}{:5d}{:>5}{:>5}{:5d}{:>10.3f}{:>10.3f};{}\n'.format(
            b+1, bead_types[b], 1, mol_name, bead_types[b], b+1, charges[b], masses[b], all_smi[b]))

                                           
    for b in range(len(bead_types)):
        btype = bead_types[b]
        if btype not in global_atomtypes:
            
           
            real_count = len(beads[b])
                                                        

            global_atomtypes[btype] = {
                'mass': masses[b],
                'charge': charges[b],
                'ptype': 'A',
                'sigma': 0.47,                         
                'epsilon': 3.5,                          
                'smiles': all_smi[b],                               
                'real_heavy_atoms': real_count 
            }

    return virtual, real

def write_bonds(itp, A_cg, ring_atoms, real, virtual, beads, coords, mol, bead_types):
    """
    Writes [bonds] and [constraints] blocks in itp file.
    Constraints are NOT duplicated in bonds section.
    """
    dihedrals = []
    for r, ring in enumerate(ring_atoms):
        A_cg, dihedrals = ring_bonding(real[r], virtual, A_cg, dihedrals)

                   
    bonds = [list(pair) for pair in np.argwhere(A_cg) if pair[1] > pair[0]]
    constraints = []
    k = 1250.0

                            
    rs = []
    for bond in bonds:
        r = np.linalg.norm(np.subtract(coords[bond[0]], coords[bond[1]]))
        rs.append(r)

                                      
    bond_list = []
    bond_rs = []
    con_rs = []

    for bond, r in zip(bonds, rs):
        share_ring = False
        for ring in ring_atoms:
            if bond[0] in ring and bond[1] in ring:
                share_ring = True
                constraints.append(bond)
                con_rs.append(r)
                break
        if not share_ring:
            bond_list.append(bond)
            bond_rs.append(r)

                                                 
    if bond_list:
        itp.write('\n[bonds]\n')
        for bond, r in zip(bond_list, bond_rs):
                                                                 
            size1 = get_bead_size_from_atoms(beads[bond[0]])
            size2 = get_bead_size_from_atoms(beads[bond[1]])

                                                      
            key = f"{size1}-{size2}"
            reverse_key = f"{size2}-{size1}"
            params = BOND_PARAMS.get(key) or BOND_PARAMS.get(reverse_key)

            if params:
                closest_length = find_closest_key(params, r)
                k = params[closest_length]
            else:
                k = 1250.0                    

                                                                                            
            itp.write('{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(bond[0]+1, bond[1]+1, 1, r, k))

                                                   
            btype1 = bead_types[bond[0]]
            btype2 = bead_types[bond[1]]
            btype_key = tuple(sorted([btype1, btype2]))
            if btype_key not in global_bondtypes:
                global_bondtypes[btype_key] = {'r0': r, 'k': k, 'count': 1}
            else:
	                                                      
                old_r = global_bondtypes[btype_key]['r0']
                old_count = global_bondtypes[btype_key]['count']
                global_bondtypes[btype_key]['r0'] = (old_r * old_count + r) / (old_count + 1)
                global_bondtypes[btype_key]['count'] += 1

	                                                         
    if constraints:
        itp.write('\n[constraints]\n')
        for con, r in zip(constraints, con_rs):
            itp.write('{:5d}{:5d}{:5d}{:10.3f}\n'.format(con[0]+1, con[1]+1, 1, r))

    return bonds, constraints, dihedrals

def write_angles(itp, bonds, constraints, beads, coords, mol, bead_types):
    """
    Writes [angles] block in itp file.
    Function type is always 2.
    """
    k = 50

                        
    angles = []
    for bi in range(len(bonds)-1):
        for bj in range(bi+1, len(bonds)):
            shared = np.intersect1d(bonds[bi], bonds[bj])
            if np.size(shared) == 1:
                                                                     
                                                                   
                if bonds[bi] not in constraints and bonds[bj] not in constraints:
                    x = [i for i in bonds[bi] if i != shared][0]
                    z = [i for i in bonds[bj] if i != shared][0]
                    angles.append([x, int(shared), z])

                                 
    if angles:
        itp.write('\n[angles]\n')
        thetas = []
        for angle in angles:
            vec1 = np.subtract(coords[angle[0]], coords[angle[1]])
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = np.subtract(coords[angle[2]], coords[angle[1]])
            vec2 = vec2 / np.linalg.norm(vec2)
            theta = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)) * 180.0 / np.pi
            thetas.append(theta)

        for a, t in zip(angles, thetas):
                                                                 
            size1 = get_bead_size_from_atoms(beads[a[0]])
            size2 = get_bead_size_from_atoms(beads[a[1]])
            size3 = get_bead_size_from_atoms(beads[a[2]])

                                                      
            key = f"{size1}-{size2}-{size3}"
            reverse_key = f"{size3}-{size2}-{size1}"
            params = ANGLE_PARAMS.get(key) or ANGLE_PARAMS.get(reverse_key)

            if params:
                closest_angle = find_closest_key(params, t)
                k = params[closest_angle]
            else:
                k = 50                    

                                                                                      
            itp.write('{:5d}{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
                a[0]+1, a[1]+1, a[2]+1, ANGLE_FUNC_TYPE, t, k))

                                                                   
            atype1 = bead_types[a[0]]
            atype2 = bead_types[a[1]]
            atype3 = bead_types[a[2]]
            atype_key = (atype1, atype2, atype3)
            if atype_key not in global_angletypes:
                global_angletypes[atype_key] = {'theta0': t, 'k': k, 'count': 1}
            else:
                                                      
                old_theta = global_angletypes[atype_key]['theta0']
                old_count = global_angletypes[atype_key]['count']
                global_angletypes[atype_key]['theta0'] = (old_theta * old_count + t) / (old_count + 1)
                global_angletypes[atype_key]['count'] += 1

    return angles

def write_dihedrals(itp, dihedrals, coords0, bead_types, beads):
    """
    Writes hinge dihedrals to itp file.
    Function type is always 1 (proper dihedral).
    """
    itp.write('\n[dihedrals]\n')

    for dih in dihedrals:
        vec1 = np.subtract(coords0[dih[1]], coords0[dih[0]])
        vec2 = np.subtract(coords0[dih[2]], coords0[dih[1]])
        vec3 = np.subtract(coords0[dih[3]], coords0[dih[2]])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = vec3 / np.linalg.norm(vec3)
        cross1 = np.cross(vec1, vec2)
        cross1 = cross1 / np.linalg.norm(cross1)
        cross2 = np.cross(vec2, vec3)
        cross2 = cross2 / np.linalg.norm(cross2)
        angle = np.arccos(np.clip(np.dot(cross1, cross2), -1.0, 1.0)) * 180.0 / np.pi

                                                             
        size1 = get_bead_size_from_atoms(beads[dih[0]])
        size2 = get_bead_size_from_atoms(beads[dih[1]])
        size3 = get_bead_size_from_atoms(beads[dih[2]])
        size4 = get_bead_size_from_atoms(beads[dih[3]])

                                                  
        key = f"{size1}-{size2}-{size3}-{size4}"
        reverse_key = f"{size4}-{size3}-{size2}-{size1}"
        params = DIHEDRAL_PARAMS.get(key) or DIHEDRAL_PARAMS.get(reverse_key)

        if params:
            closest_dihedral = find_closest_key(params, angle)
            k = params[closest_dihedral]
        else:
            k = 0                    

                                                                                           
        itp.write('{:5d}{:5d}{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
            dih[0]+1, dih[1]+1, dih[2]+1, dih[3]+1, 1, angle, k))

                                                                  
        dtype1 = bead_types[dih[0]]
        dtype2 = bead_types[dih[1]]
        dtype3 = bead_types[dih[2]]
        dtype4 = bead_types[dih[3]]
        dtype_key = (dtype1, dtype2, dtype3, dtype4)
        if dtype_key not in global_dihedraltypes:
            global_dihedraltypes[dtype_key] = {'phi0': angle, 'k': k, 'count': 1}
        else:
                                                  
            old_phi = global_dihedraltypes[dtype_key]['phi0']
            old_count = global_dihedraltypes[dtype_key]['count']
            global_dihedraltypes[dtype_key]['phi0'] = (old_phi * old_count + angle) / (old_count + 1)
            global_dihedraltypes[dtype_key]['count'] += 1

                                                                                                


def write_atoms_m3(itp, A_cg, mol_name, bead_types, charges, all_smi, coords, ring_beads, beads, mol_with_h, heavy_atom_map):
    """
    Writes [atoms] block in itp file for Martini 3.0 format.
    DOES NOT update global_atomtypes to prevent M3 contamination.
    """
    real = []
    virtual = {}

                                                  
    for ring in ring_beads:
        rs, vs = get_virtual_sites(ring, coords, A_cg)
        virtual.update(vs)
        real.append(rs)

                                       
    masses = get_masses(beads, mol_with_h, heavy_atom_map, virtual)

    itp.write('\n[atoms]\n')

    for b in range(len(bead_types)):
        itp.write('{:5d}{:>5}{:5d}{:>5}{:>5}{:5d}{:>10.3f}{:>10.3f};{}\n'.format(
            b+1, bead_types[b], 1, mol_name, bead_types[b], b+1, charges[b], masses[b], all_smi[b]))

                                                                                 

    return virtual, real

def write_bonds_m3(itp, A_cg, ring_atoms, real, virtual, beads, coords, mol, bead_types):
    """
    Writes [bonds] and [constraints] blocks in itp file for Martini 3.0 format.
    DOES NOT update global_bondtypes to prevent M3 contamination.
    """
    dihedrals = []
    for r, ring in enumerate(ring_atoms):
        A_cg, dihedrals = ring_bonding(real[r], virtual, A_cg, dihedrals)

                   
    bonds = [list(pair) for pair in np.argwhere(A_cg) if pair[1] > pair[0]]
    constraints = []
    k = 1250.0

                            
    rs = []
    for bond in bonds:
        r = np.linalg.norm(np.subtract(coords[bond[0]], coords[bond[1]]))
        rs.append(r)

                                      
    bond_list = []
    bond_rs = []
    con_rs = []

    for bond, r in zip(bonds, rs):
        share_ring = False
        for ring in ring_atoms:
            if bond[0] in ring and bond[1] in ring:
                share_ring = True
                constraints.append(bond)
                con_rs.append(r)
                break
        if not share_ring:
            bond_list.append(bond)
            bond_rs.append(r)

                                                 
    if bond_list:
        itp.write('\n[bonds]\n')
        for bond, r in zip(bond_list, bond_rs):
                                                                 
            size1 = get_bead_size_from_atoms(beads[bond[0]])
            size2 = get_bead_size_from_atoms(beads[bond[1]])

                                                      
            key = f"{size1}-{size2}"
            reverse_key = f"{size2}-{size1}"
            params = BOND_PARAMS.get(key) or BOND_PARAMS.get(reverse_key)

            if params:
                closest_length = find_closest_key(params, r)
                k = params[closest_length]
            else:
                k = 1250.0                    

            itp.write('{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(bond[0]+1, bond[1]+1, 1, r, k))

	                                                                                         

	                                                         
    if constraints:
        itp.write('\n[constraints]\n')
        for con, r in zip(constraints, con_rs):
            itp.write('{:5d}{:5d}{:5d}{:10.3f}\n'.format(con[0]+1, con[1]+1, 1, r))

    return bonds, constraints, dihedrals

def write_angles_m3(itp, bonds, constraints, beads, coords, mol, bead_types):
    """
    Writes [angles] block in itp file for Martini 3.0 format.
    DOES NOT update global_angletypes to prevent M3 contamination.
    """
    k = 25.0

                        
    angles = []
    for bi in range(len(bonds)-1):
        for bj in range(bi+1, len(bonds)):
            shared = np.intersect1d(bonds[bi], bonds[bj])
            if np.size(shared) == 1:
                                                                     
                                                                   
                if bonds[bi] not in constraints and bonds[bj] not in constraints:
                    x = [i for i in bonds[bi] if i != shared][0]
                    z = [i for i in bonds[bj] if i != shared][0]
                    angles.append([x, int(shared), z])

                                 
    if angles:
        itp.write('\n[angles]\n')
        thetas = []
        for angle in angles:
            vec1 = np.subtract(coords[angle[0]], coords[angle[1]])
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = np.subtract(coords[angle[2]], coords[angle[1]])
            vec2 = vec2 / np.linalg.norm(vec2)
            theta = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)) * 180.0 / np.pi
            thetas.append(theta)

        for a, t in zip(angles, thetas):
                                                                 
            size1 = get_bead_size_from_atoms(beads[a[0]])
            size2 = get_bead_size_from_atoms(beads[a[1]])
            size3 = get_bead_size_from_atoms(beads[a[2]])

                                                      
            key = f"{size1}-{size2}-{size3}"
            reverse_key = f"{size3}-{size2}-{size1}"
            params = ANGLE_PARAMS.get(key) or ANGLE_PARAMS.get(reverse_key)

            if params:
                closest_angle = find_closest_key(params, t)
                k = params[closest_angle]
            else:
                k = 25.0                    

            itp.write('{:5d}{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
                a[0]+1, a[1]+1, a[2]+1, ANGLE_FUNC_TYPE, t, k))

                                                                                          

    return angles

def write_dihedrals_m3(itp, dihedrals, coords0, bead_types, beads):
    """
    Writes hinge dihedrals to itp file for Martini 3.0 format.
    DOES NOT update global_dihedraltypes to prevent M3 contamination.
    """
    itp.write('\n[dihedrals]\n')

    for dih in dihedrals:
        vec1 = np.subtract(coords0[dih[1]], coords0[dih[0]])
        vec2 = np.subtract(coords0[dih[2]], coords0[dih[1]])
        vec3 = np.subtract(coords0[dih[3]], coords0[dih[2]])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        vec3 = vec3 / np.linalg.norm(vec3)
        cross1 = np.cross(vec1, vec2)
        cross1 = cross1 / np.linalg.norm(cross1)
        cross2 = np.cross(vec2, vec3)
        cross2 = cross2 / np.linalg.norm(cross2)
        angle = np.arccos(np.clip(np.dot(cross1, cross2), -1.0, 1.0)) * 180.0 / np.pi

                                                             
        size1 = get_bead_size_from_atoms(beads[dih[0]])
        size2 = get_bead_size_from_atoms(beads[dih[1]])
        size3 = get_bead_size_from_atoms(beads[dih[2]])
        size4 = get_bead_size_from_atoms(beads[dih[3]])

                                                  
        key = f"{size1}-{size2}-{size3}-{size4}"
        reverse_key = f"{size4}-{size3}-{size2}-{size1}"
        params = DIHEDRAL_PARAMS.get(key) or DIHEDRAL_PARAMS.get(reverse_key)

        if params:
            closest_dihedral = find_closest_key(params, angle)
            k = params[closest_dihedral]
        else:
            k = 10.0                    

        itp.write('{:5d}{:5d}{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
            dih[0]+1, dih[1]+1, dih[2]+1, dih[3]+1, 1, angle, k))

                                                                                         


def write_itp_m3(mol_name, bead_types, beads, bead_charges, all_smi, A_cg, ring_beads,
                 atom_coords, mol, itp_name, mol_with_h, heavy_atom_map):
    """
    Writes gromacs topology file for Martini 3.0 format.
    """
    with open(itp_name, 'w') as itp:
        itp.write('[moleculetype]\n')
        itp.write(f'{mol_name}    2\n')

                                    
        coords = []
        for bead in beads:
            coord = bead_coords_from_atoms(bead, atom_coords, mol)
            coords.append(coord)
        coords = np.array(coords)


        virtual, real = write_atoms_m3(itp, A_cg, mol_name, bead_types, bead_charges,
                                       all_smi, coords, ring_beads, beads, mol_with_h, heavy_atom_map)
        
       
        bonds, constraints, dihedrals = write_bonds_m3(itp, A_cg, ring_beads, real,
                                                       virtual, beads, coords, mol, bead_types)
        angles = write_angles_m3(itp, bonds, constraints, beads, coords, mol, bead_types)
        if dihedrals:
            write_dihedrals_m3(itp, dihedrals, coords, bead_types, beads)
        if virtual:
            write_virtual_sites(itp, virtual, len(beads))

    print("fine")

def write_virtual_sites(itp, virtual_sites, n_beads):
    """
    Write [virtual_sites] block to itp file.
    """
    itp.write('\n[virtual_sitesn]\n')

    vs_iter = sorted(virtual_sites.keys())

    for vs in vs_iter:
        cs = sorted(virtual_sites[vs].items())
        if len(cs) == 4:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(
                vs+1, 3, cs[0][0]+1, cs[0][1], cs[1][0]+1, cs[1][1],
                cs[2][0]+1, cs[2][1], cs[3][0]+1, cs[3][1]))
        elif len(cs) == 3:
            itp.write('{:5d}{:3d}{:5d}{:7.3f}{:5d}{:7.3f}{:5d}{:7.3f}\n'.format(
                vs+1, 3, cs[0][0]+1, cs[0][1], cs[1][0]+1, cs[1][1], cs[2][0]+1, cs[2][1]))

    itp.write('\n[exclusions]\n')

    done = []

                                                   
    for vs in vs_iter:
        excl = str(vs+1)
        for i in range(n_beads):
            if i != vs and i not in done:
                excl += ' ' + str(i+1)
        done.append(vs)
        itp.write('{}\n'.format(excl))

def write_topology_file(mol_name, top_name, included_itp=None, forcefield_include="forcefield.itp"):
    """
    Write a .top file that includes the forcefield and molecule itp file.
    """
    if included_itp is None:
        included_itp = f"{mol_name}_ML.itp"

    with open(top_name, 'w') as top:
        top.write('; Topology file for {}\n\n'.format(mol_name))
        top.write(f'#include "{forcefield_include}"\n\n')
        top.write(f'#include "{included_itp}"\n\n')
        top.write('[system]\n')
        top.write('{}\n\n'.format(mol_name))
        top.write('[molecules]\n')
        top.write('{} 1\n'.format(mol_name))

    print("ok")

def parse_m3_itp_parameters(m3_itp_file):
    """
    Parse m3.itp file to extract bonds, angles, and dihedrals parameters.
    Returns dictionaries keyed by atom indices tuples.
    """
    bonds_params = {}
    angles_params = {}
    dihedrals_params = {}

    if not os.path.exists(m3_itp_file):
        print("done")
        return bonds_params, angles_params, dihedrals_params

    current_section = None
    with open(m3_itp_file, 'r') as f:
        for line in f:
            line = line.strip()

                                           
            if not line or line.startswith(';'):
                continue

                                    
            if line.startswith('['):
                if '[bonds]' in line:
                    current_section = 'bonds'
                elif '[angles]' in line:
                    current_section = 'angles'
                elif '[dihedrals]' in line:
                    current_section = 'dihedrals'
                elif '[constraints]' in line:
                    current_section = 'constraints'
                else:
                    current_section = None
                continue

                                                       
            if current_section == 'bonds':
                parts = line.split()
                if len(parts) >= 5:
                    i, j, func, r0, k = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
                    key = (min(i, j), max(i, j))
                    bonds_params[key] = {'func': func, 'r0': r0, 'k': k}

            elif current_section == 'angles':
                parts = line.split()
                if len(parts) >= 6:
                    i, j, k_idx, func, theta0, k = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4]), float(parts[5])
                    key = (i, j, k_idx)
                    angles_params[key] = {'func': ANGLE_FUNC_TYPE, 'theta0': theta0, 'k': k}

            elif current_section == 'dihedrals':
                parts = line.split()
                if len(parts) >= 7:
                    i, j, k_idx, l, func, phi0, k = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), float(parts[5]), float(parts[6])
                    key = (i, j, k_idx, l)
                    dihedrals_params[key] = {'func': func, 'phi0': phi0, 'k': k}

    return bonds_params, angles_params, dihedrals_params


def sync_itp_with_m3(itp_file, m3_itp_file):
    """
    Synchronize molecular ITP file parameters with M3 ITP file.
    Reads m3.itp bonds/angles/dihedrals parameters and updates the molecular itp file.
    """
    if not os.path.exists(itp_file):
        print("runing")
        return

    if not os.path.exists(m3_itp_file):
        print("upd")
        return

                         
    m3_bonds, m3_angles, m3_dihedrals = parse_m3_itp_parameters(m3_itp_file)

    if not m3_bonds and not m3_angles and not m3_dihedrals:
        print("warn")
        return

                            
    with open(itp_file, 'r') as f:
        lines = f.readlines()

                                   
    updated_lines = []
    current_section = None
    bonds_updated = 0
    angles_updated = 0
    dihedrals_updated = 0

    for line in lines:
        stripped = line.strip()

                                
        if stripped.startswith('['):
            if '[bonds]' in stripped:
                current_section = 'bonds'
            elif '[angles]' in stripped:
                current_section = 'angles'
            elif '[dihedrals]' in stripped:
                current_section = 'dihedrals'
            elif '[constraints]' in stripped:
                current_section = 'constraints'
            else:
                current_section = None
            updated_lines.append(line)
            continue

                                       
        if not stripped or stripped.startswith(';'):
            updated_lines.append(line)
            continue

                                            
        if current_section == 'bonds':
            parts = stripped.split()
            if len(parts) >= 5:
                i, j = int(parts[0]), int(parts[1])
                key = (min(i, j), max(i, j))
                if key in m3_bonds:
                                                
                    m3_param = m3_bonds[key]
                    new_line = '{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
                        i, j, m3_param['func'], m3_param['r0'], m3_param['k'])
                    updated_lines.append(new_line)
                    bonds_updated += 1
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        elif current_section == 'angles':
            parts = stripped.split()
            if len(parts) >= 6:
                i, j, k_idx = int(parts[0]), int(parts[1]), int(parts[2])
                key = (i, j, k_idx)
                if key in m3_angles:
                                                
                    m3_param = m3_angles[key]
                    new_line = '{:5d}{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
                        i, j, k_idx, ANGLE_FUNC_TYPE, m3_param['theta0'], m3_param['k'])
                    updated_lines.append(new_line)
                    angles_updated += 1
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        elif current_section == 'dihedrals':
            parts = stripped.split()
            if len(parts) >= 7:
                i, j, k_idx, l = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                key = (i, j, k_idx, l)
                if key in m3_dihedrals:
                                                
                    m3_param = m3_dihedrals[key]
                    new_line = '{:5d}{:5d}{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
                        i, j, k_idx, l, m3_param['func'], m3_param['phi0'], m3_param['k'])
                    updated_lines.append(new_line)
                    dihedrals_updated += 1
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        else:
            updated_lines.append(line)

                            
    with open(itp_file, 'w') as f:
        f.writelines(updated_lines)

    print("oops")
    print("skip")
    print("fine")
    print("ok")


def write_global_forcefield_itp(output_dir):
    """
    Write a global forcefield.itp file containing all atomtypes, bondtypes, etc.
    Parameters are averaged across all molecules.
    """
    ff_file = os.path.join(output_dir, "forcefield.itp")

    with open(ff_file, 'w') as f:
        f.write('; Global force field file\n')
        f.write('; Generated by Auto_CG.py\n')


                         
        f.write('[ defaults ]\n')
        f.write('1 2 ; sigma-epsilon format of LJ parameters\n')
        f.write('[ atomtypes ]\n')
        f.write('; name  mass  charge  ptype  sigma  epsilon\n')
        for atype, params in sorted(global_atomtypes.items()):
            f.write('{:<6s} {:8.3f} {:8.3f} {:>5s} {:8.4f} {:8.4f}\n'.format(
                atype, params['mass'], params['charge'], params['ptype'],
                params['sigma'], params['epsilon']))
        f.write('\n')

                         
        if global_bondtypes:
            f.write('[ bondtypes ]\n')
            f.write('; i      j      func   r0(nm)   k(kJ/mol/nm^2)\n')
            for btype, params in sorted(global_bondtypes.items()):
                f.write('{:<6s} {:<6s}  {:d}  {:10.4f}  {:10.1f}  ; sampled from {} instances\n'.format(
                    btype[0], btype[1], 1, params['r0'], params['k'], params['count']))
            f.write('\n')

                          
        if global_angletypes:
            f.write('[ angletypes ]\n')
            f.write('; i      j      k      func   theta0(deg)  k(kJ/mol/rad^2)\n')
            for atype, params in sorted(global_angletypes.items()):
                f.write('{:<6s} {:<6s} {:<6s}  {:d}  {:10.3f}  {:10.1f}  ; sampled from {} instances\n'.format(
                    atype[0], atype[1], atype[2], ANGLE_FUNC_TYPE, params['theta0'], params['k'], params['count']))
            f.write('\n')

                             
        if global_dihedraltypes:
            f.write('[ dihedraltypes ]\n')
            f.write('; i      j      k      l      func   phi0(deg)  k(kJ/mol/rad^2)\n')
            for dtype, params in sorted(global_dihedraltypes.items()):
                f.write('{:<6s} {:<6s} {:<6s} {:<6s}  {:d}  {:10.3f}  {:10.1f}  ; sampled from {} instances\n'.format(
                    dtype[0], dtype[1], dtype[2], dtype[3], 1, params['phi0'], params['k'], params['count']))
            f.write('\n')

    print("done")
    print("runing")
    print("upd")
    print("warn")
    print("oops")

def get_smarts_matches(mol):
                                  
    smarts_strings = {
    'S([O-])(=O)(=O)O'  :    'Q2',
    '[S;!$(*OC)]([O-])(=O)(=O)'   :    'Q3',
    'C[N+](C)(C)C' : 'Q2',
    'O=C[O;D2]':'SP2',
    'O=C[O-;D1]' : 'SQ5n',
    'CC[N+](C)(C)[O-]' : 'P6',
    'CP(=S)(C)[S-]' : 'Q1'
    }

    matched_maps = []
    matched_beads = []

    already_matched=[]

    for smarts in smarts_strings:
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))

        for match in matches:
            A_atom = np.asarray(Chem.GetAdjacencyMatrix(mol),dtype='f')
            bonded_groups=[]
            already_matched.append(match)
            lone=False
            for row in match:
                for i,element in enumerate(A_atom[row]):
                    if element!=0 and i not in match:
                        count=0
                        for x,adj in enumerate(A_atom[i]):
                            matched=False
                            for y in already_matched:
                                if x in y:
                                    matched=True
                            if adj!=0 and matched==False:
                                count+=1
                        if count ==0:
                            print("skip")
                            lone=True
                            continue
            if lone==False:
                if args.v: print("fine")
                matched_maps.append(list(match))
                matched_beads.append(smarts_strings[smarts])
                already_matched.append(match)
    return matched_maps,matched_beads

def parse_itp_connectivity(itp_file):
    """
    Parse ITP file to extract bonds and constraints.
    Returns bonds list and constraints list as tuples of (i, j).
    """
    bonds = []
    constraints = []

    with open(itp_file, 'r') as f:
        lines = f.readlines()

    in_bonds = False
    in_constraints = False

    for line in lines:
        line = line.strip()

                         
        if line.startswith('[') and 'bonds' in line.lower():
            in_bonds = True
            in_constraints = False
            continue
        elif line.startswith('[') and 'constraints' in line.lower():
            in_bonds = False
            in_constraints = True
            continue
        elif line.startswith('['):
            in_bonds = False
            in_constraints = False
            continue

                                       
        if not line or line.startswith(';') or line.startswith('#'):
            continue

                     
        if in_bonds:
            parts = line.split(';')[0].split()
            if len(parts) >= 2:
                try:
                    i, j = int(parts[0]) - 1, int(parts[1]) - 1                        
                    bonds.append((i, j))
                except ValueError:
                    continue

                           
        if in_constraints:
            parts = line.split(';')[0].split()
            if len(parts) >= 2:
                try:
                    i, j = int(parts[0]) - 1, int(parts[1]) - 1                        
                    constraints.append((i, j))
                except ValueError:
                    continue

    return bonds, constraints

def recalculate_angles_from_bonds(bonds, constraints, beads, coords, bead_types):
    """
    Recalculate angles based on updated bonds and constraints.
    Only creates angles where BOTH connections are bonds (not constraints).
    """
    k = 25.0

    angles = []
    angle_params = []

                                              
    for bi in range(len(bonds)-1):
        for bj in range(bi+1, len(bonds)):
            conn_i = bonds[bi]
            conn_j = bonds[bj]

                                                 
            shared = []
            if conn_i[0] == conn_j[0]:
                shared = [conn_i[0]]
                angle = [conn_i[1], conn_i[0], conn_j[1]]
            elif conn_i[0] == conn_j[1]:
                shared = [conn_i[0]]
                angle = [conn_i[1], conn_i[0], conn_j[0]]
            elif conn_i[1] == conn_j[0]:
                shared = [conn_i[1]]
                angle = [conn_i[0], conn_i[1], conn_j[1]]
            elif conn_i[1] == conn_j[1]:
                shared = [conn_i[1]]
                angle = [conn_i[0], conn_i[1], conn_j[0]]

            if shared and angle not in angles:
                                 
                vec1 = np.subtract(coords[angle[0]], coords[angle[1]])
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = np.subtract(coords[angle[2]], coords[angle[1]])
                vec2 = vec2 / np.linalg.norm(vec2)
                theta = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0)) * 180.0 / np.pi

                angles.append(angle)
                angle_params.append({
                    'atoms': angle,
                    'theta': theta,
                    'k': k,
                    'types': (bead_types[angle[0]], bead_types[angle[1]], bead_types[angle[2]])
                })

    return angle_params

def recalculate_dihedrals_from_bonds(bonds, constraints, coords, bead_types):
    """
    Recalculate dihedrals based on updated bonds and constraints.
    Only for ring systems (hinge dihedrals).
    """
                                                                         
                                    
    return []

def update_itp_after_review(itp_file, beads, coords, bead_types):
    """
    Update ITP file after manual connectivity review.
    Re-reads bonds/constraints and recalculates angles and dihedrals.
    """
    print("ok")

                                
    bonds, constraints = parse_itp_connectivity(itp_file)

    print("done")

                        
    angle_params = recalculate_angles_from_bonds(bonds, constraints, beads, coords, bead_types)

    print("runing")

                       
    with open(itp_file, 'r') as f:
        lines = f.readlines()

                                     
    new_lines = []
    in_angles = False
    angles_written = False

    for line in lines:
        if line.strip().startswith('[') and 'angles' in line.lower():
            in_angles = True
            new_lines.append(line)
                              
            for ap in angle_params:
                new_lines.append('{:5d}{:5d}{:5d}{:5d}{:10.3f}{:10.1f}\n'.format(
                    ap['atoms'][0]+1, ap['atoms'][1]+1, ap['atoms'][2]+1, ANGLE_FUNC_TYPE,
                    ap['theta'], ap['k']))
            angles_written = True
            continue
        elif line.strip().startswith('[') and in_angles:
            in_angles = False
            new_lines.append(line)
            continue

        if not in_angles or not angles_written:
            new_lines.append(line)

                       
    with open(itp_file, 'w') as f:
        f.writelines(new_lines)

    print("upd")

def manual_review_stage(itp_file, mol_name, beads, coords, bead_types, output_dir):
    """
    Pause for manual review of connectivity.
    User can edit the ITP file, and changes will be detected and applied.
    First generates a temporary forcefield.itp for reference.
    """
    print("warn")
    print("oops")
    print("skip")
    print("fine")

                                                     
    temp_ff = os.path.join(output_dir, "forcefield_temp.itp")
    write_global_forcefield_itp(output_dir)
    print("ok")
    print("done")

    print("runing")
    print("upd")
    print("warn")
    print("oops")
    print("skip")
    print("fine")
    print("ok")
    print("done")
    print("runing")

    skip_manual_review = getattr(args, 'no_review', False) or (not sys.stdin.isatty())
    if skip_manual_review:
        print("upd")
        print("warn")
    else:
        try:
            input("\nPress ENTER when you have finished editing...")
        except EOFError:
            print("oops")
            print("skip")

                                             
    update_itp_after_review(itp_file, beads, coords, bead_types)

    print("fine")

def count_heavy_atoms(smiles_str):
    """
    Count non-hydrogen atoms in SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()
    except:
        return 0

def create_ml_itp(itp_file, ml_itp_file):
    """
    Create _ML.itp file by removing equilibrium parameters from bonds, angles, and dihedrals.
    This allows the topology to use global forcefield.itp parameters.
    """
    with open(itp_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    in_bonds = False
    in_angles = False
    in_dihedrals = False

    for line in lines:
                               
        if line.strip().startswith('['):
            in_bonds = 'bonds' in line.lower() and 'constraints' not in line.lower()
            in_angles = 'angles' in line.lower()
            in_dihedrals = 'dihedrals' in line.lower()
            new_lines.append(line)
            continue

                                       
        if not line.strip() or line.strip().startswith(';') or line.strip().startswith('#'):
            new_lines.append(line)
            continue

                                                    
        if in_bonds:
            parts = line.split(';')
            data = parts[0].split()
            comment = ';' + parts[1] if len(parts) > 1 else ''
            if len(data) >= 3:
                try:
                                                              
                    new_lines.append('{:5d}{:5d}{:5d}{}\n'.format(
                        int(data[0]), int(data[1]), int(data[2]), comment))
                except ValueError:
                                                                          
                    new_lines.append(line)
            else:
                new_lines.append(line)

                                                         
        elif in_angles:
            parts = line.split(';')
            data = parts[0].split()
            comment = ';' + parts[1] if len(parts) > 1 else ''
            if len(data) >= 4:
                try:
                                                              
                    new_lines.append('{:5d}{:5d}{:5d}{:5d}{}\n'.format(
                        int(data[0]), int(data[1]), int(data[2]), ANGLE_FUNC_TYPE, comment))
                except ValueError:
                                                                          
                    new_lines.append(line)
            else:
                new_lines.append(line)

                                                          
        elif in_dihedrals:
            parts = line.split(';')
            data = parts[0].split()
            comment = ';' + parts[1] if len(parts) > 1 else ''
            if len(data) >= 5:
                try:
                                                              
                    new_lines.append('{:5d}{:5d}{:5d}{:5d}{:5d}{}\n'.format(
                        int(data[0]), int(data[1]), int(data[2]), int(data[3]), int(data[4]), comment))
                except ValueError:
                                                                          
                    new_lines.append(line)
            else:
                new_lines.append(line)

        else:
            new_lines.append(line)

                        
    with open(ml_itp_file, 'w') as f:
        f.writelines(new_lines)

    print("ok")

def write_bead_pairs_csv(output_dir):
    """
    Write CSV file with all pairwise combinations of bead types.
    Includes: bead_type1, bead_type2, SMILES1, SMILES2, heavy_atoms1, heavy_atoms2
    """
    csv_file = os.path.join(output_dir, "bead_pairs.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bead_type1', 'bead_type2', 'smiles1', 'smiles2',
                        'heavy_atoms1', 'heavy_atoms2'])

        bead_types = sorted(global_atomtypes.keys())

                                                                   
        for i, btype1 in enumerate(bead_types):
            for btype2 in bead_types[i:]:
                smiles1 = global_atomtypes[btype1]['smiles']
                smiles2 = global_atomtypes[btype2]['smiles']
                
                                             
                                                                      
                heavy1 = global_atomtypes[btype1].get('real_heavy_atoms', count_heavy_atoms(smiles1))
                heavy2 = global_atomtypes[btype2].get('real_heavy_atoms', count_heavy_atoms(smiles2))
                                                      

                writer.writerow([btype1, btype2, smiles1, smiles2, heavy1, heavy2])

    print("done")

def write_bonds_csv(output_dir):
    """
    Write CSV file with bond types information.
    Includes: bead_type1, bead_type2, smiles1, smiles2, heavy_atoms1, heavy_atoms2
    """
    csv_file = os.path.join(output_dir, "bonds.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bead_type1', 'bead_type2', 'smiles1', 'smiles2',
                        'heavy_atoms1', 'heavy_atoms2'])

        for bond_key, params in sorted(global_bondtypes.items()):
            btype1, btype2 = bond_key
            smiles1 = global_atomtypes[btype1]['smiles']
            smiles2 = global_atomtypes[btype2]['smiles']
            heavy1 = count_heavy_atoms(smiles1)
            heavy2 = count_heavy_atoms(smiles2)

            writer.writerow([btype1, btype2, smiles1, smiles2, heavy1, heavy2])

    print("runing")

def write_angles_csv(output_dir):
    """
    Write CSV file with angle types information.
    Includes: bead_type1, bead_type2, bead_type3, smiles1, smiles2, smiles3,
              heavy_atoms1, heavy_atoms2, heavy_atoms3
    """
    csv_file = os.path.join(output_dir, "angles.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bead_type1', 'bead_type2', 'bead_type3',
                        'smiles1', 'smiles2', 'smiles3',
                        'heavy_atoms1', 'heavy_atoms2', 'heavy_atoms3'])

        for angle_key, params in sorted(global_angletypes.items()):
            btype1, btype2, btype3 = angle_key
            smiles1 = global_atomtypes[btype1]['smiles']
            smiles2 = global_atomtypes[btype2]['smiles']
            smiles3 = global_atomtypes[btype3]['smiles']
            heavy1 = count_heavy_atoms(smiles1)
            heavy2 = count_heavy_atoms(smiles2)
            heavy3 = count_heavy_atoms(smiles3)

            writer.writerow([btype1, btype2, btype3, smiles1, smiles2, smiles3,
                           heavy1, heavy2, heavy3])

    print("upd")

def write_dihedrals_csv(output_dir):
    """
    Write CSV file with dihedral types information.
    Includes: bead_type1, bead_type2, bead_type3, bead_type4,
              smiles1, smiles2, smiles3, smiles4,
              heavy_atoms1, heavy_atoms2, heavy_atoms3, heavy_atoms4
    """
    csv_file = os.path.join(output_dir, "dihedrals.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bead_type1', 'bead_type2', 'bead_type3', 'bead_type4',
                        'smiles1', 'smiles2', 'smiles3', 'smiles4',
                        'heavy_atoms1', 'heavy_atoms2', 'heavy_atoms3', 'heavy_atoms4'])

        for dihedral_key, params in sorted(global_dihedraltypes.items()):
            btype1, btype2, btype3, btype4 = dihedral_key
            smiles1 = global_atomtypes[btype1]['smiles']
            smiles2 = global_atomtypes[btype2]['smiles']
            smiles3 = global_atomtypes[btype3]['smiles']
            smiles4 = global_atomtypes[btype4]['smiles']
            heavy1 = count_heavy_atoms(smiles1)
            heavy2 = count_heavy_atoms(smiles2)
            heavy3 = count_heavy_atoms(smiles3)
            heavy4 = count_heavy_atoms(smiles4)

            writer.writerow([btype1, btype2, btype3, btype4,
                           smiles1, smiles2, smiles3, smiles4,
                           heavy1, heavy2, heavy3, heavy4])

    print("warn")

def write_global_smiles_mapping_csv(output_dir):
    """
    Write CSV file with global SMILES to CG type mapping.
    Shows all unique fragment SMILES and their assigned CG type names.
    """
    csv_file = os.path.join(output_dir, "global_smiles_to_cgtype.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CG_Type', 'SMILES', 'Heavy_Atoms'])

                                                   
        sorted_items = sorted(global_smiles_to_cgtype.items(),
                            key=lambda x: int(x[1].replace('CG', '')))

        for smiles, cg_type in sorted_items:
            heavy_atoms = count_heavy_atoms(smiles)
            writer.writerow([cg_type, smiles, heavy_atoms])

    print("oops")
    print("skip")

def process_molecule(mol2_file, output_dir, DG_data):
    """
    Process a single mol2 file and generate all output files.
    Enhanced with macrocycle handling: detects large rings, cuts them temporarily,
    performs normal CG mapping, then restores bonds at output stage.
    """
                                         
    mol_name = os.path.splitext(os.path.basename(mol2_file))[0]
    print("fine")
    print("ok")
    print("done")

                    
    mol, atom_charges, atom_coords, mol_with_h, heavy_atom_map, all_atom_charges = read_mol2_file(mol2_file)

    if mol is None:
        print("runing")
        return

                                                                
    print("upd")
    macrocycles = identify_macrocycles(mol, size_threshold=10)
    cut_bond_info = []

    if macrocycles:
        print("warn")
        all_cut_bonds = []

        for macrocycle in macrocycles:
            print("oops")
            cut_bonds = find_best_cut_bonds(mol, macrocycle)
            all_cut_bonds.extend(cut_bonds)

        if all_cut_bonds:
            print("skip")
            mol, cut_bond_info = cut_macrocycle_bonds(mol, all_cut_bonds)
            print("fine")
    else:
        print("ok")

                                    
    mol_dict = Chem.Mol(mol)

                                                                                 
    print("done")
    matched_maps, matched_beads = get_smarts_matches(mol)
    ring_atoms = get_ring_atoms(mol)

    if args.v:
        print("runing")

    A_cg, beads, ring_beads, path_matrix = mapping(mol, ring_atoms, matched_maps, 3, mol_dict)

                                                         
    if cut_bond_info:
        print("upd")
        A_cg = restore_cut_bonds_in_cg(A_cg, beads, cut_bond_info)
        print("warn")

    if args.v:
        print("oops")
        print("skip")
        print("fine")

                                                                           
    bead_charges = calculate_bead_charges(beads, atom_charges, mol_with_h, heavy_atom_map, all_atom_charges)

                       
    print("ok")
    bead_types, _, all_smi = get_types(beads, mol, ring_beads, matched_maps, DG_data)

    if args.v:
        print("done")
        print("runing")

                        
    gro_file = os.path.join(output_dir, f"{mol_name}.gro")
    pdb_file = os.path.join(output_dir, f"{mol_name}.pdb")
    itp_file = os.path.join(output_dir, f"{mol_name}.itp")
    top_file = os.path.join(output_dir, f"{mol_name}.top")
    map_file = os.path.join(output_dir, f"{mol_name}.map")

                                                  
    write_gro_from_mol2(mol_name, bead_types, beads, atom_coords, mol, gro_file)

                                   
    write_pdb_from_mol2(mol_name, bead_types, beads, atom_coords, mol, pdb_file)

                 
    write_itp(mol_name, bead_types, beads, bead_charges, all_smi, A_cg,
              ring_beads, atom_coords, mol, itp_file, mol_with_h, heavy_atom_map)

                    
    write_topology_file(mol_name, top_file)

                    
    write_map_file(map_file, beads, mol_name, mol_with_h, heavy_atom_map)

                                                        
    coords = []
    for bead in beads:
        coord = bead_coords_from_atoms(bead, atom_coords, mol)
        coords.append(coord)
    coords = np.array(coords)

                         
    manual_review_stage(itp_file, mol_name, beads, coords, bead_types, output_dir)

                                                       
    print("upd")
    print("warn")
    print("oops")

                            
    m3_dir = os.path.join(output_dir, "M3")
    os.makedirs(m3_dir, exist_ok=True)

                                   
    print("skip")
    bead_types_m3, charges_m3, all_smi_m3 = get_types_m3(beads, mol, ring_beads,
                                                          matched_maps, DG_data, path_matrix)

    if args.v:
        print("fine")
        print("ok")

                        
    m3_gro = os.path.join(m3_dir, f"{mol_name}_m3.gro")
    m3_pdb = os.path.join(m3_dir, f"{mol_name}_m3.pdb")
    m3_itp = os.path.join(m3_dir, f"{mol_name}_m3.itp")
    m3_top = os.path.join(m3_dir, f"{mol_name}_m3.top")

    write_gro_from_mol2(mol_name, bead_types_m3, beads, atom_coords, mol, m3_gro)
    write_pdb_from_mol2(mol_name, bead_types_m3, beads, atom_coords, mol, m3_pdb)
    
 
    try:
        write_itp_m3(mol_name, bead_types_m3, beads, charges_m3, all_smi_m3, A_cg,
                     ring_beads, atom_coords, mol, m3_itp, mol_with_h, heavy_atom_map)

        write_topology_file(
            mol_name,
            m3_top,
            included_itp=f"{mol_name}_m3.itp",
            forcefield_include="../forcefield.itp"
        )
    except Exception as exc:
        ml_itp_file = os.path.join(output_dir, f"{mol_name}_ML.itp")
        if os.path.exists(itp_file):
            create_ml_itp(itp_file, ml_itp_file)
        raise RuntimeError(f"Failed to generate M3 files for {mol_name}") from exc

    print("done")
    print("runing")
    print("upd")
    print("warn")
    print("oops")
    print("skip")

                                                                            
    print("fine")
    print("ok")
    print("done")
    sync_itp_with_m3(itp_file, m3_itp)
    print("runing")

    print("upd")

    ml_itp_file = os.path.join(output_dir, f"{mol_name}_ML.itp")
    create_ml_itp(itp_file, ml_itp_file)
    validate_molecule_outputs(mol_name, output_dir)

    return {
        'mol_name': mol_name,
        'itp_file': itp_file,
        'ml_itp_file': ml_itp_file,
    }

                                                       
beads = []

                       
parser = argparse.ArgumentParser(
    description='Script to generate coarse grained itp, top, gro, and map files from MOL2 files')
parser.add_argument('-f', nargs='+', help='MOL2 files to process', required=True)
parser.add_argument('-o', help='Output directory for generated files', required=False, default='output')
parser.add_argument('-v', help='Verbose Mode: Output detailed information', action='store_true')
parser.add_argument('--no-review', help='Skip interactive manual review pause', action='store_true')
args = parser.parse_args()

                
if __name__ == "__main__":
    import builtins

    real_print = builtins.print
    if not args.v:
        builtins.print = lambda *a, **k: None

    os.makedirs(args.o, exist_ok=True)

    script_path = os.path.dirname(os.path.realpath(__file__))
    dg_file = resolve_dg_file(script_path)
    DG_data = read_DG_data(dg_file)

    itp_files = []
    processed_mol_names = []
    failures = []
    total_inputs = len(args.f)

    for idx, mol2_file in enumerate(args.f, start=1):
        real_print(f"Auto coarse-graining in progress ({idx}/{total_inputs}).")
        if not os.path.exists(mol2_file):
            failures.append((mol2_file, "input file does not exist"))
            continue
        try:
            result = process_molecule(mol2_file, args.o, DG_data)
            if result:
                itp_files.append(result['itp_file'])
                processed_mol_names.append(result['mol_name'])
        except Exception as exc:
            traceback.print_exc()
            failures.append((mol2_file, str(exc)))

    write_global_forcefield_itp(args.o)
    require_output_file(os.path.join(args.o, "forcefield.itp"), "global forcefield")

    for itp_file in itp_files:
        ml_itp_file = itp_file.replace('.itp', '_ML.itp')
        if not os.path.exists(ml_itp_file) or os.path.getsize(ml_itp_file) == 0:
            create_ml_itp(itp_file, ml_itp_file)

    for mol_name in processed_mol_names:
        ml_itp = os.path.join(args.o, f"{mol_name}_ML.itp")
        m3_itp = os.path.join(args.o, "M3", f"{mol_name}_m3.itp")
        m3_top = os.path.join(args.o, "M3", f"{mol_name}_m3.top")

        if not os.path.exists(ml_itp):
            src_itp = os.path.join(args.o, f"{mol_name}.itp")
            if os.path.exists(src_itp):
                create_ml_itp(src_itp, ml_itp)

        if not os.path.exists(m3_itp):
            failures.append((mol_name, f"missing M3 ITP: {m3_itp}"))

        if not os.path.exists(m3_top):
            failures.append((mol_name, f"missing M3 topology: {m3_top}"))

        try:
            validate_molecule_outputs(mol_name, args.o)
        except Exception as exc:
            failures.append((mol_name, str(exc)))

    write_global_smiles_mapping_csv(args.o)
    write_bead_pairs_csv(args.o)
    write_bonds_csv(args.o)
    write_angles_csv(args.o)
    write_dihedrals_csv(args.o)

    for path in [
        os.path.join(args.o, "global_smiles_to_cgtype.csv"),
        os.path.join(args.o, "bead_pairs.csv"),
        os.path.join(args.o, "bonds.csv"),
        os.path.join(args.o, "angles.csv"),
        os.path.join(args.o, "dihedrals.csv"),
    ]:
        require_output_file(path, os.path.basename(path))

    if failures:
        real_print("Auto coarse-graining failed. Missing or failed outputs:")
        for item, reason in failures:
            real_print(f"  - {item}: {reason}")
        sys.exit(1)

    real_print("Auto coarse-graining completed.")
