# AutoCG-ML: Automated Machine Learning-Assisted Coarse-Graining Protocol

**AutoCG-ML** is an integrated workflow designed to automate the parameterization of Coarse-Grained (CG) models for molecular dynamics simulations. It combines spectral clustering for bead mapping with machine learning models to predict non-bonded interaction parameters ($\sigma, \epsilon$) based on free energies.

The protocol generates fully formatted GROMACS topology files (`.itp`, `.top`, `.gro`) and supports complex features such as macrocycle handling and manual connectivity reviews.

## Key Features

* **Automated Mapping:** Uses spectral graph clustering to group atoms into beads automatically.
* **Macrocycle Support:** Intelligent detection and handling of large ring systems (e.g., temporarily cutting bonds for mapping and restoring them for topology).
* **ML-Based Parameterization:** * Predicts Solvation Free Energy ($G$) from chemical structures.
    * Derives Lennard-Jones parameters ($\sigma, \epsilon$) for bead pairs using trained ML models (GBR, KNN, etc.).

* **Martini 3.0 Compatibility:** Outputs force field files are additionally compatible with the Martini 3.0 framework in ./output/M3

## Prerequisites
Proper mol2 files generated with Chemdraw or Avogardo are recomended.


```bash
conda env create -f Environment.yml
conda activate autocg_ml
python AutoCG.py -f 1_1.mol2 2_2.mol2 3_3.mol2 --no-review # e.g.
bash autorun.sh
# You can see all your output files at ./output. And we do not recommend to modify the preset output path for the scripts in autorun.sh were pre-referred to.
