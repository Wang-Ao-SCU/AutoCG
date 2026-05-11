# AutoCG-ML

ML-based auto coarse-graining workflow for MOL2 inputs.

## Quick Start

```bash
conda env create -f Environment.yml
conda activate autocg_ml
python AutoCG.py -f 1_1.mol2 2_2.mol2 --no-review
bash autorun.sh
```

## Main Outputs

- `output/*.itp`
- `output/*.top`
- `output/*.gro`
- `output/*.pdb`
- `output/*.map`
- `output/forcefield.itp`
- `output/nocharge/*`

## Repo Notes

- `AutoCG.py`: mapping + topology generation.
- `scripts/`: ML prediction and force-field assembly steps.
- `autorun.sh`: end-to-end post-processing pipeline.
