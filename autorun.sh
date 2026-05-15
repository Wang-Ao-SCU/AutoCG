set -euo pipefail

cd "$(dirname "$0")"

echo "Processing complete topology file..."

{
mkdir -p output/nocharge

python scripts/2_G_pre.py
python scripts/3_pair_pre.py

python scripts/4_append_pairs.py --csv ./pair_predict_results/GBR_predictions.csv \
  --input-itp ./output/forcefield.itp \
  --output-itp all_molecules_ff_with_nonbond.itp
cp all_molecules_ff_with_nonbond.itp ./output/forcefield.itp

python scripts/5.py
cp ./output/forcefield_new.itp ./output/forcefield.itp
python scripts/5_1.py

python scripts/5_2_gen_w_csv.py
python scripts/5_3_nocharge.py
python scripts/6_W_PRE.py
python scripts/6_W_append.py
python scripts/6_ION_PRE.py
python scripts/6_ION_append.py

python scripts/7_dihedral.py
cp ./output/forcefield_new.itp ./output/forcefield.itp
cp ./output/forcefield_new.itp ./output/nocharge/forcefield.itp
python scripts/7_benz.py
cp ./output/forcefield.itp ./output/nocharge/forcefield.itp

python scripts/ML4_bond_pre.py
python scripts/ML4_angle_pre.py
python scripts/ML4_update.py
cp ./output/forcefield_ML.itp ./output/nocharge/forcefield_ML.itp

python scripts/7_charge_neu.py
python scripts/8_fix_constraint.py
python scripts/9_m3_ml_itp.py


# Optional solvent tuning: apply empirical Water/Dioxane force-field correction.
# python scripts/10_water_dioxane_forcefield_experience.py --input-output-dir ./output --output-dir ./output_water_dioxane_experience --preset v2 --force

find output -type f -name "*.bak" -delete
rm -f output/forcefield_new.itp output/nocharge/forcefield_new.itp
rm -f **.itp 
rm -f **.csv
} >/dev/null 2>&1

echo "Complete topology generated."
