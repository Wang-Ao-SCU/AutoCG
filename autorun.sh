set -euo pipefail

cd "$(dirname "$0")"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-autocg}"

mkdir -p output/nocharge output/logs
LOG_FILE="output/logs/autorun.log"
: > "$LOG_FILE"

run() {
  echo ">>> $*" | tee -a "$LOG_FILE"
  if "$@" >>"$LOG_FILE" 2>&1; then
    return 0
  else
    status=$?
    echo "ERROR: step failed with exit code $status: $*" >&2
    echo "Last 80 lines from $LOG_FILE:" >&2
    tail -n 80 "$LOG_FILE" >&2
    exit "$status"
  fi
}

echo "Processing complete topology file..."
echo "Detailed log: $LOG_FILE"

run python scripts/2_G_pre.py
run python scripts/3_pair_pre.py

run python scripts/4_append_pairs.py --csv ./pair_predict_results/GBR_predictions.csv \
  --input-itp ./output/forcefield.itp \
  --output-itp all_molecules_ff_with_nonbond.itp
run cp all_molecules_ff_with_nonbond.itp ./output/forcefield.itp

run python scripts/5.py
run cp ./output/forcefield_new.itp ./output/forcefield.itp
run python scripts/5_1.py

run python scripts/5_2_gen_w_csv.py
run python scripts/5_3_nocharge.py
run python scripts/6_W_PRE.py
run python scripts/6_W_append.py
run python scripts/6_ION_PRE.py
run python scripts/6_ION_append.py

run python scripts/7_dihedral.py
run cp ./output/forcefield_new.itp ./output/forcefield.itp
run cp ./output/forcefield_new.itp ./output/nocharge/forcefield.itp
run python scripts/7_benz.py
run cp ./output/forcefield.itp ./output/nocharge/forcefield.itp

run python scripts/ML4_bond_pre.py
run python scripts/ML4_angle_pre.py
run python scripts/ML4_update.py
run cp ./output/forcefield_ML.itp ./output/nocharge/forcefield_ML.itp

run python scripts/7_charge_neu.py
run python scripts/8_fix_constraint.py
run python scripts/9_m3_ml_itp.py
run python scripts/10_m3_inline_ml_bonded_params.py

# Optional solvent tuning: apply empirical Water/Dioxane force-field correction.
# run python scripts/10_water_dioxane_forcefield_experience.py --input-output-dir ./output --output-dir ./output_water_dioxane_experience --preset v2 --force

run find output -type f -name "*.bak" -delete
run rm -f output/forcefield_new.itp output/nocharge/forcefield_new.itp all_molecules_ff_with_nonbond.itp

echo "Complete topology generated."
