import pandas as pd
import os
import sys
import numpy as np
try:
    from ML4_bond import DualTargetModelWrapper, get_pair_enhanced_features
except ImportError:
    sys.exit(1)

def main():
    INPUT_CSV = './output/bonds.csv'
    OUTPUT_CSV = './output/bonds_with_predictions.csv'
    MODEL_DIR = './ML4/bond_prediction_results/models'
    MODEL_TYPE = 'gbr'
    if not os.path.exists(INPUT_CSV):
        return
    model_r_path = os.path.join(MODEL_DIR, f'{MODEL_TYPE}_model_r.pkl')
    model_k_path = os.path.join(MODEL_DIR, f'{MODEL_TYPE}_model_k.pkl')
    if not os.path.exists(model_r_path) or not os.path.exists(model_k_path):
        return
    df = pd.read_csv(INPUT_CSV)
    required_cols = ['smiles1', 'smiles2']
    if not all((col in df.columns for col in required_cols)):
        return
    smiles_a_list = df['smiles1'].tolist()
    smiles_b_list = df['smiles2'].tolist()
    try:
        X_features, _ = get_pair_enhanced_features(smiles_a_list, smiles_b_list)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
    try:
        wrapper = DualTargetModelWrapper(base_model_type=MODEL_TYPE)
        wrapper.load_models(MODEL_DIR)
    except Exception as e:
        return
    predictions = wrapper.predict(X_features)
    df['predicted_r'] = predictions[:, 0]
    df['predicted_k'] = predictions[:, 1]
    df.to_csv(OUTPUT_CSV, index=False)
if __name__ == '__main__':
    main()
