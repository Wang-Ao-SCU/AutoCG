import pandas as pd
import numpy as np
from ML_Ghex import SolvationEnergyPredictor

def predict_smiles_pairs(csv_path, save_path='../pair_predictions', model_dir='../hex/models'):
    import os
    os.makedirs(save_path, exist_ok=True)
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['smiles1', 'smiles2']
        if not all((col in df.columns for col in required_columns)):
            raise ValueError(f'CSV: {required_columns}')
    except Exception as e:
        return
    try:
        predictor = SolvationEnergyPredictor(save_path=save_path)
        predictor.models_dir = model_dir
        predictor.load_all_models()
    except Exception as e:
        return
    model_names = ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']
    for model_name in model_names:
        results_df = df.copy()
        preds_1 = []
        for smiles in df['smiles1']:
            try:
                pred = predictor.predict_single_smiles(smiles, model_name)
                preds_1.append(pred)
            except Exception as e:
                preds_1.append(np.nan)
        preds_2 = []
        for smiles in df['smiles2']:
            try:
                pred = predictor.predict_single_smiles(smiles, model_name)
                preds_2.append(pred)
            except Exception as e:
                preds_2.append(np.nan)
        results_df[f'G1'] = preds_1
        results_df[f'G2'] = preds_2
        output_path = os.path.join(save_path, f'{model_name}_predictions.csv')
        results_df.to_csv(output_path, index=False)
if __name__ == '__main__':
    INPUT_CSV_PATH = './output/bead_pairs.csv'
    OUTPUT_SAVE_PATH = './G_predictions'
    TRAINED_MODEL_DIR = './ML1/hex/models'
    predict_smiles_pairs(INPUT_CSV_PATH, OUTPUT_SAVE_PATH, TRAINED_MODEL_DIR)
