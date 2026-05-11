import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings
warnings.filterwarnings('ignore')

def get_single_morgan_fingerprint(smiles, radius=2, nBits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nBits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp)
    except Exception as e:
        return np.zeros(nBits)

def get_single_maccs_key(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    except Exception as e:
        return np.zeros(167)

def get_single_descriptors(smiles, desc_names):
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(desc_names))
        desc_vals = calculator.CalcDescriptors(mol)
        return np.array([0 if np.isnan(x) else x for x in desc_vals])
    except Exception as e:
        return np.zeros(len(desc_names))

def get_triple_enhanced_features(smilesa_list, smilesb_list, smilesc_list):
    desc_names = [desc[0] for desc in Descriptors._descList]
    features_a, features_b, features_c = ([], [], [])
    for smilesa, smilesb, smilesc in zip(smilesa_list, smilesb_list, smilesc_list):
        morgan_a = get_single_morgan_fingerprint(smilesa)
        maccs_a = get_single_maccs_key(smilesa)
        desc_a = get_single_descriptors(smilesa, desc_names)
        feat_a = np.hstack([morgan_a, maccs_a, desc_a])
        morgan_b = get_single_morgan_fingerprint(smilesb)
        maccs_b = get_single_maccs_key(smilesb)
        desc_b = get_single_descriptors(smilesb, desc_names)
        feat_b = np.hstack([morgan_b, maccs_b, desc_b])
        morgan_c = get_single_morgan_fingerprint(smilesc)
        maccs_c = get_single_maccs_key(smilesc)
        desc_c = get_single_descriptors(smilesc, desc_names)
        feat_c = np.hstack([morgan_c, maccs_c, desc_c])
        features_a.append(feat_a)
        features_b.append(feat_b)
        features_c.append(feat_c)
    features_a = np.array(features_a)
    features_b = np.array(features_b)
    features_c = np.array(features_c)
    features_ab = np.abs(features_a - features_b)
    features_ac = np.abs(features_a - features_c)
    features_bc = np.abs(features_b - features_c)
    triple_features = np.hstack([features_a, features_b, features_c, features_ab, features_ac, features_bc])
    return triple_features

class SingleTargetModel:

    def __init__(self, target_name, model_type):
        self.target_name = target_name
        self.model_type = model_type
        self.model = None
        self.best_params = None

    def load(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'modelfile不存在: {model_path}')
        data = joblib.load(model_path)
        if data['target_name'] != self.target_name or data['model_type'] != self.model_type:
            raise ValueError(f'model不匹配: 期望 ({self.model_type}, {self.target_name}), actual ({data['model_type']}, {data['target_name']})')
        self.model = data['model']
        self.best_params = data['best_params']

    def predict(self, X):
        if self.model is None:
            raise RuntimeError(f'Please call load() first to load {self.model_type}_{self.target_name} model')
        X = np.nan_to_num(X)
        return self.model.predict(X)

class AngleParameterPredictor:

    def __init__(self, model_dir):
        self.model_dir = model_dir
        if not os.path.isdir(model_dir):
            raise NotADirectoryError(f'Model directory does not exist: {model_dir}')
        self.loaded_models = {}

    def load_models(self, model_types=['ridge', 'gbr']):
        for model_type in model_types:
            if model_type not in ['ridge', 'gbr']:
                continue
            model_r_path = os.path.join(self.model_dir, f'{model_type}_model_r.pkl')
            model_k_path = os.path.join(self.model_dir, f'{model_type}_model_k.pkl')
            if not os.path.exists(model_r_path) or not os.path.exists(model_k_path):
                continue
            model_r = SingleTargetModel(target_name='r', model_type=model_type)
            model_r.load(model_r_path)
            model_k = SingleTargetModel(target_name='k', model_type=model_type)
            model_k.load(model_k_path)
            self.loaded_models[model_type] = {'r': model_r, 'k': model_k}
        if not self.loaded_models:
            raise RuntimeError('No valid models were loaded. Check the model directory and files完整性')

    def predict_batch(self, input_df):
        required_cols = ['smiles1', 'smiles2', 'smiles3']
        if not all((col in input_df.columns for col in required_cols)):
            raise ValueError(f'The input CSV must contain three SMILES columns: {required_cols}')
        valid_mask = []
        for idx, row in input_df.iterrows():
            smilesa, smilesb, smilesc = (row['smiles1'], row['smiles2'], row['smiles3'])
            try:
                mol_a = Chem.MolFromSmiles(smilesa)
                mol_b = Chem.MolFromSmiles(smilesb)
                mol_c = Chem.MolFromSmiles(smilesc)
                valid = mol_a is not None and mol_b is not None and (mol_c is not None)
            except Exception as e:
                valid = False
            valid_mask.append(valid)
            if not valid:
                pass
        valid_df = input_df[valid_mask].copy()
        if len(valid_df) == 0:
            raise ValueError('No valid SMILES triples were found in the input data; prediction cannot proceed')
        X_features = get_triple_enhanced_features(smilesa_list=valid_df['smiles1'].tolist(), smilesb_list=valid_df['smiles2'].tolist(), smilesc_list=valid_df['smiles3'].tolist())
        for model_type, models in self.loaded_models.items():
            pred_r = models['r'].predict(X_features)
            pred_k = models['k'].predict(X_features)
            valid_df[f'{model_type}_theta0_pred'] = np.round(pred_r, 6)
            valid_df[f'{model_type}_cth_pred'] = np.round(pred_k, 6)
        result_df = input_df.copy()
        for model_type in self.loaded_models.keys():
            result_df[f'{model_type}_theta0_pred'] = np.nan
            result_df[f'{model_type}_cth_pred'] = np.nan
        result_df.loc[valid_mask, [f'{model_type}_theta0_pred' for model_type in self.loaded_models.keys()]] = valid_df[[f'{model_type}_theta0_pred' for model_type in self.loaded_models.keys()]].values
        result_df.loc[valid_mask, [f'{model_type}_cth_pred' for model_type in self.loaded_models.keys()]] = valid_df[[f'{model_type}_cth_pred' for model_type in self.loaded_models.keys()]].values
        return result_df

def main():
    INPUT_CSV_PATH = './output/angles.csv'
    OUTPUT_CSV_PATH = './all_angletypes_with_predictions.csv'
    MODEL_DIR = './ML4/angle_prediction_results/models'
    MODEL_TYPES = ['ridge', 'gbr']
    try:
        if not os.path.exists(INPUT_CSV_PATH):
            raise FileNotFoundError(f'inputfile不存在: {INPUT_CSV_PATH}')
        input_df = pd.read_csv(INPUT_CSV_PATH)
        predictor = AngleParameterPredictor(model_dir=MODEL_DIR)
        predictor.load_models(model_types=MODEL_TYPES)
        result_df = predictor.predict_batch(input_df)
        result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        valid_result_df = result_df.dropna(subset=[f'{model_type}_theta0_pred' for model_type in MODEL_TYPES])
        if len(valid_result_df) > 0:
            preview_cols = list(input_df.columns) + [f'{model_type}_{target}_pred' for model_type in MODEL_TYPES for target in ['r', 'k']]
    except Exception as e:
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()
