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

def get_quartet_enhanced_features(A_list, B_list, C_list, D_list):
    desc_names = [desc[0] for desc in Descriptors._descList]
    features_list = []
    for s1, s2, s3, s4 in zip(A_list, B_list, C_list, D_list):
        morgan1 = get_single_morgan_fingerprint(s1)
        maccs1 = get_single_maccs_key(s1)
        desc1 = get_single_descriptors(s1, desc_names)
        feat1 = np.hstack([morgan1, maccs1, desc1])
        morgan2 = get_single_morgan_fingerprint(s2)
        maccs2 = get_single_maccs_key(s2)
        desc2 = get_single_descriptors(s2, desc_names)
        feat2 = np.hstack([morgan2, maccs2, desc2])
        morgan3 = get_single_morgan_fingerprint(s3)
        maccs3 = get_single_maccs_key(s3)
        desc3 = get_single_descriptors(s3, desc_names)
        feat3 = np.hstack([morgan3, maccs3, desc3])
        morgan4 = get_single_morgan_fingerprint(s4)
        maccs4 = get_single_maccs_key(s4)
        desc4 = get_single_descriptors(s4, desc_names)
        feat4 = np.hstack([morgan4, maccs4, desc4])
        diff12 = np.abs(feat1 - feat2)
        diff23 = np.abs(feat2 - feat3)
        diff34 = np.abs(feat3 - feat4)
        diff13 = np.abs(feat1 - feat3)
        diff24 = np.abs(feat2 - feat4)
        quartet_features = np.hstack([feat1, feat2, feat3, feat4, diff12, diff23, diff34, diff13, diff24])
        features_list.append(quartet_features)
    features_array = np.array(features_list)
    return features_array

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
            raise ValueError(
                f"model不匹配: 期望 ({self.model_type}, {self.target_name}), "
                f"actual ({data['model_type']}, {data['target_name']})"
            )
        self.model = data['model']
        self.best_params = data['best_params']

    def predict(self, X):
        if self.model is None:
            raise RuntimeError(f'Please call load() first to load {self.model_type}_{self.target_name} model')
        X = np.nan_to_num(X)
        return self.model.predict(X)

class DihedralParameterPredictor:

    def __init__(self, model_dir):
        self.model_dir = model_dir
        if not os.path.isdir(model_dir):
            raise NotADirectoryError(f'Model directory does not exist: {model_dir}')
        self.loaded_models = {}

    def load_models(self, model_types=['ridge', 'gbr']):
        for model_type in model_types:
            if model_type not in ['ridge', 'gbr']:
                continue
            model_phi0_path = os.path.join(self.model_dir, f'{model_type}_model_phi0.pkl')
            model_cp_path = os.path.join(self.model_dir, f'{model_type}_model_cp.pkl')
            if not os.path.exists(model_phi0_path) or not os.path.exists(model_cp_path):
                continue
            model_phi0 = SingleTargetModel(target_name='phi0', model_type=model_type)
            model_phi0.load(model_phi0_path)
            model_cp = SingleTargetModel(target_name='cp', model_type=model_type)
            model_cp.load(model_cp_path)
            self.loaded_models[model_type] = {'phi0': model_phi0, 'cp': model_cp}
        if not self.loaded_models:
            raise RuntimeError('No valid models were loaded. Check the model directory and files完整性')

    def predict_batch(self, input_df):
        required_cols = ['smiles1', 'smiles2', 'smiles3', 'smiles4']
        if not all((col in input_df.columns for col in required_cols)):
            raise ValueError(f'inputCSV必须包含四列SMILES: {required_cols}')
        valid_mask = []
        for idx, row in input_df.iterrows():
            smilesa, smilesb, smilesc, smilesd = (row['smiles1'], row['smiles2'], row['smiles3'], row['smiles4'])
            try:
                mol_a = Chem.MolFromSmiles(smilesa)
                mol_b = Chem.MolFromSmiles(smilesb)
                mol_c = Chem.MolFromSmiles(smilesc)
                mol_d = Chem.MolFromSmiles(smilesd)
                valid = mol_a is not None and mol_b is not None and (mol_c is not None) and (mol_d is not None)
            except Exception as e:
                valid = False
            valid_mask.append(valid)
            if not valid:
                pass
        valid_df = input_df[valid_mask].copy()
        if len(valid_df) == 0:
            raise ValueError('No valid SMILES quadruplets were found in the input data; prediction cannot proceed')
        X_features = get_quartet_enhanced_features(A_list=valid_df['smiles1'].tolist(), B_list=valid_df['smiles2'].tolist(), C_list=valid_df['smiles3'].tolist(), D_list=valid_df['smiles4'].tolist())
        for model_type, models in self.loaded_models.items():
            pred_phi0 = models['phi0'].predict(X_features)
            pred_cp = models['cp'].predict(X_features)
            valid_df[f'{model_type}_phi0_pred'] = np.round(pred_phi0, 6)
            valid_df[f'{model_type}_cp_pred'] = np.round(pred_cp, 6)
        result_df = input_df.copy()
        for model_type in self.loaded_models.keys():
            result_df[f'{model_type}_phi0_pred'] = np.nan
            result_df[f'{model_type}_cp_pred'] = np.nan
        result_df.loc[valid_mask, [f'{model_type}_phi0_pred' for model_type in self.loaded_models.keys()]] = valid_df[[f'{model_type}_phi0_pred' for model_type in self.loaded_models.keys()]].values
        result_df.loc[valid_mask, [f'{model_type}_cp_pred' for model_type in self.loaded_models.keys()]] = valid_df[[f'{model_type}_cp_pred' for model_type in self.loaded_models.keys()]].values
        return result_df

def main():
    INPUT_CSV_PATH = './output/dihedrals.csv'
    OUTPUT_CSV_PATH = './all_dihedraltypes_with_predictions.csv'
    MODEL_DIR = './ML4/dihedral_results/models'
    MODEL_TYPES = ['ridge', 'gbr']
    try:
        if not os.path.exists(INPUT_CSV_PATH):
            raise FileNotFoundError(f'inputfile不存在: {INPUT_CSV_PATH}')
        input_df = pd.read_csv(INPUT_CSV_PATH)
        predictor = DihedralParameterPredictor(model_dir=MODEL_DIR)
        predictor.load_models(model_types=MODEL_TYPES)
        result_df = predictor.predict_batch(input_df)
        result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        valid_result_df = result_df.dropna(subset=[f'{model_type}_phi0_pred' for model_type in MODEL_TYPES])
        if len(valid_result_df) > 0:
            preview_cols = list(input_df.columns) + [f'{model_type}_{target}_pred' for model_type in MODEL_TYPES for target in ['phi0', 'cp']]
    except Exception as e:
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()
