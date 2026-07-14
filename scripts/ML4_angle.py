import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings
import contextlib
import os
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')
if not hasattr(contextlib, 'nullcontext'):

    class nullcontext:

        def __enter__(self):
            return None

        def __exit__(self, *excinfo):
            return False
    contextlib.nullcontext = nullcontext

class AngleParameterDataset:

    def __init__(self, data_path):
        self.data = self._load_data(data_path)
        self.smiles_a_list = self.data['smiles_1'].tolist()
        self.smiles_b_list = self.data['smiles_2'].tolist()
        self.smiles_c_list = self.data['smiles_3'].tolist()
        self.target_r = self.data['r'].values.astype(np.float32)
        self.target_k = self.data['k'].values.astype(np.float32)
        self.valid_indices = self._filter_valid_smiles_triples()

    def _load_data(self, path):
        data = pd.read_csv(path)
        required_columns = ['smiles_1', 'smiles_2', 'smiles_3', 'r', 'k']
        if not all((col in data.columns for col in required_columns)):
            raise ValueError(f'The CSV file must contain the following columns: {required_columns}')
        data = data.dropna(subset=['smiles_1', 'smiles_2', 'smiles_3', 'r', 'k'])
        return data

    def _filter_valid_smiles_triples(self):
        valid_indices = []
        for i in range(len(self.smiles_a_list)):
            smiles_a = self.smiles_a_list[i]
            smiles_b = self.smiles_b_list[i]
            smiles_c = self.smiles_c_list[i]
            try:
                mol_a = Chem.MolFromSmiles(smiles_a)
                mol_b = Chem.MolFromSmiles(smiles_b)
                mol_c = Chem.MolFromSmiles(smiles_c)
                if mol_a is not None and mol_b is not None and (mol_c is not None):
                    valid_indices.append(i)
            except Exception as e:
                continue
        return valid_indices

    def get_processed_data(self):
        X_smiles_a = [self.smiles_a_list[i] for i in self.valid_indices]
        X_smiles_b = [self.smiles_b_list[i] for i in self.valid_indices]
        X_smiles_c = [self.smiles_c_list[i] for i in self.valid_indices]
        y_r = self.target_r[self.valid_indices]
        y_k = self.target_k[self.valid_indices]
        return (X_smiles_a, X_smiles_b, X_smiles_c, y_r, y_k)

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

def get_triple_enhanced_features(smiles_a_list, smiles_b_list, smiles_c_list):
    desc_names = [desc[0] for desc in Descriptors._descList]
    features_a = []
    features_b = []
    features_c = []
    for smiles_a, smiles_b, smiles_c in zip(smiles_a_list, smiles_b_list, smiles_c_list):
        morgan_a = get_single_morgan_fingerprint(smiles_a)
        maccs_a = get_single_maccs_key(smiles_a)
        desc_a = get_single_descriptors(smiles_a, desc_names)
        feat_a = np.hstack([morgan_a, maccs_a, desc_a])
        morgan_b = get_single_morgan_fingerprint(smiles_b)
        maccs_b = get_single_maccs_key(smiles_b)
        desc_b = get_single_descriptors(smiles_b, desc_names)
        feat_b = np.hstack([morgan_b, maccs_b, desc_b])
        morgan_c = get_single_morgan_fingerprint(smiles_c)
        maccs_c = get_single_maccs_key(smiles_c)
        desc_c = get_single_descriptors(smiles_c, desc_names)
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
    return (triple_features, desc_names)

class SingleTargetModel:

    def __init__(self, target_name, model_type):
        self.target_name = target_name
        self.model_type = model_type
        self.model = None
        self.best_params = None

    def _init_base_model(self):
        if self.model_type == 'ridge':
            return Ridge()
        elif self.model_type == 'gbr':
            return GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError("model_type must be 'ridge' or 'gbr'")

    def train(self, X_train, y_train):
        self._validate_train_data(X_train, y_train)
        param_grid = self._get_param_grid()
        grid = GridSearchCV(self._init_base_model(), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1 if self.model_type == 'gbr' else 1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

    def _get_param_grid(self):
        if self.model_type == 'ridge':
            return {'alpha': [0.01, 0.1, 1, 10, 100]}
        elif self.model_type == 'gbr':
            return {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'min_samples_split': [2, 5]}

    def _validate_train_data(self, X_train, y_train):
        if X_train is None or y_train is None:
            raise ValueError(f'[{self.model_type}_{self.target_name}] Training data must not be empty (contains None)')
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError(f'[{self.model_type}_{self.target_name}] Training set is empty; adjust the test-set ratio')
        if len(X_train) != len(y_train):
            raise ValueError(f'[{self.model_type}_{self.target_name}] Feature and label lengths do not match: X={len(X_train)}, y={len(y_train)}')
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError(f'[{self.model_type}_{self.target_name}] Training data contain NaN values')

    def predict(self, X):
        if self.model is None:
            raise RuntimeError(f'[{self.model_type}_{self.target_name}] Model has not been trained; call train() first')
        return self.model.predict(X)

    def save(self, save_path):
        joblib.dump({'target_name': self.target_name, 'model_type': self.model_type, 'model': self.model, 'best_params': self.best_params}, save_path)

    def load(self, model_path):
        data = joblib.load(model_path)
        if data['target_name'] != self.target_name or data['model_type'] != self.model_type:
            raise ValueError(
                f"Model type mismatch, expected ({self.model_type}, {self.target_name}), "
                f"actual ({data['model_type']}, {data['target_name']})"
            )
        self.model = data['model']
        self.best_params = data['best_params']

class DualTargetModelWrapper:

    def __init__(self, base_model_type):
        self.base_model_type = base_model_type
        self.model_r = SingleTargetModel(target_name='r', model_type=base_model_type)
        self.model_k = SingleTargetModel(target_name='k', model_type=base_model_type)

    def train(self, X_train, y_train_r, y_train_k):
        self.model_r.train(X_train, y_train_r)
        self.model_k.train(X_train, y_train_k)

    def predict(self, X):
        y_pred_r = self.model_r.predict(X).reshape(-1, 1)
        y_pred_k = self.model_k.predict(X).reshape(-1, 1)
        return np.hstack([y_pred_r, y_pred_k])

    def evaluate(self, X_test, y_test_r, y_test_k):
        y_pred_r = self.model_r.predict(X_test)
        y_pred_k = self.model_k.predict(X_test)
        r_metrics = {'mae': mean_absolute_error(y_test_r, y_pred_r), 'rmse': np.sqrt(mean_squared_error(y_test_r, y_pred_r)), 'r2': r2_score(y_test_r, y_pred_r)}
        k_metrics = {'mae': mean_absolute_error(y_test_k, y_pred_k), 'rmse': np.sqrt(mean_squared_error(y_test_k, y_pred_k)), 'r2': r2_score(y_test_k, y_pred_k)}
        return {'r': r_metrics, 'k': k_metrics, 'avg_mae': (r_metrics['mae'] + k_metrics['mae']) / 2, 'avg_rmse': (r_metrics['rmse'] + k_metrics['rmse']) / 2, 'avg_r2': (r_metrics['r2'] + k_metrics['r2']) / 2}

    def save_models(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model_r_path = os.path.join(save_dir, f'{self.base_model_type}_model_r.pkl')
        model_k_path = os.path.join(save_dir, f'{self.base_model_type}_model_k.pkl')
        self.model_r.save(model_r_path)
        self.model_k.save(model_k_path)

    def load_models(self, model_dir):
        model_r_path = os.path.join(model_dir, f'{self.base_model_type}_model_r.pkl')
        model_k_path = os.path.join(model_dir, f'{self.base_model_type}_model_k.pkl')
        self.model_r.load(model_r_path)
        self.model_k.load(model_k_path)

class ModelPipeline:

    def __init__(self, data_path, save_path='./angle_results'):
        self.dataset = AngleParameterDataset(data_path)
        self.X_smiles_a, self.X_smiles_b, self.X_smiles_c, self.y_r, self.y_k = self.dataset.get_processed_data()
        self.X_features, _ = get_triple_enhanced_features(self.X_smiles_a, self.X_smiles_b, self.X_smiles_c)
        self.save_path = save_path
        self.models_dir = os.path.join(save_path, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.trained_models = {}

    def split_data(self, test_size=0.2, random_state=42):
        if len(self.X_features) < 10:
            raise ValueError(f'Too few valid samples({len(self.X_features)}), consider reducing test_size')
        self.X_train, self.X_test, y_r_train, y_r_test, y_k_train, y_k_test = train_test_split(self.X_features, self.y_r, self.y_k, test_size=test_size, random_state=random_state)
        return {'X_train': self.X_train, 'X_test': self.X_test, 'y_r': {'train': y_r_train, 'test': y_r_test}, 'y_k': {'train': y_k_train, 'test': y_k_test}}

    def train_models(self, data_split, model_types=['ridge', 'gbr']):
        for model_type in model_types:
            if model_type not in ['ridge', 'gbr']:
                continue
            model_wrapper = DualTargetModelWrapper(base_model_type=model_type)
            model_wrapper.train(X_train=data_split['X_train'], y_train_r=data_split['y_r']['train'], y_train_k=data_split['y_k']['train'])
            self.trained_models[model_type] = model_wrapper
            model_wrapper.save_models(self.models_dir)

    def evaluate_models(self, data_split):
        all_metrics = []
        X_test = data_split['X_test']
        y_test_r = data_split['y_r']['test']
        y_test_k = data_split['y_k']['test']
        for model_type, model_wrapper in self.trained_models.items():
            metrics = model_wrapper.evaluate(X_test, y_test_r, y_test_k)
            y_pred = model_wrapper.predict(X_test)
            y_pred_r = y_pred[:, 0]
            y_pred_k = y_pred[:, 1]
            self._save_evaluation_results(model_type, y_test_r, y_pred_r, y_test_k, y_pred_k, metrics)
            all_metrics.append(self._format_metrics_df(model_type, metrics))
        if all_metrics:
            metrics_df = pd.concat(all_metrics, ignore_index=True)
            metrics_df.to_csv(os.path.join(self.save_path, 'all_models_metrics.csv'), index=False)
            return metrics_df
        else:
            return None

    def _save_evaluation_results(self, model_type, y_true_r, y_pred_r, y_true_k, y_pred_k, metrics):
        result_data = pd.DataFrame({f'True_r_{model_type}': y_true_r, f'Pred_r_{model_type}': y_pred_r, f'True_k_{model_type}': y_true_k, f'Pred_k_{model_type}': y_pred_k}).dropna()
        result_data.to_csv(os.path.join(self.save_path, f'{model_type}_prediction_results.csv'), index=False)
        self._plot_prediction_scatter(model_type, y_true_r, y_pred_r, y_true_k, y_pred_k, metrics)

    def _plot_prediction_scatter(self, model_type, y_true_r, y_pred_r, y_true_k, y_pred_k, metrics):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(y_true_r, y_pred_r, alpha=0.6, color='blue')
        plt.plot([y_true_r.min(), y_true_r.max()], [y_true_r.min(), y_true_r.max()], 'r--', lw=2)
        plt.xlabel(f'True {model_type.upper()} r (equilibrium angle)')
        plt.ylabel(f'Predicted {model_type.upper()} r')
        plt.title(f"{model_type.upper()} - r: True vs Predicted (R2={metrics['r']['r2']:.4f})")
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.scatter(y_true_k, y_pred_k, alpha=0.6, color='green')
        plt.plot([y_true_k.min(), y_true_k.max()], [y_true_k.min(), y_true_k.max()], 'r--', lw=2)
        plt.xlabel(f'True {model_type.upper()} k (force constant)')
        plt.ylabel(f'Predicted {model_type.upper()} k')
        plt.title(f"{model_type.upper()} - k: True vs Predicted (R2={metrics['k']['r2']:.4f})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'{model_type}_prediction_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _format_metrics_df(self, model_type, metrics):
        r_df = pd.DataFrame({'Model_Type': [model_type.upper()], 'Target': ['r'], 'MAE': [metrics['r']['mae']], 'RMSE': [metrics['r']['rmse']], 'R2': [metrics['r']['r2']]})
        k_df = pd.DataFrame({'Model_Type': [model_type.upper()], 'Target': ['k'], 'MAE': [metrics['k']['mae']], 'RMSE': [metrics['k']['rmse']], 'R2': [metrics['k']['r2']]})
        return pd.concat([r_df, k_df], ignore_index=True)

    def predict_single_triple(self, smiles_a, smiles_b, smiles_c, model_type):
        if model_type not in self.trained_models:
            raise ValueError(f'model {model_type} has not been trained; available trained models: {list(self.trained_models.keys())}')
        X = get_triple_enhanced_features([smiles_a], [smiles_b], [smiles_c])[0]
        X = np.nan_to_num(X)
        model_wrapper = self.trained_models[model_type]
        y_pred_r = model_wrapper.model_r.predict(X)[0]
        y_pred_k = model_wrapper.model_k.predict(X)[0]
        return {'model_type': model_type.upper(), 'smiles_a': smiles_a, 'smiles_b': smiles_b, 'smiles_c': smiles_c, 'predicted_r': round(y_pred_r, 4), 'predicted_k': round(y_pred_k, 4)}

    def load_saved_models(self, model_types=['ridge', 'gbr']):
        for model_type in model_types:
            model_r_path = os.path.join(self.models_dir, f'{model_type}_model_r.pkl')
            model_k_path = os.path.join(self.models_dir, f'{model_type}_model_k.pkl')
            if not os.path.exists(model_r_path) or not os.path.exists(model_k_path):
                continue
            model_wrapper = DualTargetModelWrapper(base_model_type=model_type)
            model_wrapper.load_models(self.models_dir)
            self.trained_models[model_type] = model_wrapper

def main():
    DATA_PATH = './test/angle_data.csv'
    SAVE_PATH = './angle_prediction_results'
    TEST_SIZE = 0.2
    TRAIN_MODEL_TYPES = ['ridge', 'gbr']
    try:
        pipeline = ModelPipeline(data_path=DATA_PATH, save_path=SAVE_PATH)
        data_split = pipeline.split_data(test_size=TEST_SIZE)
        pipeline.train_models(data_split, model_types=TRAIN_MODEL_TYPES)
        pipeline.evaluate_models(data_split)
        test_smiles_a = 'CCC'
        test_smiles_b = 'COC'
        test_smiles_c = 'CCO'
        for model_type in TRAIN_MODEL_TYPES:
            if model_type in pipeline.trained_models:
                pred_result = pipeline.predict_single_triple(test_smiles_a, test_smiles_b, test_smiles_c, model_type)
        new_pipeline = ModelPipeline(data_path=DATA_PATH, save_path=SAVE_PATH)
        new_pipeline.load_saved_models(model_types=TRAIN_MODEL_TYPES)
        for model_type in TRAIN_MODEL_TYPES:
            if model_type in new_pipeline.trained_models:
                pred_result = new_pipeline.predict_single_triple(test_smiles_a, test_smiles_b, test_smiles_c, model_type)
    except Exception as e:
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()
