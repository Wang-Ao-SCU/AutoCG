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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyG_DataLoader
from torch_geometric.nn import SGConv, global_mean_pool
import warnings
import contextlib
import os
import joblib
warnings.filterwarnings('ignore')
if not hasattr(contextlib, 'nullcontext'):

    class nullcontext:

        def __enter__(self):
            return None

        def __exit__(self, *excinfo):
            return False
    contextlib.nullcontext = nullcontext

class SolvationEnergyDataset:

    def __init__(self, data_path):
        self.data = self._load_data(data_path)
        self.smiles_list = self.data['SMILES'].tolist()
        self.target_values = self.data['Ghex'].values.astype(np.float32)
        self.valid_indices = self._filter_valid_smiles()

    def _load_data(self, path):
        data = pd.read_csv(path)
        required_columns = ['SMILES', 'Ghex']
        if not all((col in data.columns for col in required_columns)):
            raise ValueError(f'The CSV file must contain the following columns: {required_columns}')
        data = data.dropna(subset=['SMILES', 'Ghex'])
        return data

    def _filter_valid_smiles(self):
        valid_indices = []
        for i, smiles in enumerate(self.smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                valid_indices.append(i)
            except Exception as e:
                continue
        return valid_indices

    def get_processed_data(self):
        X_smiles = [self.smiles_list[i] for i in self.valid_indices]
        y_target = self.target_values[self.valid_indices]
        return (X_smiles, y_target)

def get_morgan_fingerprints(smiles_list, radius=2, nBits=1024):
    fps = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fps.append(np.zeros(nBits))
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps.append(np.array(fp))
        except Exception as e:
            fps.append(np.zeros(nBits))
    return np.array(fps)

def get_molecular_descriptors(smiles_list):
    desc_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    descriptors = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                descriptors.append(np.zeros(len(desc_names)))
                continue
            desc_vals = calculator.CalcDescriptors(mol)
            desc_vals = np.array([0 if np.isnan(x) else x for x in desc_vals])
            descriptors.append(desc_vals)
        except Exception as e:
            descriptors.append(np.zeros(len(desc_names)))
    return (np.array(descriptors), desc_names)

def get_maccs_keys(smiles_list):
    fps = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fps.append(np.zeros(167))
                continue
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(np.array(fp))
        except Exception as e:
            fps.append(np.zeros(167))
    return np.array(fps)

def get_enhanced_features(smiles_list):
    morgan_fp = get_morgan_fingerprints(smiles_list)
    maccs_fp = get_maccs_keys(smiles_list)
    descriptors, _ = get_molecular_descriptors(smiles_list)
    enhanced_features = np.hstack([morgan_fp, maccs_fp, descriptors])
    return enhanced_features

def get_graph_data(smiles_list):
    graph_list = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                graph_list.append(Data(x=torch.zeros((1, 1)), edge_index=torch.zeros((2, 0), dtype=torch.long), pos=torch.zeros((1, 3)), y=torch.tensor(0.0, dtype=torch.float32)))
                continue
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float32).unsqueeze(1)
            edges = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges.append([i, j])
                edges.append([j, i])
            edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
            pos = torch.tensor([mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=torch.float32)
            graph_list.append(Data(x=x, edge_index=edge_index, pos=pos, y=torch.tensor(0.0, dtype=torch.float32)))
        except Exception as e:
            graph_list.append(Data(x=torch.zeros((1, 1)), edge_index=torch.zeros((2, 0), dtype=torch.long), pos=torch.zeros((1, 3)), y=torch.tensor(0.0, dtype=torch.float32)))
    return graph_list

class SGCN(nn.Module):

    def __init__(self, in_channels=1, hidden_channels=64, out_channels=1, K=2):
        super().__init__()
        self.conv1 = SGConv(in_channels, hidden_channels, K=K)
        self.conv2 = SGConv(hidden_channels, hidden_channels * 2, K=K)
        self.fc1 = nn.Linear(hidden_channels * 2, 64)
        self.fc2 = nn.Linear(64, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class MorganRidge:

    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        if X_train is None or y_train is None:
            raise ValueError('trainingdata不能为空')
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError('trainingdata不能为空数组')
        if len(X_train) != len(y_train):
            raise ValueError(f'features和标签长度不匹配: {len(X_train)} vs {len(y_train)}')
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
        grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)

    def save(self, path):
        joblib.dump({'model': self.model, 'best_params': self.best_params}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']

class EnhancedGBR:

    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        if X_train is None or y_train is None:
            raise ValueError('trainingdata不能为空')
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError('trainingdata不能为空数组')
        if len(X_train) != len(y_train):
            raise ValueError(f'features和标签长度不匹配: {len(X_train)} vs {len(y_train)}')
        param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5]}
        grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)

    def save(self, path):
        joblib.dump({'model': self.model, 'best_params': self.best_params}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']

class ModelTrainer:

    def __init__(self, model_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_type = model_type
        self.device = device
        self.model = self._init_model()
        self.optimizer = None
        self.criterion = nn.MSELoss()

    def _init_model(self):
        if self.model_type == 'sg_cnn':
            return SGCN().to(self.device)
        elif self.model_type == 'morgan_ridge':
            return MorganRidge()
        elif self.model_type == 'enhanced_gbr':
            return EnhancedGBR()
        else:
            raise ValueError("Model type must be 'sg_cnn', 'morgan_ridge' or 'enhanced_gbr'")

    def train(self, train_data, val_data, epochs=50, batch_size=16, lr=0.001):
        if self.model_type == 'sg_cnn':
            if not train_data:
                raise ValueError(f'{self.model_type}的trainingdata为空')
            train_loader = PyG_DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = PyG_DataLoader(val_data, batch_size=batch_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    y_true = batch.y.unsqueeze(1)
                    y_pred = self.model(batch)
                    loss = self.criterion(y_pred, y_true)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * batch_size
                val_loss = self.evaluate(val_loader, 'loss')
                if (epoch + 1) % 10 == 0:
                    pass
        elif self.model_type in ['morgan_ridge', 'enhanced_gbr']:
            self.model.train(train_data, val_data)

    def evaluate(self, data_loader, metric_type):
        if self.model_type == 'sg_cnn':
            self.model.eval()
        y_true_list = []
        y_pred_list = []
        with torch.no_grad() if self.model_type == 'sg_cnn' else contextlib.nullcontext():
            for batch in data_loader:
                if self.model_type == 'sg_cnn':
                    batch = batch.to(self.device)
                    y_true = batch.y.unsqueeze(1)
                    y_pred = self.model(batch)
                else:
                    X_batch, y_batch = batch
                    y_true = y_batch.reshape(-1, 1)
                    y_pred = self.model.predict(X_batch)
                if self.model_type == 'sg_cnn':
                    y_true_list.extend(y_true.cpu().numpy().flatten())
                    y_pred_list.extend(y_pred.cpu().numpy().flatten())
                else:
                    y_true_list.extend(y_true.flatten())
                    y_pred_list.extend(y_pred.flatten())
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        if metric_type == 'loss':
            return mean_squared_error(y_true, y_pred)
        elif metric_type == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric_type == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric_type == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError("Metric type must be 'loss', 'mae', 'rmse' or 'r2'")

    def predict(self, data):
        if self.model_type == 'sg_cnn':
            self.model.eval()
        if self.model_type in ['morgan_ridge', 'enhanced_gbr']:
            result = self.model.predict(data)
        else:
            with torch.no_grad():
                data = data.to(self.device)
                result = self.model(data)
            result = result.cpu().numpy()
        if len(result.shape) == 1:
            return result.reshape(-1, 1)
        return result

    def save_model(self, save_path):
        if self.model_type == 'sg_cnn':
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None}, save_path)
        else:
            self.model.save(save_path)

    def load_model(self, model_path):
        if self.model_type == 'sg_cnn':
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load(model_path)

def plot_scatter(y_true, y_pred, model_name, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'{model_name} - True vs Predicted Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/{model_name}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(y_true, y_pred, metrics, model_name, save_path):
    scatter_data = pd.DataFrame({'True_Value': y_true, 'Predicted_Value': y_pred})
    scatter_data.to_csv(f'{save_path}/{model_name}_scatter_data.csv', index=False)
    metrics_data = pd.DataFrame({'Model': [model_name], 'MAE': [metrics['mae']], 'RMSE': [metrics['rmse']], 'R2': [metrics['r2']]})
    return metrics_data

class SolvationEnergyPredictor:

    def __init__(self, data_path=None, save_path='./results'):
        import os
        os.makedirs(save_path, exist_ok=True)
        self.models_dir = os.path.join(save_path, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.data_loader = None
        self.X_smiles = None
        self.y_target = None
        if data_path:
            self.data_loader = SolvationEnergyDataset(data_path)
            self.X_smiles, self.y_target = self.data_loader.get_processed_data()
            if len(self.X_smiles) == 0 or len(self.y_target) == 0:
                raise ValueError('没有有效的data样本, 请检查data集')
        self.save_path = save_path
        self.models = {}
        self.data_dict = None

    def split_data(self, test_size=0.2, random_state=42):
        if not self.X_smiles or not self.y_target.any():
            raise ValueError('请先initializedataload器')
        X_morgan = get_morgan_fingerprints(self.X_smiles)
        X_enhanced = get_enhanced_features(self.X_smiles)
        X_graph = get_graph_data(self.X_smiles)
        for i, data in enumerate(X_graph):
            data.y = torch.tensor(self.y_target[i], dtype=torch.float32)
        train_idx, test_idx = train_test_split(range(len(self.y_target)), test_size=test_size, random_state=random_state)
        self.data_dict = {'morgan_ridge': {'X_train': X_morgan[train_idx], 'y_train': self.y_target[train_idx], 'X_test': X_morgan[test_idx], 'y_test': self.y_target[test_idx]}, 'enhanced_gbr': {'X_train': X_enhanced[train_idx], 'y_train': self.y_target[train_idx], 'X_test': X_enhanced[test_idx], 'y_test': self.y_target[test_idx]}, 'sg_cnn': {'train': [X_graph[i] for i in train_idx], 'test': [X_graph[i] for i in test_idx]}, 'test_idx': test_idx, 'y_true': self.y_target[test_idx]}
        return self.data_dict

    def train_all_models(self, data_dict=None, epochs=50, batch_size=16, lr=0.001):
        if data_dict is None:
            if self.data_dict is None:
                raise ValueError('请先调用split_data()获取data字典')
            data_dict = self.data_dict
        model_types = ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']
        for model_type in model_types:
            trainer = ModelTrainer(model_type)
            if model_type not in data_dict:
                raise ValueError(f'data字典中没有 {model_type} 的data')
            if model_type in ['morgan_ridge', 'enhanced_gbr']:
                X_train = data_dict[model_type]['X_train']
                y_train = data_dict[model_type]['y_train']
                trainer.train(X_train, y_train)
            else:
                train_data = data_dict[model_type]['train']
                val_data = data_dict[model_type]['test'][:len(data_dict[model_type]['test']) // 2]
                if not train_data:
                    raise ValueError(f'{model_type}的trainingdata为空, 请检查featuresgenerate过程')
                trainer.train(train_data, val_data, epochs=epochs, batch_size=batch_size, lr=lr)
            self.models[model_type] = trainer
        self.save_all_models()

    def evaluate_all_models(self, data_dict=None):
        if data_dict is None:
            if self.data_dict is None:
                raise ValueError('请先调用split_data()获取data字典')
            data_dict = self.data_dict
        all_metrics = []
        y_true = data_dict['y_true']
        for model_name, trainer in self.models.items():
            if model_name not in data_dict:
                raise ValueError(f'data字典中没有 {model_name} 的data')
            if model_name in ['morgan_ridge', 'enhanced_gbr']:
                X_test = data_dict[model_name]['X_test']
                y_pred = trainer.predict(X_test).flatten()
            else:
                test_data = data_dict[model_name]['test']
                test_loader = PyG_DataLoader(test_data, batch_size=16)
                y_pred = []
                for batch in test_loader:
                    batch = batch.to(trainer.device)
                    yp = trainer.predict(batch)
                    if isinstance(yp, np.ndarray):
                        yp_flat = yp.flatten()
                        y_pred.extend(yp_flat.tolist())
                    elif isinstance(yp, torch.Tensor):
                        yp_flat = yp.cpu().numpy().flatten()
                        y_pred.extend(yp_flat.tolist())
                    else:
                        y_pred.append(yp)
            y_pred = np.array(y_pred)
            metrics = {'mae': mean_absolute_error(y_true, y_pred), 'rmse': np.sqrt(mean_squared_error(y_true, y_pred)), 'r2': r2_score(y_true, y_pred)}
            plot_scatter(y_true, y_pred, model_name, self.save_path)
            metrics_df = save_results(y_true, y_pred, metrics, model_name, self.save_path)
            all_metrics.append(metrics_df)
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_metrics_df.to_csv(f'{self.save_path}/all_models_metrics.csv', index=False)

    def predict_single_smiles(self, smiles, model_name):
        if model_name not in self.models:
            raise ValueError(f'Model {model_name} not trained. Train first with train_all_models()')
        trainer = self.models[model_name]
        if model_name == 'morgan_ridge':
            X = get_morgan_fingerprints([smiles])
            pred = trainer.predict(X)[0][0]
        elif model_name == 'enhanced_gbr':
            X = get_enhanced_features([smiles])
            pred = trainer.predict(X)[0][0]
        elif model_name == 'sg_cnn':
            X = get_graph_data([smiles])[0]
            X = X.to(trainer.device)
            pred = trainer.predict(X)[0][0].item()
        else:
            raise ValueError("Model name must be 'morgan_ridge', 'enhanced_gbr' or 'sg_cnn'")
        return pred

    def save_all_models(self):
        for model_name, trainer in self.models.items():
            model_path = os.path.join(self.models_dir, f'{model_name}_model.pkl')
            trainer.save_model(model_path)

    def load_all_models(self):
        model_types = ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']
        for model_type in model_types:
            model_path = os.path.join(self.models_dir, f'{model_type}_model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'modelfile不存在: {model_path}')
            trainer = ModelTrainer(model_type)
            trainer.load_model(model_path)
            self.models[model_type] = trainer

def main():
    DATA_PATH = './ML1_Ghex.csv'
    SAVE_PATH = './hex'
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 0.001
    try:
        predictor = SolvationEnergyPredictor(DATA_PATH, SAVE_PATH)
        data_dict = predictor.split_data(test_size=0.2)
        predictor.train_all_models(data_dict, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
        predictor.evaluate_all_models(data_dict)
        test_smiles = 'CCCC'
        for model_name in ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']:
            pred = predictor.predict_single_smiles(test_smiles, model_name)
        new_predictor = SolvationEnergyPredictor(save_path=SAVE_PATH)
        new_predictor.load_all_models()
        for model_name in ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']:
            pred = new_predictor.predict_single_smiles(test_smiles, model_name)
    except Exception as e:
        import traceback
        traceback.print_exc()
if __name__ == '__main__':
    main()
