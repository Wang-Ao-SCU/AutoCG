
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

# -------------------------- 1. dataload与预process --------------------------
class SolvationEnergyDataset:
    def __init__(self, data_path):
        self.data = self._load_data(data_path)
        self.smiles_list = self.data['SMILES'].tolist()
        self.target_values = self.data['Ghex'].values.astype(np.float32)
        self.valid_indices = self._filter_valid_smiles()

    def _load_data(self, path):
        data = pd.read_csv(path)
        required_columns = ['SMILES', 'Ghex']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"The CSV file must contain the following columns: {required_columns}")
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
                print(f"processSMILES: {smiles}, 错误: {e}")
                continue
        return valid_indices

    def get_processed_data(self):
        X_smiles = [self.smiles_list[i] for i in self.valid_indices]
        y_target = self.target_values[self.valid_indices]
        return X_smiles, y_target

# -------------------------- 2. featuresExtract(UsingRDKit进行升维) --------------------------
def get_morgan_fingerprints(smiles_list, radius=2, nBits=1024):
    """获取Morgan指纹features"""
    fps = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无效的SMILES: {smiles}")
                fps.append(np.zeros(nBits))
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps.append(np.array(fp))
        except Exception as e:
            print(f"Error generating Morgan fingerprint: {smiles}, 错误: {e}")
            fps.append(np.zeros(nBits))
    return np.array(fps)

def get_molecular_descriptors(smiles_list):
    """UsingRDKitExtractmolecule描述符作为升维features"""
    # 获取RDKit所有可用的molecule描述符
    desc_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    
    descriptors = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无效的SMILES: {smiles}")
                descriptors.append(np.zeros(len(desc_names)))
                continue
            # 计算所有描述符
            desc_vals = calculator.CalcDescriptors(mol)
            # Handle possible NaN values值
            desc_vals = np.array([0 if np.isnan(x) else x for x in desc_vals])
            descriptors.append(desc_vals)
        except Exception as e:
            print(f"Error generating molecular descriptors: {smiles}, 错误: {e}")
            descriptors.append(np.zeros(len(desc_names)))
    
    return np.array(descriptors), desc_names

def get_maccs_keys(smiles_list):
    """获取MACCSmolecule指纹"""
    fps = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无效的SMILES: {smiles}")
                fps.append(np.zeros(167))  # MACCS有167个features
                continue
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(np.array(fp))
        except Exception as e:
            print(f"Error generating MACCS fingerprint: {smiles}, 错误: {e}")
            fps.append(np.zeros(167))
    return np.array(fps)

def get_enhanced_features(smiles_list):
    """融合多种moleculefeatures进行升维"""
    # 获取三种不同features
    morgan_fp = get_morgan_fingerprints(smiles_list)
    maccs_fp = get_maccs_keys(smiles_list)
    descriptors, _ = get_molecular_descriptors(smiles_list)
    
    # features融合(升维)
    enhanced_features = np.hstack([morgan_fp, maccs_fp, descriptors])
    print(f"升维后的features维度: {enhanced_features.shape[1]}")
    return enhanced_features

def get_graph_data(smiles_list):
    """generatemolecule图data, 适用于图神经网络"""
    graph_list = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无效的SMILES: {smiles}")
                graph_list.append(Data(
                    x=torch.zeros((1,1)), 
                    edge_index=torch.zeros((2,0), dtype=torch.long), 
                    pos=torch.zeros((1,3)), 
                    y=torch.tensor(0.0, dtype=torch.float32)
                ))
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
            edge_index = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2,0), dtype=torch.long)
            
            pos = torch.tensor([mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=torch.float32)
            
            graph_list.append(Data(x=x, edge_index=edge_index, pos=pos, y=torch.tensor(0.0, dtype=torch.float32)))
        except Exception as e:
            print(f"generate图data: {smiles}, 错误: {e}")
            graph_list.append(Data(
                x=torch.zeros((1,1)), 
                edge_index=torch.zeros((2,0), dtype=torch.long), 
                pos=torch.zeros((1,3)), 
                y=torch.tensor(0.0, dtype=torch.float32)
            ))
    return graph_list

# -------------------------- 3. model定义 --------------------------
class SGCN(nn.Module):
    """简单图卷积网络, 用于processmolecule图data"""
    def __init__(self, in_channels=1, hidden_channels=64, out_channels=1, K=2):
        super().__init__()
        self.conv1 = SGConv(in_channels, hidden_channels, K=K)
        self.conv2 = SGConv(hidden_channels, hidden_channels*2, K=K)
        self.fc1 = nn.Linear(hidden_channels*2, 64)
        self.fc2 = nn.Linear(64, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x  # 不Usingsqueeze(), 确保返回二维张量 (batch_size, 1)

class MorganRidge:
    """基于Morgan指纹和Ridge回归的model"""
    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        if X_train is None or y_train is None:
            raise ValueError("trainingdata不能为空")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("trainingdata不能为空数组")
        if len(X_train) != len(y_train):
            raise ValueError(f"features和标签长度不匹配: {len(X_train)} vs {len(y_train)}")
            
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
        grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

    def predict(self, X):
        # 确保返回二维数组格式
        return self.model.predict(X).reshape(-1, 1)

    def save(self, path):
        """savemodel"""
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, path)

    def load(self, path):
        """load the model"""
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']

class EnhancedGBR:
    """基于升维features的梯度提升回归model"""
    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train):
        if X_train is None or y_train is None:
            raise ValueError("trainingdata不能为空")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("trainingdata不能为空数组")
        if len(X_train) != len(y_train):
            raise ValueError(f"features和标签长度不匹配: {len(X_train)} vs {len(y_train)}")
        
        # GBRparameter网格搜索
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        
        grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1  # Using所有可用CPU
        )
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        print(f"GBRBest parameters: {self.best_params}")

    def predict(self, X):
        # 确保返回二维数组格式
        return self.model.predict(X).reshape(-1, 1)

    def save(self, path):
        """savemodel"""
        joblib.dump({
            'model': self.model,
            'best_params': self.best_params
        }, path)

    def load(self, path):
        """load the model"""
        data = joblib.load(path)
        self.model = data['model']
        self.best_params = data['best_params']

# -------------------------- 4. modeltraining与evaluation工具类 --------------------------
class ModelTrainer:
    """modeltraining和evaluation的工具类, 支持三 model typestype"""
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
                raise ValueError(f"{self.model_type}的trainingdata为空")
                
            train_loader = PyG_DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = PyG_DataLoader(val_data, batch_size=batch_size)
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    y_true = batch.y.unsqueeze(1)  # 确保目标值是二维的
                    y_pred = self.model(batch)
                    
                    loss = self.criterion(y_pred, y_true)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * batch_size
                
                val_loss = self.evaluate(val_loader, 'loss')
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
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
                    y_true = batch.y.unsqueeze(1)  # 确保目标值是二维的
                    y_pred = self.model(batch)
                else:
                    X_batch, y_batch = batch
                    y_true = y_batch.reshape(-1, 1)  # 确保目标值是二维的
                    y_pred = self.model.predict(X_batch)
                
                # 确保predictionresults是可迭代的数组
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
        
        # 确保返回的是二维数组, 便于统一process
        if len(result.shape) == 1:
            return result.reshape(-1, 1)
        return result

    def save_model(self, save_path):
        """savemodel到指定path"""
        if self.model_type == 'sg_cnn':
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            }, save_path)
        else:
            self.model.save(save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, model_path):
        """从指定pathload the model"""
        if self.model_type == 'sg_cnn':
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and checkpoint['optimizer_state_dict']:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.model.load(model_path)
        print(f"loaded from {model_path} load the model")

# -------------------------- 5. results可视化与save --------------------------
def plot_scatter(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name, save_path):
    plt.figure(figsize=(8, 6))
    
    # 绘制training set
    plt.scatter(y_train_true, y_train_pred, alpha=0.5, color='blue', label='Train')
    # 绘制test set
    plt.scatter(y_test_true, y_test_pred, alpha=0.5, color='orange', label='Test')
    
    # 理想线
    all_true = np.concatenate([y_train_true, y_test_true])
    plt.plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], 'r--', lw=2)
    
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'{model_name} - True vs Predicted Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}/{model_name}_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_results(y_train_true, y_train_pred, y_test_true, y_test_pred, metrics, model_name, save_path):
    # savetraining set和test set的散点data
    train_df = pd.DataFrame({
        'True_Value': y_train_true,
        'Predicted_Value': y_train_pred,
        'Set': 'Train'
    })
    test_df = pd.DataFrame({
        'True_Value': y_test_true,
        'Predicted_Value': y_test_pred,
        'Set': 'Test'
    })
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df.to_csv(f"{save_path}/{model_name}_scatter_data.csv", index=False)
    
    # saveevaluation指标
    metrics_data = pd.DataFrame({
        'Model': [model_name],
        'Train_MAE': [metrics['train_mae']],
        'Train_RMSE': [metrics['train_rmse']],
        'Train_R2': [metrics['train_r2']],
        'Test_MAE': [metrics['test_mae']],
        'Test_RMSE': [metrics['test_rmse']],
        'Test_R2': [metrics['test_r2']]
    })
    return metrics_data

# -------------------------- 6. model封装与调用接口 --------------------------
class SolvationEnergyPredictor:
    """溶剂化能prediction器, 封装了dataprocess和modeltrainingprediction的完整workflow"""
    def __init__(self, data_path=None, save_path='./results'):
        import os
        os.makedirs(save_path, exist_ok=True)
        self.models_dir = os.path.join(save_path, 'models')
        os.makedirs(self.models_dir, exist_ok=True)  # 创建modelsavedirectory
        
        self.data_loader = None
        self.X_smiles = None
        self.y_target = None
        
        if data_path:
            self.data_loader = SolvationEnergyDataset(data_path)
            self.X_smiles, self.y_target = self.data_loader.get_processed_data()
            
            if len(self.X_smiles) == 0 or len(self.y_target) == 0:
                raise ValueError("没有有效的data样本, 请检查data集")
            print(f"Successfully loaded {len(self.X_smiles)} 个有效样本")
        
        self.save_path = save_path
        self.models = {}
        self.data_dict = None

    def split_data(self, test_size=0.2, random_state=42):
        if not self.X_smiles or not self.y_target.any():
            raise ValueError("请先initializedataload器")
            
        # generate不同model需要的features
        X_morgan = get_morgan_fingerprints(self.X_smiles)
        X_enhanced = get_enhanced_features(self.X_smiles)  # 升维后的features
        X_graph = get_graph_data(self.X_smiles)
        
        # 为图data填充目标值
        for i, data in enumerate(X_graph):
            data.y = torch.tensor(self.y_target[i], dtype=torch.float32)
        
        print(f"Morgan指纹features维度: {X_morgan.shape[1]}")
        print(f"增强features维度: {X_enhanced.shape[1]}")
        
        train_idx, test_idx = train_test_split(range(len(self.y_target)), test_size=test_size, random_state=random_state)
        
        self.data_dict = {
            'morgan_ridge': {
                'X_train': X_morgan[train_idx],
                'y_train': self.y_target[train_idx],
                'X_test': X_morgan[test_idx],
                'y_test': self.y_target[test_idx]
            },
            'enhanced_gbr': {
                'X_train': X_enhanced[train_idx],
                'y_train': self.y_target[train_idx],
                'X_test': X_enhanced[test_idx],
                'y_test': self.y_target[test_idx]
            },
            'sg_cnn': {
                'train': [X_graph[i] for i in train_idx],
                'test': [X_graph[i] for i in test_idx]
            },
            'test_idx': test_idx,
            'y_true': self.y_target[test_idx]
        }
        return self.data_dict

    def train_all_models(self, data_dict=None, epochs=50, batch_size=16, lr=0.001):
        if data_dict is None:
            if self.data_dict is None:
                raise ValueError("请先调用split_data()获取data字典")
            data_dict = self.data_dict
            
        # modeltype改为: morgan_ridge, enhanced_gbr, sg_cnn
        model_types = ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']
        for model_type in model_types:
            print(f"\nTraining {model_type}...")
            trainer = ModelTrainer(model_type)
            
            if model_type not in data_dict:
                raise ValueError(f"data字典中没有 {model_type} 的data")
            
            if model_type in ['morgan_ridge', 'enhanced_gbr']:
                X_train = data_dict[model_type]['X_train']
                y_train = data_dict[model_type]['y_train']
                trainer.train(X_train, y_train)
            else:
                train_data = data_dict[model_type]['train']
                val_data = data_dict[model_type]['test'][:len(data_dict[model_type]['test'])//2]
                
                if not train_data:
                    raise ValueError(f"{model_type}的trainingdata为空, 请检查featuresgenerate过程")
                
                trainer.train(train_data, val_data, epochs=epochs, batch_size=batch_size, lr=lr)
            
            self.models[model_type] = trainer
        
        # training完成后save所有model
        self.save_all_models()

    def evaluate_all_models(self, data_dict=None):
        if data_dict is None:
            if self.data_dict is None:
                raise ValueError("请先调用split_data()获取data字典")
            data_dict = self.data_dict
            
        all_metrics = []
        
        for model_name, trainer in self.models.items():
            print(f"\nEvaluating {model_name}...")
            if model_name not in data_dict:
                raise ValueError(f"data字典中没有 {model_name} 的data")
            
            # --- 准备trainingdata (Validation/Train) ---
            if model_name in ['morgan_ridge', 'enhanced_gbr']:
                X_train = data_dict[model_name]['X_train']
                y_train_true = data_dict[model_name]['y_train']
                # Predict
                y_train_pred = trainer.predict(X_train).flatten()
            elif model_name == 'sg_cnn':
                train_data_list = data_dict[model_name]['train']
                y_train_true = np.array([d.y.item() for d in train_data_list])
                # 批量prediction
                train_loader = PyG_DataLoader(train_data_list, batch_size=16)
                y_train_pred_list = []
                for batch in train_loader:
                    batch = batch.to(trainer.device)
                    yp = trainer.predict(batch)
                    if isinstance(yp, np.ndarray):
                        y_train_pred_list.extend(yp.flatten().tolist())
                    elif isinstance(yp, torch.Tensor):
                        y_train_pred_list.extend(yp.cpu().numpy().flatten().tolist())
                    else:
                        y_train_pred_list.extend(yp)
                y_train_pred = np.array(y_train_pred_list)

            # --- 准备测试data (Test) ---
            if model_name in ['morgan_ridge', 'enhanced_gbr']:
                X_test = data_dict[model_name]['X_test']
                y_test_true = data_dict[model_name]['y_test']
                y_test_pred = trainer.predict(X_test).flatten()
            elif model_name == 'sg_cnn':
                test_data_list = data_dict[model_name]['test']
                y_test_true = np.array([d.y.item() for d in test_data_list])
                test_loader = PyG_DataLoader(test_data_list, batch_size=16)
                y_test_pred_list = []
                for batch in test_loader:
                    batch = batch.to(trainer.device)
                    yp = trainer.predict(batch)
                    if isinstance(yp, np.ndarray):
                        y_test_pred_list.extend(yp.flatten().tolist())
                    elif isinstance(yp, torch.Tensor):
                        y_test_pred_list.extend(yp.cpu().numpy().flatten().tolist())
                    else:
                        y_test_pred_list.extend(yp)
                y_test_pred = np.array(y_test_pred_list)
            
            # 计算指标
            metrics = {
                'train_mae': mean_absolute_error(y_train_true, y_train_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
                'train_r2': r2_score(y_train_true, y_train_pred),
                'test_mae': mean_absolute_error(y_test_true, y_test_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test_true, y_test_pred)),
                'test_r2': r2_score(y_test_true, y_test_pred)
            }
            
            print(f"Train - MAE: {metrics['train_mae']:.4f} | R2: {metrics['train_r2']:.4f}")
            print(f"Test  - MAE: {metrics['test_mae']:.4f} | R2: {metrics['test_r2']:.4f}")
            
            # 绘图和savedata
            plot_scatter(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name, self.save_path)
            metrics_df = save_results(y_train_true, y_train_pred, y_test_true, y_test_pred, metrics, model_name, self.save_path)
            all_metrics.append(metrics_df)
        
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_metrics_df.to_csv(f"{self.save_path}/all_models_metrics.csv", index=False)
        print(f"\nAll results saved to {self.save_path}")

    def predict_single_smiles(self, smiles, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained. Train first with train_all_models()")
        
        trainer = self.models[model_name]
        if model_name == 'morgan_ridge':
            X = get_morgan_fingerprints([smiles])
            pred = trainer.predict(X)[0][0]  # 从二维数组中Extract单个值
        elif model_name == 'enhanced_gbr':
            X = get_enhanced_features([smiles])
            pred = trainer.predict(X)[0][0]  # 从二维数组中Extract单个值
        elif model_name == 'sg_cnn':
            X = get_graph_data([smiles])[0]
            X = X.to(trainer.device)
            pred = trainer.predict(X)[0][0].item()  # 从二维数组中Extract单个值
        else:
            raise ValueError("Model name must be 'morgan_ridge', 'enhanced_gbr' or 'sg_cnn'")
        return pred

    def save_all_models(self):
        """save所有training好的model"""
        for model_name, trainer in self.models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
            trainer.save_model(model_path)

    def load_all_models(self):
        """load所有save的model"""
        model_types = ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']
        for model_type in model_types:
            model_path = os.path.join(self.models_dir, f"{model_type}_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"modelfile不存在: {model_path}")
            
            trainer = ModelTrainer(model_type)
            trainer.load_model(model_path)
            self.models[model_type] = trainer
        print("所有modelload完成")

# -------------------------- 7. 调用示例 --------------------------
def main():
    DATA_PATH = './ML1_Ghex.csv'  # replace with yourCSVdatapath
    SAVE_PATH = './hex'
    EPOCHS = 50
    BATCH_SIZE = 16
    LR = 0.001

    try:
        # training并savemodel
        predictor = SolvationEnergyPredictor(DATA_PATH, SAVE_PATH)
        data_dict = predictor.split_data(test_size=0.2)
        predictor.train_all_models(data_dict, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
        predictor.evaluate_all_models(data_dict)
        
        # 测试单一样本prediction
        test_smiles = 'CCCC'  # 水的SMILES
        print(f"\nPredicting value for SMILES: {test_smiles}")
        for model_name in ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']:
            pred = predictor.predict_single_smiles(test_smiles, model_name)
            print(f"{model_name} Prediction: {pred:.4f}")
        
        # 演示for example何Load saved models
        print("\n演示Load saved models...")
        new_predictor = SolvationEnergyPredictor(save_path=SAVE_PATH)
        new_predictor.load_all_models()
        
        # Usingload的model进行prediction
        print(f"\nUsing loaded models to predict for SMILES: {test_smiles}")
        for model_name in ['morgan_ridge', 'enhanced_gbr', 'sg_cnn']:
            pred = new_predictor.predict_single_smiles(test_smiles, model_name)
            print(f"Loaded {model_name} Prediction: {pred:.4f}")
            
    except Exception as e:
        print(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
