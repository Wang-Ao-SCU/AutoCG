

import pandas as pd
import numpy as np
import matplotlib
# Use a non-interactive Matplotlib backend to avoid Qt platform plugin issues
matplotlib.use('Agg')  # Key change: use the Agg backend so no GUI is required
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Create the output directory
output_dir = 'model_results_test3'
os.makedirs(output_dir, exist_ok=True)
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)
models_dir = os.path.join(output_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

class ModelComparison:
    def __init__(self, data_path=None, random_state=42):
        """Initialize the model comparison class"""
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.scaler = StandardScaler()
        
        # Load data if a data path is provided
        if data_path:
            self.data = pd.read_csv(data_path)
            self.save_original_data()
        else:
            self.data = None
            
        # Initialize evaluation results
        self.evaluation_results = {
            'sigma': pd.DataFrame(columns=['Model', 'Train_RMSE', 'Train_R2', 'Train_MAE', 
                                          'Val_RMSE', 'Val_R2', 'Val_MAE']),
            'epsilon': pd.DataFrame(columns=['Model', 'Train_RMSE', 'Train_R2', 'Train_MAE', 
                                            'Val_RMSE', 'Val_R2', 'Val_MAE'])
        }
        
        # Store prediction results for plotting
        self.predictions = {
            'sigma': {'train': {}, 'val': {}},
            'epsilon': {'train': {}, 'val': {}}
        }

    def load_data(self, data_path):
        """Load the dataset"""
        self.data = pd.read_csv(data_path)
        self.save_original_data()
        return self

    def save_original_data(self):
        """Save the original data to CSV"""
        if self.data is not None:
            self.data.to_csv(os.path.join(output_dir, 'original_data.csv'), index=False)
        return self

    def preprocess_data(self, target):
        """Preprocess and split the data"""
        if self.data is None:
            raise ValueError("Please load the data first")
            
        # Features and target variables
        X = self.data[['A', 'A_num', 'B', 'B_num']]
        y = self.data[target]
        
        # Split the training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        return X_train_scaled, X_val_scaled, y_train, y_val, X_train, X_val

    def define_models(self):
        """Define models and hyperparameter grids"""
        # Linear regression does not require hyperparameter tuning
        self.models['LinearRegression'] = {
            'model': LinearRegression(),
            'params': {}
        }
        
        # KRRmodel
        self.models['KRR'] = {
            'model': KernelRidge(),
            'params': {
                'alpha': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        }
        
        # GBRmodel
        self.models['GBR'] = {
            'model': GradientBoostingRegressor(random_state=self.random_state),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        # LightGBMmodel
        self.models['LightGBM'] = {
            'model': LGBMRegressor(random_state=self.random_state),
            'params': {
                'n_estimators': [200],
                'learning_rate': [ 0.1],
                'num_leaves': [31, 127],
                'max_depth': [3, ]
            }
        }
        
        # KNNmodel
        self.models['KNN'] = {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1是Manhattan distance, 2是Euclidean distance
            }
        }
        
        return self

    def train_and_evaluate(self):
        """training和evaluation所有model"""
        # 确保model已定义
        if not self.models:
            self.define_models()
            
        # Train and evaluate each target variable separately
        for target in ['sigma', 'epsilon']:
            print(f"\n===== Start training {target} 的predictionmodel =====")
            
            # 预processdata
            X_train, X_val, y_train, y_val, X_train_original, X_val_original = self.preprocess_data(target)
            
            # training每个model
            for name, model_info in self.models.items():
                print(f"\nTraining {name} model...")
                
                # 网格搜索超parameter
                if model_info['params']:  # for example果有超parameter需要调优
                    grid_search = GridSearchCV(
                        estimator=model_info['model'],
                        param_grid=model_info['params'],
                        cv=5,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    print(f"Best parameters: {grid_search.best_params_}")
                else:  # 线性回归不需要调优
                    best_model = model_info['model']
                    best_model.fit(X_train, y_train)
                
                # save最佳model
                self.best_models[(name, target)] = best_model
                joblib.dump(best_model, os.path.join(models_dir, f"{name}_{target}_model.pkl"))
                
                # Predict
                y_train_pred = best_model.predict(X_train)
                y_val_pred = best_model.predict(X_val)
                
                # savepredictionresults
                self.predictions[target]['train'][name] = {
                    'true': y_train,
                    'pred': y_train_pred,
                    'features': X_train_original
                }
                self.predictions[target]['val'][name] = {
                    'true': y_val,
                    'pred': y_val_pred,
                    'features': X_val_original
                }
                
                # 计算evaluation指标
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                
                # 存储evaluationresults
                new_row = pd.DataFrame({
                    'Model': [name],
                    'Train_RMSE': [train_rmse],
                    'Train_R2': [train_r2],
                    'Train_MAE': [train_mae],
                    'Val_RMSE': [val_rmse],
                    'Val_R2': [val_r2],
                    'Val_MAE': [val_mae]
                })
                self.evaluation_results[target] = pd.concat(
                    [self.evaluation_results[target], new_row], ignore_index=True
                )
                
                print(f"{name} model evaluation results:")
                print(f"training set - RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}, MAE: {train_mae:.4f}")
                print(f"validate集 - RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}, MAE: {val_mae:.4f}")
        
        # saveevaluationresults
        self.save_evaluation_results()
        return self

    def save_evaluation_results(self):
        """saveevaluationresults到CSV"""
        for target in ['sigma', 'epsilon']:
            self.evaluation_results[target].to_csv(
                os.path.join(output_dir, f'{target}_evaluation_results.csv'), index=False
            )
        return self

    def find_best_model(self):
        """找出每个目标变量的最佳model"""
        best_models = {}
        
        for target in ['sigma', 'epsilon']:
            # 按validate集R2降序排序, R2越高越好
            sorted_results = self.evaluation_results[target].sort_values('Val_R2', ascending=False)
            best_model_name = sorted_results.iloc[0]['Model']
            best_models[target] = best_model_name
            
            print(f"\n{target} prediction的最佳model是: {best_model_name}")
            print("性能指标:")
            print(sorted_results.iloc[0])
        
        return best_models

    def plot_predictions(self):
        """绘制predicted values与true values的散点图"""
        for target in ['sigma', 'epsilon']:
            for name in self.models.keys():
                plt.figure(figsize=(10, 6))
                
                # trainingdata
                train_true = self.predictions[target]['train'][name]['true']
                train_pred = self.predictions[target]['train'][name]['pred']
                plt.scatter(train_true, train_pred, alpha=0.5, label='Training_data', color='blue')
                
                # validatedata
                val_true = self.predictions[target]['val'][name]['true']
                val_pred = self.predictions[target]['val'][name]['pred']
                plt.scatter(val_true, val_pred, alpha=0.5, label='Validation_data', color='orange')
                
                # 理想线(y=x)
                min_val = min(min(train_true), min(val_true), min(train_pred), min(val_pred))
                max_val = max(max(train_true), max(val_true), max(train_pred), max(val_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.xlabel('True value')
                plt.ylabel('Prediction value')
                plt.title(f'{name}_{target}_predictions ')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # save图像
                plt.savefig(os.path.join(figures_dir, f'{name}_{target}_predictions.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # save散点图data到CSV
                train_data = pd.DataFrame({
                    'true values': train_true,
                    'predicted values': train_pred,
                    'datatype': 'trainingdata'
                })
                val_data = pd.DataFrame({
                    'true values': val_true,
                    'predicted values': val_pred,
                    'datatype': 'validatedata'
                })
                plot_data = pd.concat([train_data, val_data], ignore_index=True)
                plot_data.to_csv(os.path.join(output_dir, f'{name}_{target}_predictions.csv'), index=False)
        
        return self

    def save_scaler(self):
        """Save the scaler, 用于后续prediction"""
        joblib.dump(self.scaler, os.path.join(models_dir, 'scaler.pkl'))
        return self


def predict_from_csv(model_name, target, input_csv, output_csv=None):
    """
    Usingsave的model和CSVdata进行prediction
    
    parameter:
    model_name: model名称 (LinearRegression, KRR, GBR, LightGBM, KNN)
    target: prediction目标 (sigma 或 epsilon)
    input_csv: 包含inputfeatures的CSVfilepath (需要包含A, A_num, B, B_num列)
    output_csv: savepredictionresults的CSVfilepath, 若为None则不save
    """
    # loaddata
    input_data = pd.read_csv(input_csv)
    
    # 检查必要的列是否存在
    required_columns = ['A', 'A_num', 'B', 'B_num']
    if not set(required_columns).issubset(input_data.columns):
        missing = set(required_columns) - set(input_data.columns)
        raise ValueError(f"inputCSV缺少必要的列: {missing}")
    
    # Extractfeatures
    X = input_data[required_columns]
    
    # load标准化器和model
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    model = joblib.load(os.path.join(models_dir, f"{model_name}_{target}_model.pkl"))
    
    # 标准化features
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    # 准备results
    result = input_data.copy()
    result[f'prediction_{target}'] = predictions
    
    # saveresults(for example果指定)
    if output_csv:
        result.to_csv(output_csv, index=False)
        print(f"Prediction results saved to {output_csv}")
    
    return result


# Main function
def main(data_path):
    # initializemodel比较器
    comparator = ModelComparison(data_path=data_path)
    
    # 定义model
    comparator.define_models()
    
    # training和evaluationmodel
    comparator.train_and_evaluate()
    
    # 找出最佳model
    best_models = comparator.find_best_model()
    
    # Plot prediction scatter plots
    comparator.plot_predictions()
    
    # Save the scaler
    comparator.save_scaler()
    
    print("\n所有modeltraining、evaluation和可视化已完成！")
    print(f"Results are stored in {output_dir} directory")
    return comparator, best_models


if __name__ == "__main__":
    # Replace this with your actual data file path
    data_file_path = "energy_new1.csv"  # 用户需要替换为自己的datafilepath
    
    # 检查file是否存在
    if not os.path.exists(data_file_path):
        print(f"错误: 找不到datafile {data_file_path}")
        print("请检查filepath或在代码中修改data_file_path变量")
    else:
        main(data_file_path)


