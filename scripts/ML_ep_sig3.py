import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
output_dir = 'model_results_test3'
os.makedirs(output_dir, exist_ok=True)
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)
models_dir = os.path.join(output_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

class ModelComparison:

    def __init__(self, data_path=None, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.scaler = StandardScaler()
        if data_path:
            self.data = pd.read_csv(data_path)
            self.save_original_data()
        else:
            self.data = None
        self.evaluation_results = {'sigma': pd.DataFrame(columns=['Model', 'Train_RMSE', 'Train_R2', 'Train_MAE', 'Val_RMSE', 'Val_R2', 'Val_MAE']), 'epsilon': pd.DataFrame(columns=['Model', 'Train_RMSE', 'Train_R2', 'Train_MAE', 'Val_RMSE', 'Val_R2', 'Val_MAE'])}
        self.predictions = {'sigma': {'train': {}, 'val': {}}, 'epsilon': {'train': {}, 'val': {}}}

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path)
        self.save_original_data()
        return self

    def save_original_data(self):
        if self.data is not None:
            self.data.to_csv(os.path.join(output_dir, 'original_data.csv'), index=False)
        return self

    def preprocess_data(self, target):
        if self.data is None:
            raise ValueError('Please load the data first')
        X = self.data[['A', 'A_num', 'B', 'B_num']]
        y = self.data[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        return (X_train_scaled, X_val_scaled, y_train, y_val, X_train, X_val)

    def define_models(self):
        self.models['LinearRegression'] = {'model': LinearRegression(), 'params': {}}
        self.models['KRR'] = {'model': KernelRidge(), 'params': {'alpha': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto', 0.1, 1]}}
        self.models['GBR'] = {'model': GradientBoostingRegressor(random_state=self.random_state), 'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}}
        self.models['LightGBM'] = {'model': LGBMRegressor(random_state=self.random_state), 'params': {'n_estimators': [200], 'learning_rate': [0.1], 'num_leaves': [31, 127], 'max_depth': [3]}}
        self.models['KNN'] = {'model': KNeighborsRegressor(), 'params': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'p': [1, 2]}}
        return self

    def train_and_evaluate(self):
        if not self.models:
            self.define_models()
        for target in ['sigma', 'epsilon']:
            X_train, X_val, y_train, y_val, X_train_original, X_val_original = self.preprocess_data(target)
            for name, model_info in self.models.items():
                if model_info['params']:
                    grid_search = GridSearchCV(estimator=model_info['model'], param_grid=model_info['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                else:
                    best_model = model_info['model']
                    best_model.fit(X_train, y_train)
                self.best_models[name, target] = best_model
                joblib.dump(best_model, os.path.join(models_dir, f'{name}_{target}_model.pkl'))
                y_train_pred = best_model.predict(X_train)
                y_val_pred = best_model.predict(X_val)
                self.predictions[target]['train'][name] = {'true': y_train, 'pred': y_train_pred, 'features': X_train_original}
                self.predictions[target]['val'][name] = {'true': y_val, 'pred': y_val_pred, 'features': X_val_original}
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                new_row = pd.DataFrame({'Model': [name], 'Train_RMSE': [train_rmse], 'Train_R2': [train_r2], 'Train_MAE': [train_mae], 'Val_RMSE': [val_rmse], 'Val_R2': [val_r2], 'Val_MAE': [val_mae]})
                self.evaluation_results[target] = pd.concat([self.evaluation_results[target], new_row], ignore_index=True)
        self.save_evaluation_results()
        return self

    def save_evaluation_results(self):
        for target in ['sigma', 'epsilon']:
            self.evaluation_results[target].to_csv(os.path.join(output_dir, f'{target}_evaluation_results.csv'), index=False)
        return self

    def find_best_model(self):
        best_models = {}
        for target in ['sigma', 'epsilon']:
            sorted_results = self.evaluation_results[target].sort_values('Val_R2', ascending=False)
            best_model_name = sorted_results.iloc[0]['Model']
            best_models[target] = best_model_name
        return best_models

    def plot_predictions(self):
        for target in ['sigma', 'epsilon']:
            for name in self.models.keys():
                plt.figure(figsize=(10, 6))
                train_true = self.predictions[target]['train'][name]['true']
                train_pred = self.predictions[target]['train'][name]['pred']
                plt.scatter(train_true, train_pred, alpha=0.5, label='Training_data', color='blue')
                val_true = self.predictions[target]['val'][name]['true']
                val_pred = self.predictions[target]['val'][name]['pred']
                plt.scatter(val_true, val_pred, alpha=0.5, label='Validation_data', color='orange')
                min_val = min(min(train_true), min(val_true), min(train_pred), min(val_pred))
                max_val = max(max(train_true), max(val_true), max(train_pred), max(val_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                plt.xlabel('True value')
                plt.ylabel('Prediction value')
                plt.title(f'{name}_{target}_predictions ')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(figures_dir, f'{name}_{target}_predictions.png'), dpi=300, bbox_inches='tight')
                plt.close()
                train_data = pd.DataFrame({'true values': train_true, 'predicted values': train_pred, 'datatype': 'trainingdata'})
                val_data = pd.DataFrame({'true values': val_true, 'predicted values': val_pred, 'datatype': 'validatedata'})
                plot_data = pd.concat([train_data, val_data], ignore_index=True)
                plot_data.to_csv(os.path.join(output_dir, f'{name}_{target}_predictions.csv'), index=False)
        return self

    def save_scaler(self):
        joblib.dump(self.scaler, os.path.join(models_dir, 'scaler.pkl'))
        return self

def predict_from_csv(model_name, target, input_csv, output_csv=None):
    input_data = pd.read_csv(input_csv)
    required_columns = ['A', 'A_num', 'B', 'B_num']
    if not set(required_columns).issubset(input_data.columns):
        missing = set(required_columns) - set(input_data.columns)
        raise ValueError(f'inputCSV缺少必要的列: {missing}')
    X = input_data[required_columns]
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    model = joblib.load(os.path.join(models_dir, f'{model_name}_{target}_model.pkl'))
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    result = input_data.copy()
    result[f'prediction_{target}'] = predictions
    if output_csv:
        result.to_csv(output_csv, index=False)
    return result

def main(data_path):
    comparator = ModelComparison(data_path=data_path)
    comparator.define_models()
    comparator.train_and_evaluate()
    best_models = comparator.find_best_model()
    comparator.plot_predictions()
    comparator.save_scaler()
    return (comparator, best_models)
if __name__ == '__main__':
    data_file_path = 'energy_new1.csv'
    if not os.path.exists(data_file_path):
        pass
    else:
        main(data_file_path)
