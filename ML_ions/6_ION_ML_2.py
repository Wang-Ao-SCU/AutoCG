import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import matplotlib.pyplot as plt
import os
import argparse
import joblib

# 强制Using无GUI后端, 避免Qt报错
plt.switch_backend('Agg')
# 设置字体(优先无依赖字体, 避免中文问题)
plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

def load_and_clean_data(file_path):
    """loaddata、去重、分割为training set和validate集(拆分两个目标变量)"""
    try:
        # 1. readCSVfile
        df = pd.read_csv(file_path)
        print(f"原始data形状: {df.shape}")
        print(f"原始datacolumn names: {df.columns.tolist()}")
        
        # 2. data去重
        df_clean = df.drop_duplicates().reset_index(drop=True)
        print(f"去重后data形状: {df_clean.shape}")
        print(f"移除重复sample count: {df.shape[0] - df_clean.shape[0]}")
        
        # 3. validate必要列
        required_columns = ['OCO', 'heavy_atoms', 'sigma', 'epsilon']
        missing_cols = [col for col in required_columns if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"CSVMissing required columns: {missing_cols}(需包含'OCO'、'heavy_atoms'、'sigma'、'epsilon')")
        
        # 4. 筛选核心列并删除缺失值
        df_core = df_clean[required_columns].dropna().reset_index(drop=True)
        print(f"删除缺失值后data形状: {df_core.shape}")
        
        # 5. 拆分features和两个目标变量
        X = df_core[['OCO', 'heavy_atoms']]
        y_sigma = df_core['sigma']  # sigma单目标
        y_epsilon = df_core['epsilon']  # epsilon单目标
        
        # 6. 切分training set和validate集(两个目标变量用相同的分割方式)
        X_train, X_val, y_sigma_train, y_sigma_val = train_test_split(
            X, y_sigma, test_size=0.2, random_state=42, shuffle=True
        )
        _, _, y_epsilon_train, y_epsilon_val = train_test_split(
            X, y_epsilon, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"\ndata准备完成: ")
        print(f"training set: {X_train.shape[0]}个样本, validate集: {X_val.shape[0]}个样本")
        return X_train, X_val, y_sigma_train, y_sigma_val, y_epsilon_train, y_epsilon_val
    
    except Exception as e:
        print(f"dataprocess失败: {str(e)}")
        raise

def objective_sigma(trial, X_train, y_train, X_val, y_val):
    """sigma目标的超parameter优化函数(兼容旧版scikit-learn)"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000, step=20),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': -1
    }
    
    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    # 兼容旧版scikit-learn, 手动计算RMSE
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def objective_epsilon(trial, X_train, y_train, X_val, y_val):
    """epsilon目标的超parameter优化函数(兼容旧版scikit-learn)"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000, step=20),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': -1
    }
    
    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    # 兼容旧版scikit-learn, 手动计算RMSE
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def optimize_hyperparameters(X_train, X_val, y_sigma_train, y_sigma_val, y_epsilon_train, y_epsilon_val, n_trials=50):
    """分别优化sigma和epsilon的超parameter, 返回两个最优model"""
    # 优化sigmamodel
    print("\n===== 开始sigmamodel超parameter优化 =====")
    study_sigma = optuna.create_study(direction='minimize', study_name='sigma_pred')
    study_sigma.optimize(
        lambda trial: objective_sigma(trial, X_train, y_sigma_train, X_val, y_sigma_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print(f"sigma最优超parameter: {study_sigma.best_params}")
    print(f"sigma最优validate集RMSE: {study_sigma.best_value:.4f}")
    
    # trainingsigma最优model
    best_model_sigma = LGBMRegressor(**study_sigma.best_params)
    best_model_sigma.fit(
        X_train, y_sigma_train,
        eval_set=[(X_val, y_sigma_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 优化epsilonmodel
    print("\n===== 开始epsilonmodel超parameter优化 =====")
    study_epsilon = optuna.create_study(direction='minimize', study_name='epsilon_pred')
    study_epsilon.optimize(
        lambda trial: objective_epsilon(trial, X_train, y_epsilon_train, X_val, y_epsilon_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print(f"epsilon最优超parameter: {study_epsilon.best_params}")
    print(f"epsilon最优validate集RMSE: {study_epsilon.best_value:.4f}")
    
    # trainingepsilon最优model
    best_model_epsilon = LGBMRegressor(**study_epsilon.best_params)
    best_model_epsilon.fit(
        X_train, y_epsilon_train,
        eval_set=[(X_val, y_epsilon_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    return (study_sigma.best_params, best_model_sigma), (study_epsilon.best_params, best_model_epsilon)

def evaluate_models(model_sigma, model_epsilon, X_train, X_val, y_sigma_train, y_sigma_val, y_epsilon_train, y_epsilon_val):
    """evaluation两个model的性能, 返回统一指标表格(兼容旧版scikit-learn)"""
    # Predict
    sigma_train_pred = model_sigma.predict(X_train)
    sigma_val_pred = model_sigma.predict(X_val)
    epsilon_train_pred = model_epsilon.predict(X_train)
    epsilon_val_pred = model_epsilon.predict(X_val)
    
    # 计算指标的辅助函数(手动计算RMSE)
    def calc_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)  # 兼容旧版, 手动开平方
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return [rmse, mae, r2]
    
    # 计算各指标
    sigma_train_metrics = calc_metrics(y_sigma_train, sigma_train_pred)
    sigma_val_metrics = calc_metrics(y_sigma_val, sigma_val_pred)
    epsilon_train_metrics = calc_metrics(y_epsilon_train, epsilon_train_pred)
    epsilon_val_metrics = calc_metrics(y_epsilon_val, epsilon_val_pred)
    
    # 组织成表格
    metrics_df = pd.DataFrame({
        '指标': ['RMSE', 'MAE', 'R²'],
        'sigma_training set': sigma_train_metrics,
        'sigma_validate集': sigma_val_metrics,
        'epsilon_training set': epsilon_train_metrics,
        'epsilon_validate集': epsilon_val_metrics
    })
    
    # 额外返回prediction原始data(用于后续save, 避免图片问题丢失)
    pred_data = {
        'sigma_train_true': y_sigma_train.values,
        'sigma_train_pred': sigma_train_pred,
        'sigma_val_true': y_sigma_val.values,
        'sigma_val_pred': sigma_val_pred,
        'epsilon_train_true': y_epsilon_train.values,
        'epsilon_train_pred': epsilon_train_pred,
        'epsilon_val_true': y_epsilon_val.values,
        'epsilon_val_pred': epsilon_val_pred
    }
    
    return metrics_df.round(4), pred_data

def save_core_results(model_sigma, model_epsilon, params_sigma, params_epsilon, metrics_df, pred_data, output_dir):
    """save核心results(model、超parameter、evaluation指标、prediction原始data)- 优先执行"""
    # 确保outputdirectory存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. savemodel
    sigma_model_path = os.path.join(output_dir, 'sigma_model.pkl')
    epsilon_model_path = os.path.join(output_dir, 'epsilon_model.pkl')
    joblib.dump(model_sigma, sigma_model_path)
    joblib.dump(model_epsilon, epsilon_model_path)
    print(f"✅ sigmaModel saved to: {sigma_model_path}")
    print(f"✅ epsilonModel saved to: {epsilon_model_path}")
    
    # 2. save超parameter
    with open(os.path.join(output_dir, 'sigma_hyperparameters.txt'), 'w', encoding='utf-8') as f:
        for k, v in params_sigma.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(output_dir, 'epsilon_hyperparameters.txt'), 'w', encoding='utf-8') as f:
        for k, v in params_epsilon.items():
            f.write(f"{k}: {v}\n")
    print("✅ 超parameter已save至对应file")
    
    # 3. saveevaluation指标
    metrics_path = os.path.join(output_dir, 'model_performance.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"✅ modelevaluation指标已save至: {metrics_path}")
    
    # 4. saveprediction原始data(避免图片绘制失败丢失data)
    pred_data_df = pd.DataFrame(pred_data)
    pred_data_path = os.path.join(output_dir, 'prediction_raw_data.csv')
    pred_data_df.to_csv(pred_data_path, index=False, encoding='utf-8-sig')
    print(f"✅ prediction原始data已save至: {pred_data_path}")

def plot_predictions(pred_data, save_path):
    """绘制散点图(容错process: directory不存在自动创建, 绘制失败不中断workflow)"""
    try:
        # 确保savedirectory存在(core fix: 自动创建directory)
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"📁 自动创建图片savedirectory: {save_dir}")
        
        # 绘制图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Features(OCO, Heavy_Atoms)→Sigma/Epsilon Prediction Results', fontsize=16, fontweight='bold')  # 用英文避免字体问题
        
        # 定义绘图函数
        def plot_scatter(ax, y_true, y_pred, title, xlabel, ylabel):
            ax.scatter(y_true, y_pred, alpha=0.6, s=30, color='#2E86AB')
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 绘制4个子图(用英文标题避免字体依赖)
        plot_scatter(axes[0,0], pred_data['sigma_train_true'], pred_data['sigma_train_pred'], 
                     'Sigma - Train Set', 'True Value', 'Predicted Value')
        plot_scatter(axes[0,1], pred_data['sigma_val_true'], pred_data['sigma_val_pred'], 
                     'Sigma - Validation Set', 'True Value', 'Predicted Value')
        plot_scatter(axes[1,0], pred_data['epsilon_train_true'], pred_data['epsilon_train_pred'], 
                     'Epsilon - Train Set', 'True Value', 'Predicted Value')
        plot_scatter(axes[1,1], pred_data['epsilon_val_true'], pred_data['epsilon_val_pred'], 
                     'Epsilon - Validation Set', 'True Value', 'Predicted Value')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
        print(f"✅ prediction对比图已save至: {save_path}")
        plt.close(fig)
    
    except Exception as e:
        # 图片绘制失败不中断workflow, 仅警告
        print(f"⚠️  图片绘制/save失败: {str(e)}")
        print(f"⚠️  已skip图片save, 核心results(model、data)不受影响")

def generate_prediction_script(output_dir):
    """generate批量prediction脚本(同时调用两个model)"""
    script_content = '''import pandas as pd
import joblib
import argparse
import os

def batch_predict(sigma_model_path, epsilon_model_path, input_csv, output_csv):
    """
    批量prediction函数: read含OCO和heavy_atoms列的CSV, outputprediction的sigma/epsilon
    """
    # load两个model
    if not os.path.exists(sigma_model_path):
        raise FileNotFoundError(f"sigmamodelfile不存在: {sigma_model_path}")
    if not os.path.exists(epsilon_model_path):
        raise FileNotFoundError(f"epsilonmodelfile不存在: {epsilon_model_path}")
    model_sigma = joblib.load(sigma_model_path)
    model_epsilon = joblib.load(epsilon_model_path)
    
    # Read input data
    df = pd.read_csv(input_csv)
    required_features = ['OCO', 'heavy_atoms']
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"inputCSV缺少列: {missing}, 当前column names: {df.columns.tolist()}")
    
    # 执行prediction
    X = df[['OCO', 'heavy_atoms']]
    df['predicted_sigma'] = model_sigma.predict(X)
    df['predicted_epsilon'] = model_epsilon.predict(X)
    
    # saveresults
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"prediction完成！results已save至: {output_csv}")
    print(f"共prediction {len(df)} 个样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量predictionsigma和epsilon(input需OCO和heavy_atoms列)')
    parser.add_argument('--sigma_model', type=str, default='model_output/sigma_model.pkl',
                      help='sigmamodelpath(默认: model_output/sigma_model.pkl)')
    parser.add_argument('--epsilon_model', type=str, default='model_output/epsilon_model.pkl',
                      help='epsilonmodelpath(默认: model_output/epsilon_model.pkl)')
    parser.add_argument('--input_csv', type=str, required=True,
                      help='inputCSVfilepath(必须包含"OCO"和"heavy_atoms"列)')
    parser.add_argument('--output_csv', type=str, default='batch_predictions.csv',
                      help='outputCSVfilepath(默认: batch_predictions.csv)')
    
    args = parser.parse_args()
    try:
        batch_predict(args.sigma_model, args.epsilon_model, args.input_csv, args.output_csv)
    except Exception as e:
        print(f"prediction失败: {str(e)}")
'''
    
    # 确保脚本savedirectory存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    script_path = os.path.join(output_dir, 'batch_predict.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    print(f"✅ 批量prediction脚本已generate至: {script_path}")

def main(data_path, n_trials=50, output_dir='model_output'):
    """主workflow: dataprocess→trainingmodel→evaluation→优先save核心results→绘制图片→generate脚本"""
    print("="*50)
    print("开始 Features(OCO, Heavy_Atoms)→Sigma/Epsilon predictionmodeltrainingworkflow")
    print("="*50)
    
    # 1. dataload与清洗(拆分两个目标变量)
    X_train, X_val, y_sigma_train, y_sigma_val, y_epsilon_train, y_epsilon_val = load_and_clean_data(data_path)
    
    # 2. 超parameter优化与modeltraining(trained separatelysigma和epsilonmodel)
    (params_sigma, model_sigma), (params_epsilon, model_epsilon) = optimize_hyperparameters(
        X_train, X_val, y_sigma_train, y_sigma_val, y_epsilon_train, y_epsilon_val, n_trials
    )
    
    # 3. modelevaluation(返回指标表格和prediction原始data)
    print("\n" + "="*30)
    print("model evaluation results")
    print("="*30)
    metrics_df, pred_data = evaluate_models(
        model_sigma, model_epsilon, X_train, X_val,
        y_sigma_train, y_sigma_val, y_epsilon_train, y_epsilon_val
    )
    print(metrics_df)
    
    # 4. 优先save核心results(model、超参、指标、predictiondata)- 关bond调整
    print("\n" + "="*30)
    print("save核心trainingresults(不受图片影响)")
    print("="*30)
    save_core_results(model_sigma, model_epsilon, params_sigma, params_epsilon, metrics_df, pred_data, output_dir)
    
    # 5. 绘制prediction对比图(容错process, 失败不影响核心results)
    print("\n" + "="*30)
    print("尝试generateprediction对比图")
    print("="*30)
    plot_path = os.path.join(output_dir, 'predictions_vs_actuals.png')
    plot_predictions(pred_data, plot_path)
    
    # 6. generate批量prediction脚本
    print("\n" + "="*30)
    print("generate批量prediction脚本")
    print("="*30)
    generate_prediction_script(output_dir)
    
    print("\n" + "="*50)
    print("所有workflow完成！核心outputfilesave在: {}".format(output_dir))
    print("⚠️  若图片save失败, 核心功能(modeltraining、批量prediction)不受影响")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于LightGBM的Features(OCO, Heavy_Atoms)→Sigma/Epsilonpredictionmodel(双单目标model)')
    parser.add_argument('--data_path', type=str, required=True,
                      help='trainingdataCSVpath(必须含OCO、heavy_atoms、sigma、epsilon四列)')
    parser.add_argument('--n_trials', type=int, default=50,
                      help='每个model的超parameter优化试验次数(默认50)')
    parser.add_argument('--output_dir', type=str, default='model_output',
                      help='outputdirectory(默认model_output)')
    
    args = parser.parse_args()
    main(args.data_path, args.n_trials, args.output_dir)