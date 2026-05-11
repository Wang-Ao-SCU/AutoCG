import pandas as pd
import os
import joblib
models_dir = './ML2/model_results_test3/models/'

def predict_sigma_epsilon(input_csv, output_dir='pair_predict_results'):
    os.makedirs(output_dir, exist_ok=True)
    input_data = pd.read_csv(input_csv)
    required_columns = ['heavy_atoms1', 'heavy_atoms2', 'G1', 'G2']
    if not set(required_columns).issubset(input_data.columns):
        missing = set(required_columns) - set(input_data.columns)
        raise ValueError(f'Input CSV is missing required columns: {missing}')
    features = pd.DataFrame({'A': input_data['G1'], 'A_num': input_data['heavy_atoms1'], 'B': input_data['G2'], 'B_num': input_data['heavy_atoms2']})
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    features_scaled = scaler.transform(features)
    model_names = ['LinearRegression', 'KRR', 'GBR', 'KNN']
    for model_name in model_names:
        sigma_model = joblib.load(os.path.join(models_dir, f'{model_name}_sigma_model.pkl'))
        epsilon_model = joblib.load(os.path.join(models_dir, f'{model_name}_epsilon_model.pkl'))
        sigma_pred = sigma_model.predict(features_scaled)
        epsilon_pred = epsilon_model.predict(features_scaled)
        result = input_data.copy()
        result[f'predicted_sigma'] = sigma_pred
        result[f'predicted_epsilon'] = epsilon_pred
        output_file = os.path.join(output_dir, f'{model_name}_predictions.csv')
        result.to_csv(output_file, index=False)
if __name__ == '__main__':
    input_csv_path = './G_predictions/enhanced_gbr_predictions.csv'
    predict_sigma_epsilon(input_csv_path)
