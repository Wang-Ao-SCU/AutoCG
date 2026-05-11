import pandas as pd
import joblib
import argparse
import os

def batch_predict(sigma_model_path, epsilon_model_path, input_csv, output_csv):
    if not os.path.exists(sigma_model_path):
        raise FileNotFoundError(f'sigmamodelfile not found: {sigma_model_path}')
    if not os.path.exists(epsilon_model_path):
        raise FileNotFoundError(f'epsilonmodelfile not found: {epsilon_model_path}')
    model_sigma = joblib.load(sigma_model_path)
    model_epsilon = joblib.load(epsilon_model_path)
    df = pd.read_csv(input_csv)
    required_cols = ['OCO', 'heavy_atoms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Input CSV is missing columns: {missing_cols}, current column names: {df.columns.tolist()}')
    X = df[['OCO', 'heavy_atoms']]
    df['predicted_sigma'] = model_sigma.predict(X)
    df['predicted_epsilon'] = model_epsilon.predict(X)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch prediction of sigma and epsilon (input only requires the `OCO` column)')
    parser.add_argument('--sigma_model', type=str, default='./ML_W/model_output/sigma_model.pkl', help='Sigma model path(default: ./ML_W/model_output/sigma_model.pkl)')
    parser.add_argument('--epsilon_model', type=str, default='./ML_W/model_output/epsilon_model.pkl', help='Epsilon model path(default: ./ML_W/model_output/epsilon_model.pkl)')
    parser.add_argument('--input_csv', type=str, default='OCO.csv', help='Input CSV path (must contain the `OCO` column)')
    parser.add_argument('--output_csv', type=str, default='./ML_W/OCO_pre.csv', help='Output CSV path(default: batch_predictions.csv)')
    args = parser.parse_args()
    try:
        batch_predict(args.sigma_model, args.epsilon_model, args.input_csv, args.output_csv)
    except Exception as e:
        pass
