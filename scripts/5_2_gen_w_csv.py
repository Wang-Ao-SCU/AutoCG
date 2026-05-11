import pandas as pd

def process_bead_energy_data(input_file, output_file):
    df = pd.read_csv(input_file)
    data1 = df[['bead_type1', 'G1', 'heavy_atoms1']].copy()
    data1.columns = ['bead_type', 'OCO', 'heavy_atoms']
    data2 = df[['bead_type2', 'G2', 'heavy_atoms2']].copy()
    data2.columns = ['bead_type', 'OCO', 'heavy_atoms']
    combined_data = pd.concat([data1, data2], ignore_index=True)
    unique_data = combined_data.groupby(['bead_type', 'heavy_atoms'])['OCO'].mean().reset_index()
    unique_data = unique_data[['bead_type', 'heavy_atoms', 'OCO']]
    unique_data.to_csv(output_file, index=False, encoding='utf-8')
    return unique_data
input_csv = './G_predictions/enhanced_gbr_predictions.csv'
output_csv = './OCO.csv'
try:
    result_df = process_bead_energy_data(input_csv, output_csv)
except FileNotFoundError:
    pass
