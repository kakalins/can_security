import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from tqdm.notebook import tqdm
# from sklearn.cluster import KMeans

# from sklearn.metrics import silhouette_score
# from sklearn.metrics import roc_curve, roc_auc_score
# from sklearn.metrics import confusion_matrix

# from tqdm.notebook import tqdm
# import plotly.express as px
# import plotly.graph_objects as go 
# from plotly.subplots import make_subplots

RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)

# Carregando o arquivo DoS_dataset.csv do Google Drive
# Certifique-se de que o arquivo está no caminho especificado
file_path = 'C:/Users/ricardo.mota/OneDrive - SENAI-PE/Documentos/Doutorado/Disciplina Segurança com IA/can_security/dataset/'
file_list = ['Fuzzy_dataset.csv','RPM_dataset.csv','gear_dataset.csv']
#df_list = []
for file in file_list:
    # Use sep=',' to explicitly set the delimiter to comma and on_bad_lines='skip' to skip problematic lines
    df = pd.read_csv(file_path + file, header=None, sep=',', on_bad_lines='skip', low_memory=False)

    # Extract the attack name from the filename
    attack_name = file.split('_')[0]+"_attack"

    # Add the 'Attack' column based on the 'Label'
    df['Attack'] = df[11].apply(lambda x: 'Benign' if x == 'R' else attack_name) # Assuming 'Label' is the 12th column (index 11)

    #df_list.append(df_aux)
#df = pd.concat(df_list, ignore_index=True)

    # You can replace these generic names with your actual column names
    column_names = ['Timestamp', 'ID', 'DLC', 'Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label', 'Attack']
    df.columns = column_names

    # Filtering rows where 'Label' is 'R' or 'T'
    initial_len = df.shape[0]
    df = df[df['Label'].isin(['R', 'T'])]
    print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros onde o Label não era R ou T')

    df.columns = df.columns.str.strip()

    df[df.duplicated()]
    # Descartando duplicadas
    initial_len = df.shape[0]
    df = df.drop_duplicates()
    print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartadas {initial_len - df.shape[0]} duplicadas')

    df.columns[df.isna().any(axis=0)]
    df.isna().sum()[df.isna().sum() > 0]

    # Descartando registros com valores NaN/Null/NA
    initial_len = df.shape[0]
    df = df.dropna()
    print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')

    df = df.reset_index(drop=True)

    cols = ['ID', 'DLC', 'Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7']

    for col in cols:
        # Convert hexadecimal strings to integers, coercing errors to NaN
        # Use a custom function to handle potential errors during conversion
        def hex_to_int(hex_str):
            try:
                return int(str(hex_str), 16)
            except (ValueError, TypeError):
                return np.nan # Convert to NaN if conversion fails

        df[col] = df[col].apply(hex_to_int)

    df_columns_isfinite = np.isfinite(df.drop(['Label', 'Attack'], axis='columns')).all(axis=0)
    df_columns_isfinite[df_columns_isfinite == False]

    df_rows_isfinite = np.isfinite(df.drop(['Label', 'Attack'], axis='columns')).all(axis=1)
    inf_indexes = df_rows_isfinite[df_rows_isfinite == False].index
    df.iloc[inf_indexes][['Label','Attack']]

    print(df.info())

    # --- Salvando o DataFrame para uso posterior ---

    # Define o caminho e o nome do arquivo de saída
    output_file = os.path.join(file_path, file.split('_')[0] + '.pkl')

    # Salva o DataFrame em formato Pickle
    df.to_pickle(output_file)

    print(f"\nDataFrame pré-processado foi salvo com sucesso em: {output_file}")

# Para carregar o DataFrame em outro script, você pode usar:
# df_carregado = pd.read_pickle('{output_file}')
# print(df_carregado.head())
