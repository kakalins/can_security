import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns


# Configurações globais
N_TRAINING_SAMPLES = 250000  
N_VAL_SAMPLES = 25000
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)

# Definindo o caminho dos arquivos
file_path = os.path.dirname(os.path.abspath(__file__)) + '/'

# file_path = 'C:/Users/ricardo.mota/OneDrive - SENAI-PE/Documentos/Doutorado/Disciplina Segurança com IA/can_security/dataset/'
file_list = ['Fuzzy_dataset.csv','RPM_dataset.csv','gear_dataset.csv']
#df_list = []

# --- Preparando o arquivo de log para o resumo ---
log_file_path = os.path.join(file_path, 'dataset/preprocessed/data_summary.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write("Resumo do Pré-processamento dos Datasets\n")
    log_file.write("=" * 40 + "\n")

for file in file_list:
    # Lendo o arquivo CSV
    df = pd.read_csv(file_path + 'dataset/' + file, header=None, sep=',', on_bad_lines='skip', low_memory=False)

    # Definindo o nome do ataque com base no nome do arquivo
    attack_name = file.split('_')[0]+"_attack"

    # Criando a coluna 'Attack' com base na coluna 11
    df['Attack'] = df[11].apply(lambda x: 'Benign' if x == 'R' else attack_name) 

    # Definindo os nomes das colunas
    column_names = ['Timestamp', 'ID', 'DLC', 'Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label', 'Attack']
    df.columns = column_names

    # Filtrando apenas os registros com Label 'R' ou 'T'
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
        # Convertendo valores hexadecimais para inteiros
        def hex_to_int(hex_str):
            try:
                return int(str(hex_str), 16)
            except (ValueError, TypeError):
                return np.nan # Retorna NaN se a conversão falhar

        df[col] = df[col].apply(hex_to_int)

    df_columns_isfinite = np.isfinite(df.drop(['Label', 'Attack'], axis='columns')).all(axis=0)
    df_columns_isfinite[df_columns_isfinite == False]

    df_rows_isfinite = np.isfinite(df.drop(['Label', 'Attack'], axis='columns')).all(axis=1)
    inf_indexes = df_rows_isfinite[df_rows_isfinite == False].index
    df.iloc[inf_indexes][['Label','Attack']]

    print(df.info())
    
    
    df_train = df.query('Label == "R"').sample(n=N_TRAINING_SAMPLES, random_state=RANDOM_SEED)
    df_val_test = df.drop(df_train.index)

    df_train = df_train.reset_index(drop=True)
    df_val_test = df_val_test.reset_index(drop=True)

    X_val_attack = df_val_test.query('Label != "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)

    df_val_test = df_val_test.drop(X_val_attack.index)

    df_val_test = df_val_test.reset_index(drop=True)
    X_val_attack = X_val_attack.reset_index(drop=True)

    # --- Salvando o DataFrame para uso posterior ---

    # Define o caminho e o nome do arquivo de saída
    output_file = [''] * 3  # Lista para armazenar os nomes dos arquivos de saída
    output_file[0] = os.path.join(file_path + 'dataset/preprocessed/', file.split('_')[0] + '_train.pkl')
    output_file[1] = os.path.join(file_path + 'dataset/preprocessed/', file.split('_')[0] + '_val_attack.pkl')
    output_file[2] = os.path.join(file_path + 'dataset/preprocessed/', file.split('_')[0] + '_test.pkl')

    # Salva o DataFrame em formato Pickle
    df_train.to_pickle(output_file[0])
    X_val_attack.to_pickle(output_file[1])
    df_val_test.to_pickle(output_file[2])

    # --- Registrando as contagens no arquivo de log ---
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\nArquivo de origem: {file}\n")
        log_file.write("-" * 30 + "\n")

        # Contagens para o arquivo de treino
        log_file.write(f"Arquivo gerado: {os.path.basename(output_file[0])}\n")
        log_file.write(f"  - Linhas benignas ('R'): {len(df_train)}\n")
        log_file.write(f"  - Linhas maliciosas ('T'): 0\n")

        # Contagens para o arquivo de validação de ataques
        log_file.write(f"Arquivo gerado: {os.path.basename(output_file[1])}\n")
        log_file.write(f"  - Linhas benignas ('R'): 0\n")
        log_file.write(f"  - Linhas maliciosas ('T'): {len(X_val_attack)}\n")

        # Contagens para o arquivo de teste
        log_file.write(f"Arquivo gerado: {os.path.basename(output_file[2])}\n")
        log_file.write(f"  - Linhas benignas ('R'): {df_val_test['Label'].value_counts().get('R', 0)}\n")
        log_file.write(f"  - Linhas maliciosas ('T'): {df_val_test['Label'].value_counts().get('T', 0)}\n")

    print(f"\nDataFrame pré-processado foi salvo com sucesso em: {output_file}")
    print("-" * 30)
    
    del df, df_val_test , df_train, X_val_attack

print("\nProcessamento concluído para todos os arquivos.")
print("-" * 30)
print(f"Resumo das contagens salvo em: {log_file_path}")
