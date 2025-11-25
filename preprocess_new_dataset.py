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

dataset_path = 'dataset/UAVCAN/UAVCAN_Attack_dataset/'
file_list = ['type1_label.bin','type2_label.bin','type3_label.bin','type4_label.bin','type5_label.bin','type6_label.bin','type7_label.bin','type8_label.bin','type9_label.bin','type10_label.bin']

# --- Preparando o arquivo de log para o resumo ---
log_dir = os.path.join(file_path, 'dataset/preprocessed/')
log_file_path = os.path.join(log_dir, 'new_dataset_summary.txt')

# Garante que o diretório de log exista antes de tentar escrever o arquivo
os.makedirs(log_dir, exist_ok=True)

with open(log_file_path, 'w') as log_file:
    log_file.write("Resumo do Pré-processamento dos Datasets\n")
    log_file.write("=" * 40 + "\n")

processed_dfs = [] # 1. Inicializa uma lista para guardar os DataFrames processados

for file in file_list:
    print(f"Processando arquivo: {file}...")
    # Lendo o arquivo e criando um DataFrame temporário
    with open(os.path.join(file_path + dataset_path, file), 'rb') as f:
        data = f.read()
    df_temp = pd.DataFrame([line.split() for line in data.decode('utf-8').split('\n') if line])
    
    df_temp = df_temp.drop(columns=[2,4])  # 2. Remove as colunas 2 e 4 do arquivo atual
    processed_dfs.append(df_temp)         # 3. Adiciona o DataFrame processado à lista

# 4. Concatena todos os DataFrames da lista em um único DataFrame
df = pd.concat(processed_dfs, ignore_index=True)
    
# Definindo os nomes das colunas
column_names = ['Label', 'Timestamp', 'ID', 'Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7']
df.columns = column_names

# Removendo os parênteses da coluna 'Timestamp'
df['Timestamp'] = df['Timestamp'].str.strip('()')

# Substituindo valores None (NaN) por '00' em todo o DataFrame
df = df.fillna('00')

# Convertendo a coluna 'Timestamp' de object para float para permitir cálculos
df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')




#     df.columns = df.columns.str.strip()

#     df[df.duplicated()]
#     # Descartando duplicadas
#     initial_len = df.shape[0]
#     df = df.drop_duplicates()
#     print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartadas {initial_len - df.shape[0]} duplicadas')

#     df.columns[df.isna().any(axis=0)]
#     df.isna().sum()[df.isna().sum() > 0]

#     # Descartando registros com valores NaN/Null/NA
#     initial_len = df.shape[0]
#     df = df.dropna()
#     print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')

#     df = df.reset_index(drop=True)

cols = ['ID', 'Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7']

for col in cols:
    # Convertendo valores hexadecimais para inteiros
    def hex_to_int(hex_str):
        try:
            return int(str(hex_str), 16)
        except (ValueError, TypeError):
            return np.nan # Retorna NaN se a conversão falhar

    df[col] = df[col].apply(hex_to_int)

#     df_columns_isfinite = np.isfinite(df.drop(['Label', 'Attack'], axis='columns')).all(axis=0)
#     df_columns_isfinite[df_columns_isfinite == False]

#     df_rows_isfinite = np.isfinite(df.drop(['Label', 'Attack'], axis='columns')).all(axis=1)
#     inf_indexes = df_rows_isfinite[df_rows_isfinite == False].index
#     df.iloc[inf_indexes][['Label','Attack']]

#     print(df.info())


# Separar amostras normais e de ataque
df_normal = df.query('Label == "Normal"')
# df_attack = df.query('Label != "Normal"')

# Determinar o número de amostras de treino, garantindo que não exceda o disponível
n_train_samples = min(len(df_normal), N_TRAINING_SAMPLES)
df_train = df_normal.sample(n=n_train_samples, random_state=RANDOM_SEED)

del df_normal, n_train_samples

df_val_test = df.drop(df_train.index)

df_train = df_train.reset_index(drop=True)
df_val_test = df_val_test.reset_index(drop=True)

X_val_attack = df_val_test.query('Label != "Normal"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)

df_val_test = df_val_test.drop(X_val_attack.index)

df_val_test = df_val_test.reset_index(drop=True)
X_val_attack = X_val_attack.reset_index(drop=True)

# --- Salvando o DataFrame para uso posterior ---

# Define o caminho e o nome do arquivo de saída
output_file = os.path.join(file_path + 'dataset/preprocessed/', file.split('_')[0] + '_train.pkl')
output_file = (output_file,
                output_file.replace('_train.pkl', '_val_attack.pkl'),
                output_file.replace('_train.pkl', '_val_test.pkl'))

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
    log_file.write(f"  - Linhas benignas: {len(df_train)}\n")
    log_file.write(f"  - Linhas maliciosas: 0\n")

    # Contagens para o arquivo de validação de ataques
    log_file.write(f"Arquivo gerado: {os.path.basename(output_file[1])}\n")
    log_file.write(f"  - Linhas benignas: 0\n")
    log_file.write(f"  - Linhas maliciosas: {len(X_val_attack)}\n")

    # Contagens para o arquivo de teste
    log_file.write(f"Arquivo gerado: {os.path.basename(output_file[2])}\n")
    log_file.write(f"  - Linhas benignas: {df_val_test['Label'].value_counts().get('Normal', 0)}\n")
    log_file.write(f"  - Linhas maliciosas: {df_val_test['Label'].value_counts().get('Attack', 0)}\n")

print(f"\nDataFrame pré-processado foi salvo com sucesso em: {output_file}")
print("-" * 30)

del df, df_val_test , df_train, X_val_attack

print("\nProcessamento concluído para todos os arquivos.")
print("-" * 30)
print(f"Resumo das contagens salvo em: {log_file_path}")
