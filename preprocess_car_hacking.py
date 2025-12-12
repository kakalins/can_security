import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import io
import seaborn as sns


# Configurações globais
N_TRAINING_SAMPLES = 250000  
N_VAL_SAMPLES = 25000
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)

# Definindo o caminho dos arquivos
file_path = os.path.dirname(os.path.abspath(__file__)) + '/'
preprocessed_path = os.path.join(file_path, 'dataset/preprocessed/car_hacking/')
dataset_path = 'dataset/car_hacking/'
file_list = ['Fuzzy_dataset.csv','gear_dataset.csv','RPM_dataset.csv']
#df_list = []

# --- Preparando o arquivo de log para o resumo ---
log_file_path = os.path.join(preprocessed_path, 'car_hacking_dataset_summary.txt')

# Garante que o diretório de log exista antes de tentar escrever o arquivo
os.makedirs(preprocessed_path, exist_ok=True)

with open(log_file_path, 'w') as log_file:
    log_file.write("Resumo do Pré-processamento dos Datasets\n")
    log_file.write("=" * 100 + "\n")

for file in file_list:
    # Lendo o arquivo CSV
    df = pd.read_csv(file_path + dataset_path + file, header=None, sep=',', on_bad_lines='skip', low_memory=False)

    # Definindo o nome do ataque com base no nome do arquivo
    # attack_name = file.split('_')[0]+"_attack"

    # Criando a coluna 'Attack' com base na coluna 11
    # df['Attack'] = df[11].apply(lambda x: 'Benign' if x == 'R' else attack_name) 

    # Definindo os nomes das colunas iniciais. O número de colunas pode ser maior que 12.
    num_cols = df.shape[1]
    column_names = ['Timestamp', 'ID', 'DLC'] + [f'Data_{i}' for i in range(num_cols - 4)] + ['Label']
    df.columns = column_names[:num_cols]

    # Encontra 'R' ou 'T' em qualquer coluna de dados e move para uma nova coluna 'Label'
    data_cols_and_label = [col for col in df.columns if 'Data_' in str(col) or 'Label' in str(col)]
    df['Label_consolidated'] = pd.Series(dtype='object')
    for col in data_cols_and_label:
        # Encontra 'R' ou 'T' e preenche a nova coluna de label
        mask = df[col].isin(['R', 'T'])
        df.loc[mask, 'Label_consolidated'] = df.loc[mask, col]
        # Limpa a coluna original onde o label foi encontrado
        df.loc[mask, col] = np.nan

    # df.columns = df.columns.str.strip()
    
    # Remove a coluna 'Label' antiga e renomeia a nova
    df = df.drop(columns=['Label']).rename(columns={'Label_consolidated': 'Label'})

    # df[df.duplicated()]
    # # Descartando duplicadas
    # initial_len = df.shape[0]
    # df = df.drop_duplicates()
    # print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartadas {initial_len - df.shape[0]} duplicadas')

    # Exibindo informações sobre valores ausentes
    print(f"\n--- Verificando valores ausentes no arquivo: {file}")
    print("Colunas com valores ausentes:")
    print(df.columns[df.isna().any(axis=0)])
    print("\nContagem de valores ausentes por coluna:")
    print(df.isna().sum()[df.isna().sum() > 0])
    
    # Define as colunas de dados que podem ter valores ausentes
    data_cols = ['Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7']
    
    # Substituindo valores None (NaN) por '00' apenas nas colunas de dados
    df[data_cols] = df[data_cols].fillna('00')

    # # Descartando registros com valores NaN/Null/NA
    # initial_len = df.shape[0]
    # df = df.dropna()
    # print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')

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

    df_columns_isfinite = np.isfinite(df.drop(['Label'], axis='columns', errors='ignore')).all(axis=0)
    df_columns_isfinite[df_columns_isfinite == False]

    df_rows_isfinite = np.isfinite(df.drop(['Label'], axis='columns', errors='ignore')).all(axis=1)
    inf_indexes = df_rows_isfinite[df_rows_isfinite == False].index

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\nPré-processamento do Dataset {file.split('_')[0]}\n")
        log_file.write("_" * 100 + "\n")
        # log_file.write(df.head().to_string() + "\n") 
        # log_file.write("-" * 100 + "\n") 
        # Captura a saída de df.info() para uma string
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        log_file.write(info_str)
        log_file.write("-" * 100 + "\n") 
    print(df.head())
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
    output_file[0] = os.path.join(preprocessed_path, file.split('_')[0] + '_train.pkl')
    output_file[1] = os.path.join(preprocessed_path, file.split('_')[0] + '_val_attack.pkl')
    output_file[2] = os.path.join(preprocessed_path, file.split('_')[0] + '_test.pkl')

    # Salva o DataFrame em formato Pickle
    df_train.to_pickle(output_file[0])
    X_val_attack.to_pickle(output_file[1])
    df_val_test.to_pickle(output_file[2])

    # --- Registrando as contagens no arquivo de log ---
    with open(log_file_path, 'a') as log_file:
        # Contagens inicial do arquivo pré-processado
        log_file.write(f"Contagens iniciais do arquivo pré-processado {file}:\n")
        log_file.write(f"  - Total de linhas: {len(df)}\n")
        log_file.write(f"  - Linhas benignas ('R'): {df['Label'].value_counts().get('R', 0)}\n")
        log_file.write(f"  - Linhas maliciosas ('T'): {df['Label'].value_counts().get('T', 0)}\n\n")
        log_file.write(df.head().to_string() + "\n")
        log_file.write("-" * 100 + "\n") 
     
        # Contagens para o arquivo de treino
        log_file.write(f"Arquivo gerado: {os.path.basename(output_file[0])}\n")
        log_file.write(f"  - Linhas benignas ('R'): {len(df_train)}\n")
        log_file.write(f"  - Linhas maliciosas ('T'): 0\n\n")
        log_file.write(df_train.head().to_string() + "\n") 
        log_file.write("-" * 100 + "\n") 

        # Contagens para o arquivo de validação de ataques
        log_file.write(f"Arquivo gerado: {os.path.basename(output_file[1])}\n")
        log_file.write(f"  - Linhas benignas ('R'): 0\n")
        log_file.write(f"  - Linhas maliciosas ('T'): {len(X_val_attack)}\n\n")
        log_file.write(X_val_attack.head().to_string() + "\n") 
        log_file.write("-" * 100 + "\n") 

        # Contagens para o arquivo de teste
        log_file.write(f"Arquivo gerado: {os.path.basename(output_file[2])}\n")
        log_file.write(f"  - Linhas benignas ('R'): {df_val_test['Label'].value_counts().get('R', 0)}\n")
        log_file.write(f"  - Linhas maliciosas ('T'): {df_val_test['Label'].value_counts().get('T', 0)}\n\n")
        log_file.write(df_val_test.head().to_string() + "\n") 
        log_file.write("=" * 100 + "\n\n") 

    print(f"\nDataFrame pré-processado foi salvo com sucesso em: {output_file}")
    print("-" * 30)
    
    del df, df_val_test , df_train, X_val_attack

print("\nProcessamento concluído para todos os arquivos.")
print("-" * 30)
print(f"Resumo das contagens salvo em: {log_file_path}")
