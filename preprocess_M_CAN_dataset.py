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
preprocessed_path = os.path.join(file_path, 'dataset/preprocessed/M_CAN/')
dataset_path = 'dataset/M_CAN/'
file_list = ['g80_mcan_normal_data.csv','g80_mcan_fuzzing_data.csv']

# --- Preparando o arquivo de log para o resumo ---
log_file_path = os.path.join(preprocessed_path, 'M_CAN_dataset_summary.txt')

# Garante que o diretório de log exista antes de tentar escrever o arquivo
os.makedirs(preprocessed_path, exist_ok=True)

with open(log_file_path, 'w') as log_file:
    log_file.write("Resumo do Pré-processamento dos Datasets\n")
    log_file.write("=" * 100 + "\n")

for file in file_list:
    # Lendo o arquivo CSV
    df = pd.read_csv(file_path + dataset_path + file, header=None, sep=',', on_bad_lines='skip', low_memory=False)

    # Definindo os nomes das colunas
    df.columns = ['Timestamp', 'ID', 'DLC', 'Payload', 'Label']

    # Separa a coluna 'Payload' em colunas 'Data_0' a 'Data_7'
    # expand=True cria novas colunas a partir dos dados divididos
    data_cols = [f'Data_{i}' for i in range(8)]
    df[data_cols] = df['Payload'].str.split(' ', expand=True)

    # Remove a coluna 'Payload' original e a coluna 'Timestamp' que não será usada
    df = df.drop(columns=['Payload', 'Timestamp', 'ID', 'DLC'])
    df = df.drop(df.index[0])  # Remove a primeira linha que pode conter cabeçalhos indesejados

    # Reordena as colunas para uma melhor visualização
    df = df[data_cols + ['Label']]
    
    # Descartando duplicadas
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

    # Descartando registros com valores NaN/Null/NA em outras colunas (exceto as de dados já tratadas)
    initial_len = df.shape[0]
    df = df.dropna()
    print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA')

    df = df.reset_index(drop=True)
    
    # Converte a coluna 'Label' para numérico, depois para inteiro (removendo '.0'), e finalmente para string.
    # pd.to_numeric lida com valores como '0.0'. .astype(int) converte para inteiro. .astype(str) converte para '0' ou '1'.
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int).astype(str)

    # Convertendo valores hexadecimais para inteiros
    def hex_to_int(hex_str):
        try:
            return int(str(hex_str), 16)
        except (ValueError, TypeError):
            return np.nan # Retorna NaN se a conversão falhar

    for col in data_cols:
        df[col] = df[col].apply(hex_to_int)
    
    # Após a conversão, podem surgir NaNs se algum valor não for um hexadecimal válido.
    # Vamos verificar e descartar essas linhas.
    initial_len = df.shape[0]
    df = df.dropna()
    print(f'Tamanho inicial: {initial_len}, tamanho final {df.shape[0]} | Descartados {initial_len - df.shape[0]} registros com valores NA após conversão')
    df = df.reset_index(drop=True)

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\nPré-processamento do Dataset {file.split('_')[2]}\n")
        log_file.write("_" * 100 + "\n")
        # Captura a saída de df.info() para uma string
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        log_file.write(info_str)
        log_file.write("-" * 100 + "\n") 
        
    print(df.head())
    print(df.info())
    
    # --- Salvando o DataFrame para uso posterior ---
    if file == 'g80_mcan_normal_data.csv':
        df_train = df.query('Label == "0"').sample(n=N_TRAINING_SAMPLES, random_state=RANDOM_SEED)
        df_train = df_train.reset_index(drop=True)
        output_file = os.path.join(preprocessed_path, file.split('_')[2] + '_train.pkl')
        # Salva o DataFrame em formato Pickle
        df_train.to_pickle(output_file)
        
        # --- Registrando as contagens no arquivo de log ---
        with open(log_file_path, 'a') as log_file:
            # Contagens inicial do arquivo pré-processado
            log_file.write(f"Contagens iniciais do arquivo pré-processado {file}:\n")
            log_file.write(f"  - Total de linhas: {len(df)}\n")
            log_file.write(f"  - Linhas benignas ('0'): {df['Label'].value_counts().get('0', 0)}\n")
            log_file.write(f"  - Linhas maliciosas ('1'): {df['Label'].value_counts().get('1', 0)}\n\n")
            log_file.write(df.head().to_string() + "\n")
            log_file.write("-" * 100 + "\n") 
        
            # Contagens para o arquivo de treino
            log_file.write(f"Arquivo gerado: {os.path.basename(output_file)}\n")
            log_file.write(f"  - Linhas benignas ('0'): {len(df_train)}\n")
            log_file.write(f"  - Linhas maliciosas ('1'): 0\n\n")
            log_file.write(df_train.head().to_string() + "\n") 
            log_file.write("=" * 100 + "\n") 
            
        del df_train
    else:
        X_val_attack = df.query('Label != "0"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)
        df_test = df.drop(X_val_attack.index)
        X_val_attack = X_val_attack.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        # Define o caminho e o nome do arquivo de saída
        output_file = [''] * 2  # Lista para armazenar os nomes dos arquivos de saída
        output_file[0] = os.path.join(preprocessed_path, file.split('_')[2] + '_val_attack.pkl')
        output_file[1] = os.path.join(preprocessed_path, file.split('_')[2] + '_test.pkl')
        # Salva o DataFrame em formato Pickle
        X_val_attack.to_pickle(output_file[0])
        df_test.to_pickle(output_file[1])

        # --- Registrando as contagens no arquivo de log ---
        with open(log_file_path, 'a') as log_file:
            # Contagens inicial do arquivo pré-processado
            log_file.write(f"Contagens iniciais do arquivo pré-processado {file}:\n")
            log_file.write(f"  - Total de linhas: {len(df)}\n")
            log_file.write(f"  - Linhas benignas ('0'): {df['Label'].value_counts().get('0', 0)}\n")
            log_file.write(f"  - Linhas maliciosas ('1'): {df['Label'].value_counts().get('1', 0)}\n\n")
            log_file.write(df.head().to_string() + "\n")
            log_file.write("-" * 100 + "\n") 
            
            # Contagens para o arquivo de validação de ataques
            log_file.write(f"Arquivo gerado: {os.path.basename(output_file[0])}\n")
            log_file.write(f"  - Linhas benignas ('0'): 0\n")
            log_file.write(f"  - Linhas maliciosas ('1'): {len(X_val_attack)}\n\n")
            log_file.write(X_val_attack.head().to_string() + "\n") 
            log_file.write("-" * 100 + "\n") 

            # Contagens para o arquivo de teste
            log_file.write(f"Arquivo gerado: {os.path.basename(output_file[1])}\n")
            log_file.write(f"  - Linhas benignas ('0'): {df_test['Label'].value_counts().get('0', 0)}\n")
            log_file.write(f"  - Linhas maliciosas ('1'): {df_test['Label'].value_counts().get('1', 0)}\n\n")
            log_file.write(df_test.head().to_string() + "\n") 
            log_file.write("=" * 100 + "\n\n") 
            
        del df_test, X_val_attack

    print(f"\nDataFrame pré-processado foi salvo com sucesso em: {output_file}")
    print("-" * 30)
    
    del df

print("\nProcessamento concluído para todos os arquivos.")
print("-" * 30)
print(f"Resumo das contagens salvo em: {log_file_path}")
