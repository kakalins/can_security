import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import roc_curve, roc_auc_score

# --- 1. Configuração de Caminhos e Constantes ---

# Obtém o caminho absoluto do diretório do script
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# Define os caminhos para os modelos, datasets e plots
ABSOLUTE_MODELS_PATH = os.path.join(ABSOLUTE_PATH, 'models/car_hacking/complete/')
ABSOLUTE_DATASET_PATH = os.path.join(ABSOLUTE_PATH, 'dataset/preprocessed/car_hacking/')
ABSOLUTE_PLOTS_PATH = os.path.join(ABSOLUTE_PATH, 'plots/car_hacking/complete/roc_auc/')

# Garante que o diretório de plots exista
os.makedirs(ABSOLUTE_PLOTS_PATH, exist_ok=True)

# Lista dos datasets que serão processados
DATASET_NAMES = ['Fuzzy', 'gear', 'RPM']

# O conjunto de features para o qual o gráfico será gerado ("quinto modelo")
FEATURE_SET_TO_PLOT = 'F5'

print("--- Iniciando a geração dos gráficos ROC AUC ---")

# --- 2. Loop para processar cada dataset ---

for dataset_name in DATASET_NAMES:
    print(f"\nProcessando o dataset: {dataset_name}")

    # --- Carregar Modelo e Scaler ---
    models_file_path = os.path.join(ABSOLUTE_MODELS_PATH, f'best_iforest_models_car_hacking_{dataset_name}.pkl')
    try:
        with open(models_file_path, 'rb') as f:
            best_models = pickle.load(f)
    except FileNotFoundError:
        print(f"  AVISO: Arquivo de modelos não encontrado para '{dataset_name}'. Pulando...")
        continue
    
    # Extrai o modelo e o scaler para o feature set F5
    if FEATURE_SET_TO_PLOT not in best_models:
        print(f"  AVISO: Modelo para '{FEATURE_SET_TO_PLOT}' não encontrado no arquivo de '{dataset_name}'. Pulando...")
        continue

    model_info = best_models[FEATURE_SET_TO_PLOT]
    model = model_info['model']
    scaler = model_info['scaler']
    features_used = model_info['features']

    # --- Carregar e Preparar Dados de Teste ---
    test_file_path = os.path.join(ABSOLUTE_DATASET_PATH, f'{dataset_name}_test.pkl')
    df_test = pd.read_pickle(test_file_path)

    X_test = df_test[features_used]
    y_test = df_test['Label'].apply(lambda c: 0 if c == 'R' else 1)

    # Normaliza os dados de teste com o scaler salvo
    norm_X_test = scaler.transform(X_test)

    # --- Gerar Scores e Calcular ROC ---
    # O método score_samples retorna o oposto do score de anomalia (maior é mais normal)
    # Invertemos o sinal para que valores maiores indiquem maior anomalia.
    y_scores = -model.score_samples(norm_X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)

    # --- Plotar o Gráfico ---
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC para iForest_{FEATURE_SET_TO_PLOT} - Dataset {dataset_name}')
    plt.legend(loc="lower right")
    
    # --- Salvar o Gráfico ---
    plot_output_path = os.path.join(ABSOLUTE_PLOTS_PATH, f'roc_auc_iforest_{FEATURE_SET_TO_PLOT}_{dataset_name}.png')
    plt.savefig(plot_output_path)
    plt.close() # Fecha a figura para liberar memória
    print(f"  Gráfico ROC AUC salvo em: {plot_output_path}")

print("\n--- Processo concluído! ---")
