import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import roc_curve, roc_auc_score

# --- 1. Configuração de Caminhos e Constantes ---

# Obtém o caminho absoluto do diretório do script
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

# Define os caminhos para os modelos, datasets e plots para o M_CAN
ABSOLUTE_MODELS_PATH = os.path.join(ABSOLUTE_PATH, 'models/M_CAN/complete/')
ABSOLUTE_DATASET_PATH = os.path.join(ABSOLUTE_PATH, 'dataset/preprocessed/M_CAN/')
ABSOLUTE_PLOTS_PATH = os.path.join(ABSOLUTE_PATH, 'plots/M_CAN/complete/roc_auc/')

# Garante que o diretório de plots exista
os.makedirs(ABSOLUTE_PLOTS_PATH, exist_ok=True)

# O conjunto de features para o qual o gráfico será gerado ("terceiro modelo")
FEATURE_SET_TO_PLOT = 'F5'

print("--- Iniciando a geração dos gráficos ROC AUC para o dataset M_CAN ---")

# --- 2. Carregar Modelo e Scaler ---

models_file_path = os.path.join(ABSOLUTE_MODELS_PATH, 'best_iforest_models_M_CAN.pkl')
try:
    with open(models_file_path, 'rb') as f:
        best_models = pickle.load(f)
    print(f"Arquivo de modelos '{models_file_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"  ERRO: Arquivo de modelos não encontrado em '{models_file_path}'. Saindo...")
    exit()

# --- 3. Carregar e Preparar Dados de Teste ---

test_file_path = os.path.join(ABSOLUTE_DATASET_PATH, 'fuzzing_test.pkl')
try:
    df_test = pd.read_pickle(test_file_path)
    print(f"Arquivo de teste '{test_file_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"  ERRO: Arquivo de teste não encontrado em '{test_file_path}'. Saindo...")
    exit()

# Converte os labels para o formato binário (0 para normal, 1 para anomalia)
y_test = df_test['Label'].apply(lambda c: 0 if c == '0' else 1)

# --- 4. Loop para gerar o gráfico para o feature set especificado ---

if FEATURE_SET_TO_PLOT in best_models:
    print(f"\nProcessando o feature set: {FEATURE_SET_TO_PLOT}")

    model_info = best_models[FEATURE_SET_TO_PLOT]
    model = model_info['model']
    scaler = model_info['scaler']
    features_used = model_info['features']

    X_test = df_test[features_used]
    norm_X_test = scaler.transform(X_test)

    y_scores = -model.score_samples(norm_X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = roc_auc_score(y_test, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC para iForest_{FEATURE_SET_TO_PLOT} - Dataset M_CAN')
    plt.legend(loc="lower right")
    
    plot_output_path = os.path.join(ABSOLUTE_PLOTS_PATH, f'roc_auc_iforest_{FEATURE_SET_TO_PLOT}_M_CAN.png')
    plt.savefig(plot_output_path)
    plt.close()
    print(f"  Gráfico ROC AUC salvo em: {plot_output_path}")
else:
    print(f"  AVISO: Modelo para '{FEATURE_SET_TO_PLOT}' não encontrado no arquivo de modelos. Nenhum gráfico foi gerado.")

print("\n--- Processo concluído! ---")
