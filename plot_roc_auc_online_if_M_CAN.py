import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

# --- 1. Configuração de Caminhos e Constantes ---

# Obtém o caminho absoluto do diretório do script
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))

# Define os caminhos para os modelos, datasets e plots para o M_CAN
ABSOLUTE_MODELS_PATH = os.path.join(ABSOLUTE_PATH, 'models/M_CAN/online_if/complete/')
ABSOLUTE_DATASET_PATH = os.path.join(ABSOLUTE_PATH, 'dataset/preprocessed/M_CAN/')
ABSOLUTE_PLOTS_PATH = os.path.join(ABSOLUTE_PATH, 'plots/M_CAN/online_if/complete/roc_auc/')

# # O conjunto de features para o qual o gráfico será gerado ("quinto modelo")
# FEATURE_SET_TO_PLOT = 'F5'

# Garante que o diretório de plots exista
os.makedirs(ABSOLUTE_PLOTS_PATH, exist_ok=True)

print("--- Iniciando a geração do gráfico ROC AUC para o modelo Online iForest (M_CAN) ---")

# --- 2. Carregar Modelo e Scaler ---

model_file_path = os.path.join(ABSOLUTE_MODELS_PATH, 'best_online_if_M_CAN.pkl')
try:
    with open(model_file_path, 'rb') as f:
        model_info = pickle.load(f)
    print(f"Arquivo de modelo '{model_file_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"  ERRO: Arquivo de modelo não encontrado em '{model_file_path}'. Saindo...")
    exit()

print(f"  Modelo carregado: {model_info}")

model = model_info['model']
scaler = model_info['scaler']
features_used = model_info['features']
# O nome do feature set não é salvo diretamente, mas podemos extraí-lo do nome da primeira feature
feature_set_name = f"F{int(features_used[0].split('_')[-1]) + 3}"

# --- 3. Carregar e Preparar Dados de Teste ---
test_file_path = os.path.join(ABSOLUTE_DATASET_PATH, 'fuzzing_test.pkl')
df_test = pd.read_pickle(test_file_path)

# Garante que a coluna 'Time_Diff' exista, como no treinamento
df_test['Time_Diff'] = df_test['Timestamp'].diff().fillna(0)

X_test = df_test[features_used]
y_test = df_test['Label'].apply(lambda c: 0 if c == '0' else 1)

# --- 4. Gerar Scores e Calcular ROC ---
y_scores = []
print("  Calculando scores no conjunto de teste...")
for _, x in tqdm(X_test.iterrows(), total=X_test.shape[0], leave=False):
    x_dict = x.to_dict()
    x_scaled = scaler.transform_one(x_dict)
    score = model.score_one(x_scaled)
    y_scores.append(score)

fpr, tpr, _ = roc_curve(y_test, y_scores)
auc_score = roc_auc_score(y_test, y_scores)

# --- 5. Plotar e Salvar o Gráfico ---
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title(f'Curva ROC para OnlineIF_{feature_set_name} - Dataset M_CAN')
plt.legend(loc="lower right")

plot_output_path = os.path.join(ABSOLUTE_PLOTS_PATH, f'roc_auc_online_if_{feature_set_name}_M_CAN.png')
plt.savefig(plot_output_path)
plt.close()
print(f"  Gráfico ROC AUC salvo em: {plot_output_path}")

print("\n--- Processo concluído! ---")
