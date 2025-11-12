import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

from itertools import product
from sklearn.utils import resample
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 

RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)


# Caminho para o arquivo salvo
file_path = 'C:/Users/ricardo.mota/OneDrive - SENAI-PE/Documentos/Doutorado/Disciplina Segurança com IA/can_security/dataset/'
file_list = ['Fuzzy.pkl','RPM.pkl','gear.pkl']
# Carrega o DataFrame
df_fuzzy = pd.read_pickle(file_path + file_list[0])
df_RPM = pd.read_pickle(file_path + file_list[1])
df_gear = pd.read_pickle(file_path + file_list[2])

#print("DataFrame carregado com sucesso!")

# seleciona 250000 amostras benignas do dataset fuzzy para treino
N_TRAINING_SAMPLES = 225000  
N_VAL_SAMPLES = 25000

initial_len = df_fuzzy.shape[0]
df_fuzzy = df_fuzzy.dropna()
print(f'Tamanho inicial: {initial_len}, tamanho final {df_fuzzy.shape[0]} | Descartados {initial_len - df_fuzzy.shape[0]} registros com valores NA')

df_fuzzy = df_fuzzy.reset_index(drop=True)



df_train_fuzzy = df_fuzzy.query('Label == "R"').sample(n=N_TRAINING_SAMPLES, random_state=RANDOM_SEED)
df_val_test = df_fuzzy.drop(df_train_fuzzy.index)

df_train_fuzzy = df_train_fuzzy.reset_index(drop=True)
df_val_test = df_val_test.reset_index(drop=True)

X_train = df_train_fuzzy.drop(['Timestamp', 'Label', 'Attack'], axis='columns')

X_val_benign = df_val_test.query('Label == "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)
X_val_attack = df_val_test.query('Label != "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)

# Concatena os dataframes de validação e embaralha as amostras
X_val = pd.concat([X_val_benign, X_val_attack]).sample(frac=1, random_state=RANDOM_SEED)
df_val_test = df_val_test.drop(X_val.index).reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
classes_val = X_val['Label']
X_val = X_val.drop(['Timestamp', 'Label','Attack'], axis='columns')
classes_test = df_val_test['Label']
X_test = df_val_test.drop(['Timestamp', 'Label', 'Attack'], axis='columns')

# X_val, X_test = X_val.reset_index(drop=True), X_test.reset_index(drop=True)
# classes_val, classes_test =  classes_val.reset_index(drop=True), classes_test.reset_index(drop=True)

y_val, y_test = classes_val.apply(lambda c: 0 if c == 'R' else 1), classes_test.apply(lambda c: 0 if c == 'R' else 1)

df_val = X_val.copy()
df_val['Attack'] = classes_val
df_val['isAttack'] = df_val['Attack'].apply(lambda l: 0 if l == 'R' else 1)

# Usando MinMax Scaler dessa vez para que a rede neural seja capaz de gerar saídas no intervalo numérico da função sigmóide

minmax_scaler = MinMaxScaler()
minmax_scaler = minmax_scaler.fit(X_train)

norm_X_train = minmax_scaler.transform(X_train)
norm_X_val = minmax_scaler.transform(X_val)
norm_X_test = minmax_scaler.transform(X_test)

# --- Início da Seção Modificada com K-Fold ---

# 1. Combinar os dados de treino e validação para a validação cruzada
X_combined = np.concatenate((norm_X_train, norm_X_val), axis=0)
# Criar os rótulos correspondentes (0 para benigno, 1 para ataque)
# norm_X_train é todo benigno (rótulo 0)
y_combined = np.concatenate((np.zeros(norm_X_train.shape[0]), y_val), axis=0)

print("\nTamanho do conjunto de treino: ", norm_X_train.shape[0])
print("\nTamanho do conjunto de validação: ", X_val.shape[0])
print("\nTamanho do conjunto combinado: ", X_combined.shape[0])
print("\nTamanho do conjunto de teste: ", norm_X_test.shape[0])

print("\n--- Iniciando busca de hiperparâmetros com validação cruzada Stratified K-Fold (k=10) ---")

# Configurando Isolation Forest
param_grid = {
    'n_estimators': [345, 350, 352],
    'contamination': ['auto', 0.0073, 0.0075, 0.0078, 0.008, 0.0082]
}

# 2. Configurar o StratifiedKFold para k=10
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

best_score = -1
best_params = {}

# Cria uma lista com todas as combinações de parâmetros para usar com o tqdm
params_list = list(product(param_grid['n_estimators'], param_grid['contamination']))

# Loop para busca de hiperparâmetros
for n_estimators, contamination in tqdm(params_list, desc="Buscando Hiperparâmetros"):
        fold_auc_scores = []
        fold_accuracy_scores = []
        fold_precision_scores = []
        fold_recall_scores = []
        fold_f1_scores = []
        # 3. Loop de validação cruzada (k=10)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
            X_train_fold, X_val_fold = X_combined[train_idx], X_combined[val_idx]
            y_train_fold, y_val_fold = y_combined[train_idx], y_combined[val_idx]

            # 4. Filtrar o conjunto de treino para conter APENAS dados benignos
            X_train_fold_benign = X_train_fold[y_train_fold == 0]

            # Treina o modelo com os dados de treino benignos do fold atual
            model_iforest = IsolationForest(n_estimators=n_estimators,
                                            contamination=contamination,
                                            random_state=RANDOM_SEED,
                                            n_jobs=-1).fit(X_train_fold_benign)

            # --- Avaliação com AUC-ROC (baseada em score) ---
            y_scores = -model_iforest.decision_function(X_val_fold)
            fold_auc_scores.append(roc_auc_score(y_val_fold, y_scores))

            # --- Avaliação com métricas baseadas em predição binária ---
            # O método .predict() retorna 1 para inlier (normal) e -1 para outlier (anomalia)
            y_pred_raw = model_iforest.predict(X_val_fold)
            # Mapear para 0 (normal) e 1 (anomalia) para corresponder aos nossos rótulos y_val_fold
            y_pred = np.array([0 if p == 1 else 1 for p in y_pred_raw])

            # Adicionar scores do fold atual às listas
            fold_accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
            # 'zero_division=0' evita warnings caso não haja predições positivas
            fold_precision_scores.append(precision_score(y_val_fold, y_pred, zero_division=0))
            fold_recall_scores.append(recall_score(y_val_fold, y_pred, zero_division=0))
            fold_f1_scores.append(f1_score(y_val_fold, y_pred, zero_division=0))

        # Calcula a pontuação média entre os 10 folds
        mean_auc = np.mean(fold_auc_scores)
        mean_accuracy = np.mean(fold_accuracy_scores)
        mean_precision = np.mean(fold_precision_scores)
        mean_recall = np.mean(fold_recall_scores)
        mean_f1 = np.mean(fold_f1_scores)

        # Calcula o desvio padrão para as métricas solicitadas
        std_accuracy = np.std(fold_accuracy_scores)
        std_precision = np.std(fold_precision_scores)
        std_recall = np.std(fold_recall_scores)

        print(f"\nParams: n_est={n_estimators}, cont={contamination} | AUC: {mean_auc:.4f}, "
              f"Acc: {mean_accuracy:.4f} (±{std_accuracy:.4f}), "
              f"Prec: {mean_precision:.4f} (±{std_precision:.4f}), "
              f"Rec: {mean_recall:.4f} (±{std_recall:.4f}), "
              f"F1: {mean_f1:.4f}")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = {'n_estimators': n_estimators, 'contamination': contamination}
            print(f"--> Novo melhor F1-score média: {best_score:.4f} com params: {best_params}")

print("\nBusca finalizada!")
print(f"Melhores hiperparâmetros encontrados: {best_params}")
print(f"Melhor F1-score na validação: {best_score:.4f}")

# Treina o modelo final com os melhores parâmetros encontrados
# Usamos o conjunto de treino original (grande e benigno) para o modelo final
best_iforest_model = IsolationForest(**best_params, random_state=RANDOM_SEED, n_jobs=-1).fit(norm_X_train)

# --- Avaliação do Modelo Final no Conjunto de Teste ---

print("\n--- Avaliando o melhor modelo no conjunto de teste ---")

# --- Avaliação com Bootstrapping para estimar o desvio padrão ---
n_iterations = 100
test_accuracy_scores = []
test_precision_scores = []
test_recall_scores = []
test_f1_scores = []

for i in tqdm(range(n_iterations), desc="Avaliando com Bootstrap"):
    # 1. Criar uma amostra de bootstrap do conjunto de teste
    X_test_sample, y_test_sample = resample(norm_X_test, y_test, random_state=RANDOM_SEED + i)

    # 2. Fazer predições na amostra
    y_pred_raw = best_iforest_model.predict(X_test_sample)
    y_pred = np.array([0 if p == 1 else 1 for p in y_pred_raw])

    # 3. Calcular e armazenar as métricas
    test_accuracy_scores.append(accuracy_score(y_test_sample, y_pred))
    test_precision_scores.append(precision_score(y_test_sample, y_pred, zero_division=0))
    test_recall_scores.append(recall_score(y_test_sample, y_pred, zero_division=0))
    test_f1_scores.append(f1_score(y_test_sample, y_pred, zero_division=0))

# 4. Calcular a média e o desvio padrão das métricas de teste
mean_test_accuracy = np.mean(test_accuracy_scores)
std_test_accuracy = np.std(test_accuracy_scores)
mean_test_precision = np.mean(test_precision_scores)
std_test_precision = np.std(test_precision_scores)
mean_test_recall = np.mean(test_recall_scores)
std_test_recall = np.std(test_recall_scores)
mean_test_f1 = np.mean(test_f1_scores)
std_test_f1 = np.std(test_f1_scores)

print(f"Acurácia no Teste: {mean_test_accuracy:.4f} (±{std_test_accuracy:.4f})")
print(f"Precisão no Teste: {mean_test_precision:.4f} (±{std_test_precision:.4f})")
print(f"Recall no Teste:   {mean_test_recall:.4f} (±{std_test_recall:.4f})")
print(f"F1-Score no Teste:  {mean_test_f1:.4f} (±{std_test_f1:.4f})")

# 4. Gerar e exibir a matriz de confusão
# A matriz de confusão é calculada no conjunto de teste completo para uma visão geral
y_pred_test_full = np.array([0 if p == 1 else 1 for p in best_iforest_model.predict(norm_X_test)])
cm = confusion_matrix(y_test, y_pred_test_full)

# Normaliza a matriz de confusão pelas linhas (pelos valores verdadeiros) para obter percentuais
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Cria os rótulos para as anotações combinando o percentual e a contagem absoluta
annot_labels = (np.asarray(["{0:.2%}\n({1})".format(p, v) for p, v in zip(cm_percent.flatten(), cm.flatten())])).reshape(cm.shape)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues', xticklabels=['Normal', 'Malicious'], yticklabels=['Normal', 'Malicious'])
plt.title('iForest_8F - Matriz de Confusão no Conjunto de Teste')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()