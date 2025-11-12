# Este script realiza a tunagem de hiperparâmetros para o modelo Isolation Forest
# utilizando validação cruzada K-Fold customizada(sugerida por IA), treina os melhores modelos para
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.datasets import make_classification # Para simular os dados
from sklearn.preprocessing import StandardScaler, MinMaxScaler

N_TRAINING_SAMPLES = 225000  
N_VAL_SAMPLES = 25000
N_FEATURES = 10
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)

def plot_roc_curve(y_true, y_score, max_fpr=1.0):
  fpr, tpr, thresholds = roc_curve(y_true, y_score)
  aucroc = roc_auc_score(y_true, y_score)
  plt.plot(100*fpr[fpr < max_fpr], 100*tpr[fpr < max_fpr], label=f'ROC Curve (AUC = {aucroc:.4f})') 
  plt.xlim(-2,102)
  plt.xlabel('FPR (%)')
  plt.ylabel('TPR (%)')
  plt.legend()
  plt.title('ROC Curve and AUCROC')

def get_tpr_per_attack(y_labels, y_pred):
  aux_df = pd.DataFrame({'Label':y_labels,'prediction':y_pred})
  total_per_label = aux_df['Label'].value_counts().to_dict()
  correct_predictions_per_label = aux_df.query('Label != "BENIGN" and prediction == True').groupby('Label').size().to_dict()
  tpr_per_attack = {}
  for attack_label, total in total_per_label.items():
    if attack_label == 'BENIGN':
      continue
    tp = correct_predictions_per_label[attack_label] if attack_label in correct_predictions_per_label else 0
    tpr = tp/total
    tpr_per_attack[attack_label] = tpr
  return tpr_per_attack

def get_overall_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  acc = (tp+tn)/(tp+tn+fp+fn)
  tpr = tp/(tp+fn)
  fpr = fp/(fp+tn)
  precision = tp/(tp+fp)
  f1 = (2*tpr*precision)/(tpr+precision)
  return {'acc':acc,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1}

def plot_confusion_matrix(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)
  group_counts = [f'{value:.0f}' for value in confusion_matrix(y_true, y_pred).ravel()]
  group_percentages = [f'{value*100:.2f}%' for value in confusion_matrix(y_true, y_pred).ravel()/np.sum(cm)]
  labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
  labels = np.array(labels).reshape(2,2)
  sns.heatmap(cm, annot=labels, cmap='Oranges', xticklabels=['Predicted Benign', 'Predicted Malicious'], yticklabels=['Actual Benign', 'Actual Malicious'], fmt='')
  plt.show()


# Caminho para o arquivo salvo
file_path = 'C:/Users/ricardo.mota/OneDrive - SENAI-PE/Documentos/Doutorado/Disciplina Segurança com IA/can_security/dataset/'
file = 'Fuzzy.pkl'
# Carrega o DataFrame
df = pd.read_pickle(file_path + file)


df_train = df.query('Label == "R"').sample(n=N_TRAINING_SAMPLES, random_state=RANDOM_SEED)
df_val_test = df.drop(df_train.index)

df_train = df_train.reset_index(drop=True)
df_val_test = df_val_test.reset_index(drop=True)

feature_sets = {
    'F1': ['Timestamp', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label', 'Attack'],
    'F2': ['Timestamp', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Label', 'Attack'],
    'F3': ['Timestamp', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Label', 'Attack'],
    'F4': ['Timestamp', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Label', 'Attack'],
    'F5': ['Timestamp', 'Data_1', 'Data_2', 'Data_3', 'Label', 'Attack'],
    'F6': ['Timestamp', 'Data_1', 'Data_2', 'Label', 'Attack'],
    'F7': ['Timestamp', 'Data_1', 'Label', 'Attack'],
    'F8': ['Timestamp', 'Label', 'Attack']
}

# Mantemos o conjunto de ataque para validação fixo
X_val_attack = df_val_test.query('Label != "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)
df_val_test = df_val_test.drop(X_val_attack.index)
df_val_test = df_val_test.reset_index(drop=True)

# O conjunto de teste é o que sobrou
classes_test = df_val_test['Label']
X_test = df_val_test.drop(['Timestamp', 'Label', 'Attack'], axis='columns')

y_test = classes_test.apply(lambda c: 0 if c == 'R' else 1)

# Hiperparâmetros para tunagem
_n_estimators = [100, 200, 300, 350, 355]
_max_samples = [50, 100, 150, 200, 250]
_contamination = [0.0078, 0.008, 0.0082, 0.0085, 0.0086, 0.0088, 0.009, 0.0092]
_bootstrap = [True, False]

# Dicionário para armazenar o melhor modelo final para cada conjunto de features
best_models = {}

# --- Início da Lógica de Validação Cruzada K-Fold Customizada ---
print("\nIniciando a tunagem de hiperparâmetros com K-Fold (k=10) customizado...")
print("-" * 30)

# 1. Configurar o KFold para dividir o conjunto de treino benigno em 10 partes
N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# Loop sobre cada conjunto de features
for f_idx in range(len(feature_sets)):
    feature_set_name = f'F{f_idx+1}'
    features_to_drop = feature_sets[feature_set_name]
    print(f"\n--- Processando Feature Set: {feature_set_name} ---")

    # Dicionário para guardar a pontuação média de cada combinação de hiperparâmetros
    params_scores = {}

    # Loop sobre os hiperparâmetros
    for n in _n_estimators:
        for s in _max_samples:
            for c in _contamination:
                for b in _bootstrap:
                    
                    fold_f1_scores = []
                    # 2. Loop de validação cruzada (k=10)
                    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
                        # Separa os dados do fold
                        df_train_fold = df_train.iloc[train_idx]
                        df_val_benign_fold = df_train.iloc[val_idx]

                        # Monta o conjunto de validação customizado
                        df_val_fold = pd.concat([df_val_benign_fold, X_val_attack]).sample(frac=1, random_state=RANDOM_SEED + fold)
                        y_val_fold = df_val_fold['Label'].apply(lambda l: 0 if l == 'R' else 1)

                        # Seleciona as features para esta iteração
                        X_train_loop = df_train_fold.drop(features_to_drop, axis='columns')
                        X_val_loop = df_val_fold.drop(features_to_drop, axis='columns', errors='ignore')

                        # Fit do scaler APENAS nos dados de treino do fold
                        minmax_scaler = MinMaxScaler().fit(X_train_loop)
                        norm_X_train_fold = minmax_scaler.transform(X_train_loop)
                        norm_X_val_fold = minmax_scaler.transform(X_val_loop)

                        # Treina o modelo
                        model_iforest = IsolationForest(n_estimators=n, max_samples=s, contamination=c, bootstrap=b, random_state=RANDOM_SEED, n_jobs=-1).fit(norm_X_train_fold)

                        # Avalia no conjunto de validação do fold
                        val_preds = model_iforest.predict(norm_X_val_fold)
                        val_preds[val_preds == 1] = 0
                        val_preds[val_preds == -1] = 1
                        
                        fold_f1_scores.append(f1_score(y_val_fold, val_preds))

                    # Calcula a média do F1-Score para a combinação de hiperparâmetros atual
                    mean_f1 = np.mean(fold_f1_scores)
                    params_key = (n, s, c, b)
                    params_scores[params_key] = mean_f1
                    print(f"FS: {feature_set_name} | Params: (n={n}, s={s}, c={c}, b={b}) | Mean F1-Score: {mean_f1:.4f}")

    # Encontra os melhores hiperparâmetros para o conjunto de features atual
    best_params_for_set_key = max(params_scores, key=params_scores.get)
    best_f1_for_set = params_scores[best_params_for_set_key]
    best_n, best_s, best_c, best_b = best_params_for_set_key

    print(f"--> Melhor para {feature_set_name}: F1-Score Médio: {best_f1_for_set:.4f} com params (n={best_n}, s={best_s}, c={best_c}, b={best_b})")

    # 3. Treinar o modelo final para este feature set com os melhores parâmetros encontrados
    # Usamos o conjunto de treino COMPLETO (df_train)
    final_X_train_loop = df_train.drop(features_to_drop, axis='columns')
    final_scaler = MinMaxScaler().fit(final_X_train_loop)
    final_norm_X_train = final_scaler.transform(final_X_train_loop)

    best_model_for_set = IsolationForest(
        n_estimators=best_n, max_samples=best_s, contamination=best_c,
        bootstrap=best_b, random_state=RANDOM_SEED, n_jobs=-1
    ).fit(final_norm_X_train)

    # Armazena o modelo final e suas informações
    best_models[feature_set_name] = {
        'model': best_model_for_set,
        'f1_score_val': best_f1_for_set, # F1-score médio da validação cruzada
        'features': final_X_train_loop.columns.to_list(),
        'scaler': final_scaler,
        'params': {'n_estimators': best_n, 'max_samples': best_s, 'contamination': best_c, 'bootstrap': best_b}
    }

# Find the overall best model from the dictionary
overall_best_feature_set = max(best_models, key=lambda k: best_models[k]['f1_score_val'])
overall_best_model_info = best_models[overall_best_feature_set]
overall_best_model = overall_best_model_info['model']
overall_best_f1 = overall_best_model_info['f1_score_val']

print("-" * 30)
print(f"\nMelhor modelo geral: {overall_best_feature_set} com params: {overall_best_model_info['params']} | Melhor F1-Score Médio na Validação: {overall_best_f1:.4f}")
print("-" * 30)

# --- Salvando os melhores modelos em um arquivo ---
output_models_file = 'best_iforest_models.pkl'
with open(output_models_file, 'wb') as f:
    pickle.dump(best_models, f)

print(f"\nDicionário com os melhores modelos foi salvo em: {output_models_file}")
print("-" * 30)

# --- Avaliação e Plot das Matrizes de Confusão no CONJUNTO DE TESTE ---

n_models = len(best_models)
n_cols = 2
n_rows = math.ceil(n_models / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 6))
axes = axes.flatten()

for i, (feature_set_name, model_info) in enumerate(best_models.items()):
    ax = axes[i]
    model = model_info['model']
    scaler = model_info['scaler']
    features_to_drop = feature_sets[feature_set_name]

    # Prepara o conjunto de teste para este modelo
    X_test_loop = X_test.drop(features_to_drop, axis='columns', errors='ignore')
    norm_X_test_loop = scaler.transform(X_test_loop)

    # Fazer predições no conjunto de TESTE
    test_preds = model.predict(norm_X_test_loop)
    test_preds[test_preds == 1] = 0
    test_preds[test_preds == -1] = 1

    # Calcula métricas no teste
    test_metrics = get_overall_metrics(y_test, test_preds)
    model_info['test_metrics'] = test_metrics # Salva as métricas de teste

    # Gerar e plotar a matriz de confusão
    cm = confusion_matrix(y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predito Benigno', 'Predito Malicioso'],
                yticklabels=['Real Benigno', 'Real Malicioso'])
    ax.set_title(f'Matriz de Confusão (Teste) para {feature_set_name}\nF1-Score: {test_metrics["f1-score"]:.4f}')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')

# Esconde eixos extras se o número de modelos for ímpar
for i in range(n_models, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('confusion_matrices_test.png')
print("\nMatrizes de confusão (no conjunto de teste) salvas em 'confusion_matrices_test.png'")
plt.show()
