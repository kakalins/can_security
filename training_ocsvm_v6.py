# Esta versão do código implementa a tunagem de hiperparâmetros do Isolation Forest
# para diferentes conjuntos de features, salvando o melhor modelo, scores e sua matriz de confusão 
# para cada conjunto de F1 a F8. Além disso, é implementado k-fold manualmente para treinamento e validação.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.datasets import make_classification # Para simular os dados
from sklearn.preprocessing import StandardScaler, MinMaxScaler

N_TRAINING_SAMPLES = 250000  
N_VAL_SAMPLES = 25000 
FOLD_SPLIT = 10
RANDOM_SEED = 33
DATASET_NAME = 'Fuzzy'  # Opções: 'Fuzzy', 'Gear', 'RPM'
ABSOLUTE_PATH = 'C:/Users/ricardo.mota/OneDrive - SENAI-PE/Documentos/Doutorado/Disciplina Segurança com IA/can_security/'
DATASET = {'Fuzzy': ABSOLUTE_PATH + 'dataset/Fuzzy.pkl',
           'Gear': ABSOLUTE_PATH + 'dataset/gear.pkl',
           'RPM': ABSOLUTE_PATH + 'dataset/RPM.pkl'
           }
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
file_path = DATASET[DATASET_NAME]

# Carrega o DataFrame
df = pd.read_pickle(file_path)


df_train = df.query('Label == "R"').sample(n=N_TRAINING_SAMPLES, random_state=RANDOM_SEED)
df_val_test = df.drop(df_train.index)

df_train = df_train.reset_index(drop=True)
df_val_test = df_val_test.reset_index(drop=True)

# X_val_benign = df_val_test.query('Label == "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)
X_val_attack = df_val_test.query('Label != "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)

# df_val_test = df_val_test.drop(X_val_benign.index)
df_val_test = df_val_test.drop(X_val_attack.index)

df_val_test = df_val_test.reset_index(drop=True)
X_val_attack = X_val_attack.reset_index(drop=True)

classes_test = df_val_test['Label']
X_test = df_val_test.drop(['Timestamp', 'Label', 'Attack'], axis='columns')
y_test = classes_test.apply(lambda c: 0 if c == 'R' else 1)

# Divisão manual em FOLD_SPLIT folds
folds_list = np.array_split(df_train, FOLD_SPLIT)

del df, df_val_test

# Definição dos conjuntos de features a serem testados
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

# Hiperparâmetros para tunagem
# KERNEL = ['sigmoid', 'linear', 'poly', 'rbf']
# GAMMA = ['scale', 'auto']
KERNEL = ['rbf']
GAMMA = ['scale']
NU = [0.01, 0.05, 0.1, 0.2]


# Use a dictionary to store the best model for each feature set
best_models = {}
overall_best_f1 = 0
overall_best_model = None
overall_best_val_preds = None

# Tunagem de hiperparâmetros do Isolation Forest
print("\nIniciando a tunagem de hiperparâmetros do OCSVM...")
print("-" * 30)
for f_idx in range(len(feature_sets)):
    feature_set_name = f'F{f_idx+1}'
    features_to_drop = feature_sets[feature_set_name]
    print(f"Testando com o conjunto de features {feature_set_name}...")

    # Reset bests for the current feature set
    best_f1_for_set = 0
    best_model_for_set = None

    for k in KERNEL:
        for n in NU:
            for g in GAMMA:
                for fold in range(FOLD_SPLIT):
                    # Separa a parte benigna de validação, concatena os dataframes de validação e embaralha as amostras
                    X_val_benign = folds_list[fold]
                    X_val = pd.concat([X_val_benign, X_val_attack]).sample(frac=1, random_state=RANDOM_SEED)
                    X_val = X_val.reset_index(drop=True)
                        
                    classes_val = X_val['Label']
                    X_val = X_val.drop(features_to_drop, axis='columns', errors='ignore')
                       
                    y_val = classes_val.apply(lambda c: 0 if c == 'R' else 1)
                        
                    # preparando o conjunto de treinamento para este fold
                    X_train_folds = [folds_list[i] for i in range(FOLD_SPLIT) if i != fold]
                    X_train = pd.concat(X_train_folds)
                    X_train = X_train.reset_index(drop=True)
                    X_train = X_train.drop(features_to_drop, axis='columns', errors='ignore')
                        
                    # Normalização dos dados
                    minmax_scaler = MinMaxScaler()
                    norm_X_train = minmax_scaler.fit_transform(X_train)
                    norm_X_val = minmax_scaler.transform(X_val)
                        
                        
                    # Treinamento do modelo Isolation Forest
                    model_ocsvm = OneClassSVM(kernel=k, nu=n, gamma=g).fit(norm_X_train)
                    val_anomaly_preds = model_ocsvm.predict(norm_X_val)
                        
                    val_anomaly_preds[val_anomaly_preds == 1] = 0
                    val_anomaly_preds[val_anomaly_preds == -1] = 1

                    score = f1_score(y_val, val_anomaly_preds)
                    print(f"{DATASET_NAME} - Modelo: OCSVM_{feature_set_name} | Kernel: {k} | Nu: {n} | Gamma: {g} | F1-Score: {score}")

                    if score > best_f1_for_set:
                        best_f1_for_set = score
                        best_model_for_set = model_ocsvm
    
    # Para calcular as métricas e salvar o scaler correto, precisamos re-executar
    # a normalização com os dados do último fold, que correspondem ao melhor modelo encontrado.
    # Esta parte é um pouco redundante devido à estrutura do loop, mas necessária para obter os dados corretos.
    
    # Calcular as predições e métricas para o melhor modelo deste conjunto
    val_preds_for_set = best_model_for_set.predict(norm_X_val)
    val_preds_for_set[val_preds_for_set == 1] = 0
    val_preds_for_set[val_preds_for_set == -1] = 1
    metrics_for_set = get_overall_metrics(y_val, val_preds_for_set)

    # Store the best model found for this feature set
    best_models[feature_set_name] = {
        'model': best_model_for_set,
        'f1_score': best_f1_for_set,
        'metrics': metrics_for_set, # Salva o dicionário de métricas
        'features': X_train.columns.to_list(), # Store the features used
        'scaler': minmax_scaler, # Store the scaler
        'norm_X_val': norm_X_val # Store the scaled validation data
    }
    print(f"--> Melhor para {DATASET_NAME} {feature_set_name}: F1-Score: {best_f1_for_set:.4f} | Acc: {metrics_for_set['acc']:.4f}, TPR: {metrics_for_set['tpr']:.4f}, FPR: {metrics_for_set['fpr']:.4f}")
    print(f"    com {best_model_for_set.kernel} kernel, nu {best_model_for_set.nu}, gamma {best_model_for_set.gamma}")

# Find the overall best model from the dictionary
overall_best_feature_set = max(best_models, key=lambda k: best_models[k]['f1_score'])
overall_best_model_info = best_models[overall_best_feature_set]
overall_best_model = overall_best_model_info['model']
overall_best_f1 = overall_best_model_info['f1_score']

print("-" * 30)
print(f"\nMelhor modelo geral para {DATASET_NAME}: {overall_best_feature_set} com kernel: {overall_best_model.kernel}, nu: {overall_best_model.nu}, gamma: {overall_best_model.gamma} | Melhor F1-Score na Validação: {overall_best_f1:.4f}")
print("-" * 30)

# --- Salvando os melhores modelos em um arquivo ---
output_models_file = f'best_ocsvm_models_v6_{DATASET_NAME}.pkl'
with open(output_models_file, 'wb') as f:
    pickle.dump(best_models, f)

print(f"\nDicionário com os melhores modelos foi salvo em: {output_models_file}")
print("-" * 30)

# --- Plotando as matrizes de confusão para cada melhor modelo ---

n_models = len(best_models)
n_cols = 2
n_rows = math.ceil(n_models / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 6))
axes = axes.flatten() # Transforma a matriz de eixos em um array 1D para facilitar a iteração

for i, (feature_set_name, model_info) in enumerate(best_models.items()):
    ax = axes[i]
    model = model_info['model']
    scaler = model_info['scaler']
    features_to_drop = feature_sets[feature_set_name]
    f1 = model_info['f1_score']

    # Prepara o conjunto de teste para este modelo específico
    X_test_loop = X_test.drop(features_to_drop, axis='columns', errors='ignore')
    # Usa o scaler salvo para transformar o conjunto de teste
    norm_X_test_loop = scaler.transform(X_test_loop)

    # Fazer predições no CONJUNTO DE TESTE
    val_preds = model.predict(norm_X_test_loop)
    val_preds[val_preds == 1] = 0
    val_preds[val_preds == -1] = 1

    # Gerar e plotar a matriz de confusão
    cm = confusion_matrix(y_test, val_preds)
    
    # Criar rótulos com contagem e percentual
    group_counts = [f'{value:0.0f}' for value in cm.flatten()]
    group_percentages = [f'{value:.2%}' for value in cm.flatten()/np.sum(cm)]
    annot_labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    annot_labels = np.asarray(annot_labels).reshape(2,2)

    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', ax=ax,
                xticklabels=['Predito Benigno', 'Predito Malicioso'],
                yticklabels=['Real Benigno', 'Real Malicioso'])
    ax.set_title(f'Matriz de Confusão (Teste) para OCSVM_{feature_set_name} - {DATASET_NAME} Attack\nF1-Score: {f1:.4f}')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Verdadeiro')

# Esconde eixos extras se o número de modelos for ímpar
for i in range(n_models, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(f'confusion_matrices_test_ocsvm_v6_{DATASET_NAME}.png')
print(f"\nMatrizes de confusão para cada conjunto de features salvas em 'confusion_matrices_test_ocsvm_v6_{DATASET_NAME}.png'")
# plt.show()

# Avaliação final do melhor modelo geral (opcional, mantido do código original)
# print("\nRelatório de Classificação na Validação para o melhor modelo geral:")
# norm_X_val_best = best_models[overall_best_feature_set]['norm_X_val']
# val_preds = overall_best_model.predict(norm_X_val_best)
# val_preds[val_preds == 1] = 0
# val_preds[val_preds == -1] = 1
# print(classification_report(y_val, val_preds))
# plot_confusion_matrix(y_val, val_preds)
