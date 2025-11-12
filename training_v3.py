import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix
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

X_train = df_train.drop(['Timestamp', 'Label', 'Attack'], axis='columns')

X_val_benign = df_val_test.query('Label == "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)
X_val_attack = df_val_test.query('Label != "R"').sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)

df_val_test = df_val_test.drop(X_val_benign.index)
df_val_test = df_val_test.drop(X_val_attack.index)

df_val_test = df_val_test.reset_index(drop=True)

# Concatena os dataframes de validação e embaralha as amostras
X_val = pd.concat([X_val_benign, X_val_attack]).sample(frac=1, random_state=RANDOM_SEED)
X_val = X_val.reset_index(drop=True)
classes_val = X_val['Label']
X_val = X_val.drop(['Timestamp', 'Label','Attack'], axis='columns')
classes_test = df_val_test['Label']
X_test = df_val_test.drop(['Timestamp', 'Label', 'Attack'], axis='columns')

y_val, y_test = classes_val.apply(lambda c: 0 if c == 'R' else 1), classes_test.apply(lambda c: 0 if c == 'R' else 1)

# Usando MinMax Scaler dessa vez para que a rede neural seja capaz de gerar saídas no intervalo numérico da função sigmóide

minmax_scaler = MinMaxScaler()
minmax_scaler = minmax_scaler.fit(X_train)

norm_X_train = minmax_scaler.transform(X_train)
norm_X_val = minmax_scaler.transform(X_val)
norm_X_test = minmax_scaler.transform(X_test)

del X_train, X_val, X_test

print(f"Formato X_train (só normais): {norm_X_train.shape}")
print(f"Formato X_val (misto 50/50): {norm_X_val.shape}")
print(f"Formato X_test (misto restante): {norm_X_test.shape}")
print("-" * 30)

_n_estimators = [100, 200, 300, 350, 355]
_contamination = [0.0078, 0.008, 0.0082, 0.0085, 0.0086, 0.0088, 0.009, 0.0092]
best_f1 = 0
best_model = None
val_preds = None

# Tunagem de hiperparâmetros do Isolation Forest
print("\nIniciando a tunagem de hiperparâmetros do Isolation Forest...")
print("-" * 30)
for n in _n_estimators:
    for c in _contamination:
        model_iforest = IsolationForest(n_estimators = n, contamination = c, random_state=RANDOM_SEED).fit(norm_X_train)

        val_anomaly_preds = model_iforest.predict(norm_X_val)

        val_anomaly_preds[val_anomaly_preds == 1] = 0
        val_anomaly_preds[val_anomaly_preds == -1] = 1

        score = f1_score(y_val, val_anomaly_preds)
        print(f"Estimator: {n} | Contamination: {c} | F1-Score: {score}")

        if score > best_f1:
            best_f1 = score
            best_model = model_iforest
            val_preds = val_anomaly_preds
            print(f"--> Novo melhor modelo encontrado! F1-Score: {best_f1}, Estimators: {n}, Contamination: {c}")

print("-" * 30)
print(f"\nMelhor modelo com estimator: {best_model.n_estimators}, contamination: {best_model.contamination} | Melhor F1-Score na Validação: {best_f1}")
print("-" * 30)
print("Relatório de Classificação na Validação:")
print(classification_report(y_val, val_preds))
plot_confusion_matrix(y_val, val_preds)

# Testando o melhor modelo no conjunto de teste
print("-" * 30)
print("Avaliação do Melhor Modelo no Conjunto de Teste:")
test_anomaly_preds = best_model.predict(norm_X_test)
test_anomaly_preds[test_anomaly_preds == 1] = 0
test_anomaly_preds[test_anomaly_preds == -1] = 1

print("-" * 30)
print("Relatório de Classificação no Conjunto de Teste:")
print(classification_report(y_test, test_anomaly_preds))
plot_confusion_matrix(y_test, test_anomaly_preds)