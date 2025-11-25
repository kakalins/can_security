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

# X_val, X_test = X_val.reset_index(drop=True), X_test.reset_index(drop=True)
# classes_val, classes_test =  classes_val.reset_index(drop=True), classes_test.reset_index(drop=True)

y_val, y_test = classes_val.apply(lambda c: 1 if c == 'R' else -1), classes_test.apply(lambda c: 1 if c == 'R' else -1)

# df_val = X_val.copy()
# df_val['Attack'] = classes_val
# df_val['isAttack'] = df_val['Attack'].apply(lambda l: 0 if l == 'R' else 1)

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

# --- 2. Preparação para o Grid Search com PredefinedSplit ---

# O GridSearchCV precisa de um único X e um único y
# Vamos combinar Treino e Validação
X_combined = np.vstack((norm_X_train, norm_X_val))

# O iForest não usa 'y_train', mas o GridSearchCV precisa dele.
# Criamos labels "dummy" para a parte de treino.
# Usamos 0 (normal) para que o y_combined tenha apenas duas classes: 0 e -1.
y_train_dummy = np.zeros(len(norm_X_train))
y_combined = np.hstack((y_train_dummy, y_val))

# Agora, criamos o índice do PredefinedSplit
# -1 significa "usar para TREINO"
#  0 significa "usar para TESTE (validação) no fold 0"

# Todos os 250.000 índices de treino são -1 (nunca serão usados para validar)
train_indices = np.full(len(norm_X_train), -1, dtype=int)

# Todos os 50.000 índices de validação são 0 (serão usados para validar no 1º e único fold)
validation_indices = np.zeros(len(norm_X_val), dtype=int)

# Combinamos os índices
test_fold = np.hstack((train_indices, validation_indices))
ps = PredefinedSplit(test_fold=test_fold)

print("PredefinedSplit configurado:")
print(f"Total de amostras para o Grid: {len(X_combined)}")
print(f"Tamanho do único fold de treino: {np.sum(test_fold == -1)}")
print(f"Tamanho do único fold de validação: {np.sum(test_fold == 0)}")
print("-" * 30)

# --- 3. Configuração do Grid Search ---

iforest = IsolationForest(random_state=RANDOM_SEED, n_jobs=-1)

# Grade de parâmetros. ATENÇÃO ao 'contamination'
param_grid = {
    'n_estimators': [100, 200, 300, 350, 355],
    # 'max_samples': [0.5, 1.0],
    'contamination': [0.0078, 0.008, 0.0082, 0.0085],
}

# Criamos um scorer F1 que sabe que a classe "Anomalia" (positiva) é -1
# Esta é a parte mais importante!
f1_scorer = make_scorer(f1_score, pos_label=-1)

grid_search = GridSearchCV(
    estimator=iforest,
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=ps, # <-- Aqui está a mágica! Usando seu split, não K-Fold=10
    verbose=2
)

# --- 4. Treinamento ---
print("Iniciando Grid Search...")
# O fit() usará X_train para treinar e X_val para pontuar
grid_search.fit(X_combined, y_combined)

print("-" * 30)
print("Grid Search Concluído.")
print(f"Melhor pontuação F1 na validação: {grid_search.best_score_:.4f}")
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)
print("-" * 30)

# --- 5. Avaliação Final no Conjunto de Teste Real ---
print("Avaliando o melhor modelo no conjunto de TESTE...")

# Pegue o melhor modelo encontrado
best_model = grid_search.best_estimator_

# Preveja no conjunto de teste
y_pred_test = best_model.predict(norm_X_test)

# Mapeando os rótulos para nomes legíveis
target_names = ['Anomalia (-1)', 'Normal (1)']
labels = [-1, 1]

print(classification_report(y_test, y_pred_test, labels=labels, target_names=target_names))

# Gerar e exibir a matriz de confusão
cm = confusion_matrix(y_test, y_pred_test, labels=labels)

# Normaliza a matriz de confusão pelas linhas (pelos valores verdadeiros) para obter percentuais
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Cria os rótulos para as anotações combinando o percentual e a contagem absoluta
annot_labels = (np.asarray(["{0:.2%}\n({1})".format(p, v) for p, v in zip(cm_percent.flatten(), cm.flatten())])).reshape(cm.shape)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Matriz de Confusão no Conjunto de Teste')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()
