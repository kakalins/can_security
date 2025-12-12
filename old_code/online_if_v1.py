import pandas as pd
import numpy as np
from river import anomaly, preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

# --- 1. Constantes e Configuração ---
N_TRAINING_SAMPLES = 250000
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)

# Constrói o caminho para o arquivo de forma dinâmica
# Pega o diretório do script atual e junta com o caminho relativo para o dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dataset', 'Fuzzy.pkl')

# --- 2. Carregamento e Divisão dos Dados ---
print("Carregando o dataset...")
df = pd.read_pickle(file_path)

# Embaralha o dataframe para garantir que a seleção seja aleatória
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Seleciona 250.000 amostras benignas para o treino online
df_train = df[df['Label'] == 'R'].head(N_TRAINING_SAMPLES)

# O restante dos dados será usado para teste
df_test = df.drop(df_train.index)

print(f"Total de amostras: {len(df)}")
print(f"Amostras de treino (benignas): {len(df_train)}")
print(f"Amostras de teste (restante): {len(df_test)}")
print("-" * 30)

# Define as features que serão usadas no modelo
features = ['ID', 'DLC', 'Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7']

X_train = df_train[features]
X_test = df_test[features]

# Cria os rótulos para o conjunto de teste (0 para benigno, 1 para anomalia)
y_test_true = df_test['Label'].apply(lambda x: 0 if x == 'R' else 1)

del df, df_train, df_test # Libera memória

# --- 3. Configuração do Modelo Online ---

# O modelo HalfSpaceTrees é o equivalente online do Isolation Forest
model = anomaly.HalfSpaceTrees(
    n_trees=10,
    height=8,
    window_size=250,
    seed=RANDOM_SEED
)

# O scaler também será atualizado de forma online
scaler = preprocessing.StandardScaler()

# --- 4. Treinamento Online ---
print("Iniciando o treinamento online...")
# Iteramos sobre as amostras de treino uma a uma
for _, x in tqdm(X_train.iterrows(), total=X_train.shape[0]):
    # Converte a linha do dataframe para um dicionário, que é o formato esperado pelo river
    x_dict = x.to_dict()
    
    # Atualiza o scaler com a amostra atual e a normaliza
    scaler.learn_one(x_dict)
    x_scaled = scaler.transform_one(x_dict)
    
    # Treina o modelo com a amostra normalizada
    model.learn_one(x_scaled)

print("Treinamento online concluído.")
print("-" * 30)

# --- 5. Avaliação no Conjunto de Teste ---
print("Avaliando o modelo no conjunto de teste...")
y_test_scores = []

# Itera sobre os dados de teste para obter os scores de anomalia
for _, x in tqdm(X_test.iterrows(), total=X_test.shape[0]):
    x_dict = x.to_dict()
    
    # Apenas normaliza os dados de teste (não aprende com eles)
    x_scaled = scaler.transform_one(x_dict)
    
    # Obtém o score de anomalia para a amostra
    score = model.score_one(x_scaled)
    y_test_scores.append(score)

# --- 6. Análise dos Resultados ---

# Para converter scores em predições (0 ou 1), precisamos de um limiar.
# Uma abordagem comum é usar um percentil dos scores.
# Por exemplo, se esperamos 10% de anomalias, podemos usar o 90º percentil como limiar.
anomaly_rate_expected = y_test_true.mean() # Taxa real de anomalias no teste
threshold = np.quantile(y_test_scores, 1 - anomaly_rate_expected)

print(f"Taxa de anomalia esperada no teste: {anomaly_rate_expected:.2%}")
print(f"Limiar de anomalia definido (quantil): {threshold:.4f}")

# Converte scores em predições binárias
y_test_pred = [1 if score > threshold else 0 for score in y_test_scores]

print("\nRelatório de Classificação no Conjunto de Teste:")
print(classification_report(y_test_true, y_test_pred, target_names=['Benigno', 'Anomalia']))

# Matriz de Confusão
cm = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predito Benigno', 'Predito Malicioso'], yticklabels=['Real Benigno', 'Real Malicioso'])
plt.title('Matriz de Confusão - Online Isolation Forest (Teste)')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')

output_filename = 'online_if_confusion_matrix.png'
plt.savefig(output_filename)
print(f"\nMatriz de confusão salva como '{output_filename}'")