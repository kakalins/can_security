import pandas as pd
import numpy as np
from river import anomaly, preprocessing
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle, gc
import os, math
import argparse


# --- Configuração dos Argumentos da Linha de Comando ---
parser = argparse.ArgumentParser(description='Treina modelos Isolation Forest com tunagem de hiperparâmetros.')
parser.add_argument('--complete', action='store_true', 
                    help='Executa o treinamento completo com mais hiperparâmetros e dados. Se não for especificado, executa um teste rápido.')
args = parser.parse_args()
COMPLETE_TRAINING = args.complete

# Use os.path.dirname to get the directory of the current script
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
ABSOLUTE_MODELS_PATH = ABSOLUTE_PATH + 'models/M_CAN/online_if/'
ABSOLUTE_PLOTS_PATH = ABSOLUTE_PATH + 'plots/M_CAN/online_if/'
ABSOLUTE_DATASET_PATH = ABSOLUTE_PATH + 'dataset/preprocessed/M_CAN/'

def get_overall_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Adicionado tratamento para divisão por zero
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # tpr é o mesmo que recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (2 * tpr * precision) / (tpr + precision) if (tpr + precision) > 0 else 0
    return {'acc': acc, 'tpr': tpr, 'recall': tpr, 'fpr': fpr, 'precision': precision, 'f1-score': f1}

if COMPLETE_TRAINING:
    print("--- MODO DE TREINAMENTO COMPLETO ATIVADO ---")
    
    # Garante que o diretório de log exista antes de tentar escrever o arquivo
    os.makedirs(ABSOLUTE_MODELS_PATH + 'complete/', exist_ok=True)
    
    # --- Preparando o arquivo de log para o resumo ---
    log_file_path = os.path.join(ABSOLUTE_MODELS_PATH, 'complete/online_if_summary_M_CAN.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Resumo do treinamento dos modelos Online Isolation Forest\n")
        log_file.write("=" * 40 + "\n")
        log_file.write("--- MODO DE TREINAMENTO COMPLETO ATIVADO ---\n")
        
    # --- 1. Constantes e Configuração ---
    N_TRAINING_SAMPLES = 250000
    N_VAL_SAMPLES = 25000
    RANDOM_SEED = 33    
    
    N_TREES = [50, 100]
    HEIGHTS = [10, 12]
    WINDOW_SIZES = [25000]
else:
    print("--- MODO DE TESTE RÁPIDO ATIVADO ---")
    
    # Garante que o diretório de log exista antes de tentar escrever o arquivo
    os.makedirs(ABSOLUTE_MODELS_PATH + 'fast/', exist_ok=True)
    
    # --- Preparando o arquivo de log para o resumo ---
    log_file_path = os.path.join(ABSOLUTE_MODELS_PATH, 'fast/online_if_summary_M_CAN.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Resumo do treinamento dos modelos Online Isolation Forest\n")
        log_file.write("=" * 40 + "\n")
        log_file.write("--- MODO DE TESTE RÁPIDO ATIVADO ---\n")
        
    # Constantes e Configuração
    N_TRAINING_SAMPLES = 10000
    N_VAL_SAMPLES = 2000
    N_TEST_SAMPLES = 100000
    RANDOM_SEED = 33    
    
    N_TREES = [50, 100, 200] # entre 50 e 256
    HEIGHTS = [8, 10, 12] #entre 8 e 16
    WINDOW_SIZES = [8000] # de 100 a 1000


# Definição dos conjuntos de features a serem testados
feature_sets = {
    # 'F1': ['Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    # 'F2': ['Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F3': ['Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F4': ['Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F5': ['Data_5', 'Data_6', 'Data_7', 'Label'],
    'F6': ['Data_6', 'Data_7', 'Label'],
    'F7': ['Data_7', 'Label'],
    'F8': ['Label']
}

feature_sets_shift = 3

np.random.seed(RANDOM_SEED)

# Tunagem de hiperparâmetros do Isolation Forest
with open(log_file_path, 'a') as log_file:
    log_file.write("\nIniciando a tunagem de hiperparâmetros do Online Isolation Forest\n")
    log_file.write("-" * 30 + "\n")
    log_file.write(f"N_TREES: {N_TREES}\nHEIGHTS: {HEIGHTS}\nWINDOW_SIZES: {WINDOW_SIZES}\n")
    log_file.write("-" * 30 + "\n")
print("\nIniciando a tunagem de hiperparâmetros do Online Isolation Forest...")
print("-" * 30)


# Carregamento e Divisão dos Dados 
# Carrega o DataFrame
df_train = pd.read_pickle(ABSOLUTE_DATASET_PATH + 'normal_train.pkl')
df_val_attack = pd.read_pickle(ABSOLUTE_DATASET_PATH + 'fuzzing_val_attack.pkl')
df_test = pd.read_pickle(ABSOLUTE_DATASET_PATH + 'fuzzing_test.pkl')

# --- Opcional: Reamostragem para testes rápidos dentro do script de treinamento ---
# Se o número de amostras no script for menor que o total no arquivo, reduz o tamanho.
if COMPLETE_TRAINING == False:  
    if len(df_train) > N_TRAINING_SAMPLES:
        print(f"Reduzindo df_train de {len(df_train)} para {N_TRAINING_SAMPLES} amostras.")
        df_train = df_train.sample(n=N_TRAINING_SAMPLES, random_state=RANDOM_SEED)
        print(f"Reduzindo df_val_attack de {len(df_val_attack)} para {N_VAL_SAMPLES} amostras.")
        df_val_attack = df_val_attack.sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)
        print(f"Reduzindo df_test de {len(df_test)} para {N_TEST_SAMPLES} amostras.")
        df_test = df_test.sample(n=N_TEST_SAMPLES, random_state=RANDOM_SEED)

# Define as features que serão usadas no modelo
features = ['Data_0', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7']
X_test = df_test[features]  

# Cria os rótulos para o conjunto de teste (0 para benigno, 1 para anomalia)
y_test_true = df_test['Label'].apply(lambda x: 0 if x == '0' else 1)

# contar numero de benignos e maliciosos no conjunto de teste
total_malicious = (y_test_true == 1).sum()
total_benign = (y_test_true == 0).sum()

# Dicionário para armazenar os melhores modelos e seus resultados por feature_set
best_models = {}

for f_idx in range(len(feature_sets)):
    feature_set_name = f'F{f_idx + feature_sets_shift}'
    features_to_drop = feature_sets[feature_set_name]
    print(f"Testando com o conjunto de features {feature_set_name}...")
    
    # Prepara os dados de treino e teste para o feature set atual
    X_train_fs = df_train.drop(features_to_drop, axis='columns', errors='ignore')
    X_test_fs = X_test.drop(features_to_drop, axis='columns', errors='ignore')

    # Prepara o conjunto de validação
    df_val_benign = df_train.sample(n=len(df_val_attack), random_state=RANDOM_SEED)
    df_val = pd.concat([df_val_benign, df_val_attack]).sample(frac=1, random_state=RANDOM_SEED)
    X_val_fs = df_val.drop(features_to_drop, axis='columns', errors='ignore')
    y_val_true = df_val['Label'].apply(lambda x: 0 if x == '0' else 1)

    best_f1_for_set = -1
    best_model_info_for_set = {}

    for _n_trees in N_TREES:         
        for _height in HEIGHTS:
            for _window_size in WINDOW_SIZES:
                # --- 3. Configuração do Modelo Online ---
                # O modelo HalfSpaceTrees é o equivalente online do Isolation Forest
                model = anomaly.HalfSpaceTrees(
                    n_trees=_n_trees,
                    height=_height,
                    window_size=_window_size,
                    seed=RANDOM_SEED
                )                   

                # O scaler também será atualizado de forma online
                scaler = preprocessing.StandardScaler()

                # --- 4. Treinamento Online ---
                print(f"Treinando M_CAN {feature_set_name} com n_trees={_n_trees}, height={_height}, window_size={_window_size}...")
                # Iteramos sobre as amostras de treino uma a uma
                for _, x in tqdm(X_train_fs.iterrows(), total=X_train_fs.shape[0], leave=False):
                    # Converte a linha do dataframe para um dicionário, que é o formato esperado pelo river
                    x_dict = x.to_dict()
                    # Atualiza o scaler com a amostra atual e a normaliza
                    scaler.learn_one(x_dict)
                    x_scaled = scaler.transform_one(x_dict)
                    # Treina o modelo com a amostra normalizada
                    model.learn_one(x_scaled)
                
                # --- 5. Avaliação no Conjunto de Validação ---
                y_val_scores = []
                for _, x in X_val_fs.iterrows():
                    x_scaled = scaler.transform_one(x.to_dict())
                    score = model.score_one(x_scaled)
                    y_val_scores.append(score)

                anomaly_rate_expected = y_val_true.mean()
                threshold = np.quantile(y_val_scores, 1 - anomaly_rate_expected)
                y_val_pred = [1 if score > threshold else 0 for score in y_val_scores]
                
                current_f1 = f1_score(y_val_true, y_val_pred)
                
                # Calcula todas as métricas para a combinação atual
                metrics_for_set = get_overall_metrics(y_val_true, y_val_pred)
                current_f1 = metrics_for_set['f1-score']
                print(f"F1-Score na validação: {current_f1:.4f}, Acc: {metrics_for_set['acc']:.4f}, Precision: {metrics_for_set['precision']:.4f}, Recall: {metrics_for_set['recall']:.4f}")
                with open(log_file_path, 'a') as log_file:
                        log_file.write(f"F1-Score na validação: {current_f1:.4f}, Acc: {metrics_for_set['acc']:.4f}, Precision: {metrics_for_set['precision']:.4f}, Recall: {metrics_for_set['recall']:.4f}\n")
                      
                if current_f1 > best_f1_for_set:
                    best_f1_for_set = current_f1
                    best_model_info_for_set = {
                        'model': model,
                        'scaler': scaler,
                        'metrics': metrics_for_set, # Salva o dicionário de métricas
                        # O modelo e o scaler são objetos grandes.
                        # Se a memória for um problema extremo, considere salvar o modelo no disco aqui
                        # e carregar depois, em vez de mantê-lo na memória.
                        # Por enquanto, vamos mantê-lo para encontrar o melhor modelo.

                        'f1_score': current_f1,
                        'hyperparameters': {'n_trees': _n_trees, 'height': _height, 'window_size': _window_size},
                        'features': X_train_fs.columns.to_list()
                    }
    
    if best_model_info_for_set:
        # --- Avaliação Imediata no Conjunto de Teste ---
        print(f"--> Avaliando o melhor modelo de {feature_set_name} no conjunto de teste...")
        best_model = best_model_info_for_set['model']
        best_scaler = best_model_info_for_set['scaler']
        
        y_test_scores = []
        for _, x in X_test_fs.iterrows():
            x_scaled = best_scaler.transform_one(x.to_dict())
            score = best_model.score_one(x_scaled)
            y_test_scores.append(score)

        # Usa o mesmo threshold derivado da validação para o teste
        anomaly_rate_expected_val = y_val_true.mean()
        threshold = np.quantile(y_val_scores, 1 - anomaly_rate_expected_val)
        y_test_pred = [1 if score > threshold else 0 for score in y_test_scores]
        
        test_metrics = get_overall_metrics(y_test_true, y_test_pred)
        best_model_info_for_set['test_metrics'] = test_metrics
        best_model_info_for_set['test_predictions'] = y_test_pred

        best_models[feature_set_name] = best_model_info_for_set
        hp = best_model_info_for_set['hyperparameters']
        val_metrics = best_model_info_for_set['metrics']
        
        print(f"--> Melhor para {feature_set_name} (Validação): F1: {val_metrics['f1-score']:.4f}, Acc: {val_metrics['acc']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f} com {hp}")
        print(f"--> Desempenho em Teste para {feature_set_name}: F1: {test_metrics['f1-score']:.4f}, Acc: {test_metrics['acc']:.4f}, Prec: {test_metrics['precision']:.4f}, Rec: {test_metrics['recall']:.4f}")
        
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Melhor modelo para {feature_set_name} (Validação): F1-Score: {val_metrics['f1-score']:.4f}, Acc: {val_metrics['acc']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}\n")
            log_file.write(f"Hyperparameters: {hp}\n")
            log_file.write(f"Desempenho em Teste para {feature_set_name}: F1-Score: {test_metrics['f1-score']:.4f}, Acc: {test_metrics['acc']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}\n")
            log_file.write("-" * 30 + "\n")

    # Libera a memória dos dataframes de validação e treino do feature set atual
    del X_train_fs, X_test_fs, df_val_benign, df_val, X_val_fs, y_val_true
    gc.collect()



# --- 6. Salvando os melhores modelos para o dataset ---
if best_models:
    overall_best_feature_set = max(best_models, key=lambda k: best_models[k]['f1_score'])
    overall_best_model_info = best_models[overall_best_feature_set]
    
    if COMPLETE_TRAINING:
        os.makedirs(ABSOLUTE_MODELS_PATH + 'complete/', exist_ok=True)
        output_model_path = os.path.join(ABSOLUTE_MODELS_PATH, 'complete/best_online_if_models_M_CAN.pkl')
    else:
        os.makedirs(ABSOLUTE_MODELS_PATH + 'fast/', exist_ok=True)
        output_model_path = os.path.join(ABSOLUTE_MODELS_PATH, 'fast/best_online_if_models_M_CAN.pkl')
        
    with open(output_model_path, 'wb') as f:
        # Salva o dicionário completo com os melhores modelos de cada feature set
        pickle.dump(best_models, f)
    
    print("-" * 30)
    print(f"Dicionário com os melhores modelos salvo em: {output_model_path}")
    print(f"O melhor modelo geral foi para o feature set '{overall_best_feature_set}'.")
    print("-" * 30)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\nMelhor modelo geral foi: {overall_best_feature_set} | F1-Score: {overall_best_model_info['f1_score']:.4f} | Acc: {overall_best_model_info['metrics']['acc']:.4f} | Precision: {overall_best_model_info['metrics']['precision']:.4f} | Recall: {overall_best_model_info['metrics']['recall']:.4f}\n")
        log_file.write("-" * 30 + "\n")

# Libera a memória dos DataFrames originais, pois não são mais necessários
del df_train, df_val_attack, X_test
gc.collect()


# --- 7. Plotando Matrizes de Confusão para os melhores modelos de cada feature set ---
print(f"Plotando matrizes de confusão para os melhores modelos de cada feature set de M_CAN.")
n_models = len(best_models)
if n_models > 0:
    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 6))
    axes = axes.flatten()

    for i, (feature_set_name, model_info) in enumerate(best_models.items()):
        ax = axes[i]
        
        # Usa as predições e métricas já calculadas e salvas
        y_test_pred = model_info['test_predictions']
        test_metrics = model_info['test_metrics']
        
        # O y_test_true é o mesmo para todos os feature sets
        total_benign_fs = (y_test_true == 0).sum()
        total_malicious_fs = (y_test_true == 1).sum()

        # Matriz de Confusão
        cm = confusion_matrix(y_test_true, y_test_pred)
        
        # Criar a matriz de percentuais para a cor do heatmap
        cm_percent = np.zeros_like(cm, dtype=float)
        if total_benign_fs > 0:
            cm_percent[0, :] = cm[0, :] / total_benign_fs
        if total_malicious_fs > 0:
            cm_percent[1, :] = cm[1, :] / total_malicious_fs
        
        # Criar rótulos para anotação com contagem absoluta e percentual
        group_counts = [f'{value:0.0f}' for value in cm.flatten()]
        group_percentages = [f'{value*100:.2f}%' for value in cm_percent.flatten()]
        annot_labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        annot_labels = np.asarray(annot_labels).reshape(2,2)
        
        title = (f'Teste: OnlineIF_{feature_set_name} - M_CAN Attack\n'
                 f"F1: {test_metrics['f1-score']:.2f} | Acc: {test_metrics['acc']:.2f} | Prec: {test_metrics['precision']:.2f} | Rec: {test_metrics['recall']:.2f}")
        
        sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues', ax=ax,
                    xticklabels=['Predito Benigno', 'Predito Malicioso'],
                    yticklabels=['Real Benigno', 'Real Malicioso'])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')

    # Esconde eixos extras
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if COMPLETE_TRAINING:
        os.makedirs(ABSOLUTE_PLOTS_PATH + 'complete/', exist_ok=True)
        output_plot_path = os.path.join(f'{ABSOLUTE_PLOTS_PATH}complete/confusion_matrices_online_if_M_CAN.png')
    else:
        os.makedirs(ABSOLUTE_PLOTS_PATH + 'fast/', exist_ok=True)
        output_plot_path = os.path.join(f'{ABSOLUTE_PLOTS_PATH}fast/confusion_matrices_online_if_M_CAN.png')
    
    plt.savefig(output_plot_path)
    print(f"Matrizes de confusão salvas em '{output_plot_path}'")

print("\nProcesso concluído para todos os datasets.")
print("-" * 30)