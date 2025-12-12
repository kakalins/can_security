# Esta versão do código implementa a tunagem de hiperparâmetros do Isolation Forest
# para diferentes conjuntos de features, salvando o melhor modelo, scores e sua matriz de confusão 
# para cada conjunto de F1 a F8. Além disso, é implementado k-fold manualmente para treinamento e validação.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pickle
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.datasets import make_classification # Para simular os dados
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse


# --- Configuração dos Argumentos da Linha de Comando ---
parser = argparse.ArgumentParser(description='Treina modelos Isolation Forest com tunagem de hiperparâmetros.')
parser.add_argument('--complete', action='store_true', 
                    help='Executa o treinamento completo com mais hiperparâmetros e dados. Se não for especificado, executa um teste rápido.')
parser.add_argument('--feature_set', type=str, default=None,
                    help='Especifica um único conjunto de features para treinar (ex: F1). Se não for especificado, treina todos.')
args = parser.parse_args()
COMPLETE_TRAINING = args.complete
SINGLE_FEATURE_SET = args.feature_set

# Use os.path.dirname to get the directory of the current script
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
DATASET_NAME = ['Fuzzy','gear','RPM']
ABSOLUTE_MODELS_PATH = ABSOLUTE_PATH + 'models/car_hacking/iForest/'
ABSOLUTE_PLOTS_PATH = ABSOLUTE_PATH + 'plots/car_hacking/iForest/'
ABSOLUTE_DATASET_PATH = ABSOLUTE_PATH + 'dataset/preprocessed/car_hacking/'


if COMPLETE_TRAINING:
    print("--- MODO DE TREINAMENTO COMPLETO ATIVADO ---")
    
    # Garante que o diretório de log exista antes de tentar escrever o arquivo
    os.makedirs(ABSOLUTE_MODELS_PATH + 'complete/', exist_ok=True)
    
    # --- Preparando o arquivo de log para o resumo ---
    log_file_path = os.path.join(ABSOLUTE_MODELS_PATH, 'complete/if_summary_car_hacking.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Resumo do treinamento dos modelos Isolation Forest\n")
        log_file.write("=" * 40 + "\n")
        log_file.write("--- MODO DE TREINAMENTO COMPLETO ATIVADO ---\n")
        
    FOLD_SPLIT = 10
    RANDOM_SEED = 33
    FEATURE_SET_SAHIFT = 1  # Options: 'F1' to 'F8'
    
    # Hiperparâmetros para tunagem
    _n_estimators = [200, 300, 384]
    _max_samples = [256]
    _contamination = [0.07, 0.075, 0.08, 0.085, 0.09]
    _bootstrap = [False]
else:
    print("--- MODO DE TESTE RÁPIDO ATIVADO ---")
        
    # Garante que o diretório de log exista antes de tentar escrever o arquivo
    os.makedirs(ABSOLUTE_MODELS_PATH + 'fast/', exist_ok=True)
    
    # --- Preparando o arquivo de log para o resumo ---
    log_file_path = os.path.join(ABSOLUTE_MODELS_PATH, 'fast/if_summary_car_hacking.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write("Resumo do treinamento dos modelos Isolation Forest\n")
        log_file.write("=" * 40 + "\n")
        log_file.write("--- MODO DE TESTE RÁPIDO ATIVADO ---\n")
        
    N_TRAINING_SAMPLES = 10000  
    N_VAL_SAMPLES = 1000
    N_TEST_SAMPLES = 100000
    FOLD_SPLIT = 10
    RANDOM_SEED = 33
    FEATURE_SET_SAHIFT = 1  # Options: 'F1' to 'F8'
    
    # Hiperparametros para testar o algoritmo
    _n_estimators = [200, 300, 384]
    _max_samples = [256]
    _contamination = [0.07, 0.075, 0.08, 0.085, 0.09]
    _bootstrap = [True,False]
    
# Definição dos conjuntos de features a serem testados
feature_sets = {
    'F1': ['Timestamp', 'ID', 'DLC', 'Data_1', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F2': ['Timestamp', 'ID', 'DLC', 'Data_2', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F3': ['Timestamp', 'ID', 'DLC', 'Data_3', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F4': ['Timestamp', 'ID', 'DLC', 'Data_4', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F5': ['Timestamp', 'ID', 'DLC', 'Data_5', 'Data_6', 'Data_7', 'Label'],
    'F6': ['Timestamp', 'ID', 'DLC', 'Data_6', 'Data_7', 'Label'],
    'F7': ['Timestamp', 'ID', 'DLC', 'Data_7', 'Label'],
    'F8': ['Timestamp', 'ID', 'DLC', 'Label']
}

np.random.seed(RANDOM_SEED)


def get_overall_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Adicionado tratamento para divisão por zero
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # tpr é o mesmo que recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = (2 * tpr * precision) / (tpr + precision) if (tpr + precision) > 0 else 0
    return {'acc': acc, 'tpr': tpr, 'recall': tpr, 'fpr': fpr, 'precision': precision, 'f1-score': f1}


# Tunagem de hiperparâmetros do Isolation Forest
with open(log_file_path, 'a') as log_file:
    log_file.write("\nIniciando a tunagem de hiperparâmetros do Isolation Forest\n")
    log_file.write("-" * 30 + "\n")
    log_file.write(f"N_Estimators: {_n_estimators}\nMax_Samples: {_max_samples}\nContamination: {_contamination}\nBootstrap: {_bootstrap}\n")
    log_file.write("-" * 30 + "\n")
print("\nIniciando a tunagem de hiperparâmetros do Isolation Forest...")
print("-" * 30)

for dataset in DATASET_NAME:
    # Use a dictionary to store the best model for each feature set
    best_models = {}
    overall_best_f1 = 0
    overall_best_model = None
    overall_best_val_preds = None

    # Carrega o DataFrame
    df_train = pd.read_pickle(ABSOLUTE_DATASET_PATH + dataset + '_train.pkl')
    df_val_attack = pd.read_pickle(ABSOLUTE_DATASET_PATH + dataset + '_val_attack.pkl')
    df_test = pd.read_pickle(ABSOLUTE_DATASET_PATH + dataset + '_test.pkl')

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
        
    # Preparando os conjuntos de teste
    classes_test = df_test['Label']
    X_test = df_test.drop(['Timestamp', 'ID', 'DLC', 'Label'], axis='columns')
    y_test = classes_test.apply(lambda c: 0 if c == 'R' else 1)

    # contar numero de benignos e maliciosos no conjunto de teste
    total_malicious = (y_test == 1).sum()
    total_benign = (y_test == 0).sum()

    # Divisão manual em FOLD_SPLIT folds
    folds_list = np.array_split(df_train, FOLD_SPLIT)
    
    # Determina quais feature sets serão executados
    if SINGLE_FEATURE_SET:
        if SINGLE_FEATURE_SET in feature_sets:
            feature_sets_to_run = [SINGLE_FEATURE_SET]
        else:
            print(f"AVISO: Feature set '{SINGLE_FEATURE_SET}' não encontrado. Ignorando para o dataset {dataset}.")
            feature_sets_to_run = []
    else:
        feature_sets_to_run = list(feature_sets.keys())

    for feature_set_name in feature_sets_to_run:
        features_to_drop = feature_sets[feature_set_name]
        print(f"Testando com o conjunto de features {feature_set_name}...")

        # Reset bests for the current feature set
        best_f1_for_set = 0
        best_model_for_set = None

        for n in _n_estimators:
            for s in _max_samples:
                for c in _contamination:
                    for b in _bootstrap:
                        for fold in range(FOLD_SPLIT):
                            # Separa a parte benigna de validação, concatena os dataframes de validação e embaralha as amostras
                            df_val_benign = folds_list[fold]
                            X_val = pd.concat([df_val_benign, df_val_attack]).sample(frac=1, random_state=RANDOM_SEED)
                            X_val = X_val.reset_index(drop=True)
                            
                            classes_val = X_val['Label']
                            X_val = X_val.drop(features_to_drop, axis='columns', errors='ignore')
                        
                            y_val = classes_val.apply(lambda cc: 0 if cc == 'R' else 1)
                            
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
                            model_iforest = IsolationForest(n_estimators = n, max_samples = s, contamination = c, max_features = 1.0, bootstrap = b, random_state=RANDOM_SEED).fit(norm_X_train)

                            val_anomaly_preds = model_iforest.predict(norm_X_val)
                            val_anomaly_preds[val_anomaly_preds == 1] = 0
                            val_anomaly_preds[val_anomaly_preds == -1] = 1

                            score = f1_score(y_val, val_anomaly_preds)
                            print(f"{dataset} - Modelo: iForest_{feature_set_name} | Estimator: {n} | Max Samples: {s} | Contamination: {c} | Bootstrap: {b} | F1-Score: {score}")
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(f"{dataset} - Modelo: iForest_{feature_set_name} | Estimator: {n} | Max Samples: {s} | Contamination: {c} | Bootstrap: {b} | F1-Score: {score}\n")
                            
                            if score > best_f1_for_set:
                                best_f1_for_set = score
                                best_model_for_set = model_iforest
        
        # Verifica se algum modelo com score > 0 foi encontrado para este feature set
        if best_model_for_set is None:
            print(f"--> Nenhum modelo com F1-Score > 0 foi encontrado para {dataset} {feature_set_name}. Pulando para o próximo.")
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"--> Nenhum modelo com F1-Score > 0 foi encontrado para {dataset} {feature_set_name}. Pulando para o próximo.\n")
            continue # Pula para o próximo feature set


        # Para calcular as métricas e salvar o scaler correto, precisamos re-executar
        # a normalização com os dados do último fold, que correspondem ao melhor modelo encontrado.
        # Esta parte é um pouco redundante devido à estrutura do loop, mas necessária para obter os dados corretos.
        
        # Calcular as predições e métricas de VALIDAÇÃO para o melhor modelo deste conjunto
        val_preds_for_set = best_model_for_set.predict(norm_X_val)
        val_preds_for_set[val_preds_for_set == 1] = 0
        val_preds_for_set[val_preds_for_set == -1] = 1
        val_metrics = get_overall_metrics(y_val, val_preds_for_set)

        # --- Avaliação Imediata no Conjunto de Teste ---
        print(f"--> Avaliando o melhor modelo de {feature_set_name} no conjunto de teste...")
        X_test_fs = X_test.drop(features_to_drop, axis='columns', errors='ignore')
        norm_X_test_fs = minmax_scaler.transform(X_test_fs)
        
        test_preds = best_model_for_set.predict(norm_X_test_fs)
        test_preds[test_preds == 1] = 0
        test_preds[test_preds == -1] = 1
        
        test_metrics = get_overall_metrics(y_test, test_preds)

        # Store the best model found for this feature set
        best_models[feature_set_name] = {
            'model': best_model_for_set,
            'f1_score': best_f1_for_set,
            'metrics': val_metrics, # Salva o dicionário de métricas de VALIDAÇÃO
            'features': X_train.columns.to_list(), # Store the features used
            'scaler': minmax_scaler, # Store the scaler
            'test_metrics': test_metrics, # Salva as métricas de TESTE
            'test_predictions': test_preds # Salva as predições de TESTE
        }
        print(f"--> Melhor para {dataset} {feature_set_name} (Validação): F1: {best_f1_for_set:.4f} | Acc: {val_metrics['acc']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")
        print(f"--> Desempenho em Teste para {dataset} {feature_set_name}: F1: {test_metrics['f1-score']:.4f}, Acc: {test_metrics['acc']:.4f}, Prec: {test_metrics['precision']:.4f}, Rec: {test_metrics['recall']:.4f}")
        print(f"    com {best_model_for_set.n_estimators} estimadores, contaminação {best_model_for_set.contamination}, bootstrap {best_model_for_set.bootstrap}, max_samples {best_model_for_set.max_samples}.")
        print("-" * 30)
        with open(log_file_path, 'a') as log_file:  
            log_file.write(f"Melhor modelo para {dataset} {feature_set_name} (Validação): F1-Score: {best_f1_for_set:.4f} | Acc: {val_metrics['acc']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}\n")
            log_file.write(f"    com {best_model_for_set.n_estimators} estimadores, contaminação {best_model_for_set.contamination}, bootstrap {best_model_for_set.bootstrap}, max_samples {best_model_for_set.max_samples}.\n")
            log_file.write(f"Desempenho em Teste para {dataset} {feature_set_name}: F1-Score: {test_metrics['f1-score']:.4f}, Acc: {test_metrics['acc']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}\n")
            log_file.write("-" * 30 + "\n")
        
    # Find the overall best model from the dictionary
    overall_best_feature_set = max(best_models, key=lambda k: best_models[k]['f1_score'])
    overall_best_model_info = best_models[overall_best_feature_set]
    overall_best_model = overall_best_model_info['model']
    overall_best_f1 = overall_best_model_info['f1_score']

    print("-" * 30)
    print(f"\nMelhor modelo geral para {dataset}: {overall_best_feature_set} com estimator: {overall_best_model.n_estimators}, contamination: {overall_best_model.contamination} | Melhor F1-Score na Validação: {overall_best_f1:.4f}")
    print("-" * 30)
    with open(log_file_path, 'a') as log_file:  
        log_file.write(f"\nMelhor modelo geral para {dataset}: {overall_best_feature_set} com estimator: {overall_best_model.n_estimators}, contamination: {overall_best_model.contamination} | Melhor F1-Score na Validação: {overall_best_f1:.4f}\n")
        log_file.write("-" * 30 + "\n")

    # --- Salvando os melhores modelos em um arquivo ---
    if COMPLETE_TRAINING:
        os.makedirs(ABSOLUTE_MODELS_PATH + 'complete/', exist_ok=True)
        output_models_file = f'{ABSOLUTE_MODELS_PATH}complete/best_iforest_models_car_hacking_{dataset}.pkl'
    else:
        os.makedirs(ABSOLUTE_MODELS_PATH + 'fast/', exist_ok=True)
        output_models_file = f'{ABSOLUTE_MODELS_PATH}fast/best_iforest_models_car_hacking_{dataset}.pkl'
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
        
        # Usa as predições e métricas de TESTE já calculadas e salvas
        test_preds = model_info['test_predictions']
        test_metrics = model_info['test_metrics']
        print(f"Plotando matriz de confusão para {dataset} {feature_set_name} com métricas de teste...")

        # Gerar e plotar a matriz de confusão
        cm = confusion_matrix(y_test, test_preds)
        
        # Criar a matriz de percentuais para a cor do heatmap
        cm_percent = np.zeros_like(cm, dtype=float)
        if total_benign > 0:
            cm_percent[0, :] = cm[0, :] / total_benign
        if total_malicious > 0:
            cm_percent[1, :] = cm[1, :] / total_malicious

        # Criar rótulos para anotação com contagem absoluta e percentual
        group_counts = [f'{value:0.0f}' for value in cm.flatten()]
        group_percentages = [f'{value*100:.2f}%' for value in cm_percent.flatten()]
        annot_labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        annot_labels = np.asarray(annot_labels).reshape(2,2)

        title = (f'Teste: iForest_{feature_set_name} - {dataset} Attack\n'
                 f"F1: {test_metrics['f1-score']:.4f} | Acc: {test_metrics['acc']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f}")
        sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues', ax=ax,
                    xticklabels=['Predito Benigno', 'Predito Malicioso'],
                    yticklabels=['Real Benigno', 'Real Malicioso'])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')

    # Esconde eixos extras se o número de modelos for ímpar
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    # Imprime totais de benignos e maliciosos no conjunto de teste
    print(f"Total benignos no teste: {total_benign}, Total maliciosos no teste: {total_malicious}")
    print("-" * 30)

    plt.tight_layout()
    if COMPLETE_TRAINING:
        os.makedirs(ABSOLUTE_PLOTS_PATH + 'complete/', exist_ok=True)
        plt.savefig(f'{ABSOLUTE_PLOTS_PATH}complete/confusion_matrices_test_if_car_hacking_{dataset}.png')
        print(f"\nMatrizes de confusão para cada conjunto de features salvas em 'plots/car_hacking/complete/confusion_matrices_test_if_car_hacking_{dataset}.png'")
    else:
        os.makedirs(ABSOLUTE_PLOTS_PATH + 'fast/', exist_ok=True)
        plt.savefig(f'{ABSOLUTE_PLOTS_PATH}fast/confusion_matrices_test_if_car_hacking_{dataset}.png')
        print(f"\nMatrizes de confusão para cada conjunto de features salvas em 'plots/car_hacking/fast/confusion_matrices_test_if_car_hacking_{dataset}.png'")

    print("-" * 30)