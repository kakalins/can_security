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
args = parser.parse_args()
COMPLETE_TRAINING = args.complete

# Use os.path.dirname to get the directory of the current script
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
DATASET_NAME = ['Fuzzy','gear','RPM']

# --- Preparando o arquivo de log para o resumo ---
log_file_path = os.path.join(ABSOLUTE_PATH, 'models/if_summary.txt')
with open(log_file_path, 'w') as log_file:
    log_file.write("Resumo do treinamento dos modelos Isolation Forest\n")
    log_file.write("=" * 40 + "\n")
  
if COMPLETE_TRAINING:
    print("--- MODO DE TREINAMENTO COMPLETO ATIVADO ---")
    with open(log_file_path, 'a') as log_file:
        log_file.write("--- MODO DE TREINAMENTO COMPLETO ATIVADO ---\n")
    FOLD_SPLIT = 10
    RANDOM_SEED = 33
    FEATURE_SET_SAHIFT = 1  # Options: 'F1' to 'F8'
    
    # Hiperparâmetros para tunagem
    _n_estimators = [100, 200, 300, 350, 355]
    _max_samples = [50, 100, 150, 200, 256]
    # _max_samples = [200, 230, 256]
    _contamination = ['auto', 0.0078, 0.008, 0.0082, 0.0085, 0.0086, 0.0088, 0.009, 0.0092, 0.01, 0.05, 0.1]
    # _contamination = ['auto', 0.078, 0.08, 0.082, 0.085, 0.086, 0.088, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    _bootstrap = [True, False]
    
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
else:
    print("--- MODO DE TESTE RÁPIDO ATIVADO ---")
    with open(log_file_path, 'a') as log_file:
        log_file.write("--- MODO DE TESTE RÁPIDO ATIVADO ---\n")
    N_TRAINING_SAMPLES = 10000  
    N_VAL_SAMPLES = 1000
    FOLD_SPLIT = 10
    RANDOM_SEED = 33
    FEATURE_SET_SAHIFT = 1  # Options: 'F1' to 'F8'
    
    # Hiperparametros para testar o algoritmo
    _n_estimators = [200, 355]
    _max_samples = [256]
    _contamination = [0.01, 0.05, 0.086, 0.1, 0.2, 0.3, 0.4, 0.5]
    _bootstrap = [False]
    
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


np.random.seed(RANDOM_SEED)


def get_overall_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    precision = tp/(tp+fp)
    f1 = (2*tpr*precision)/(tpr+precision)
    return {'acc':acc,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1}

# del df, df_val_test

# Tunagem de hiperparâmetros do Isolation Forest
with open(log_file_path, 'a') as log_file:
    log_file.write("\nIniciando a tunagem de hiperparâmetros do Isolation Forest\n")
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
    df_train = pd.read_pickle(ABSOLUTE_PATH + 'dataset/preprocessed/' + dataset + '_train.pkl')
    df_val_attack = pd.read_pickle(ABSOLUTE_PATH + 'dataset/preprocessed/' + dataset + '_val_attack.pkl')
    df_test = pd.read_pickle(ABSOLUTE_PATH + 'dataset/preprocessed/' + dataset + '_test.pkl')

    # --- Opcional: Reamostragem para testes rápidos dentro do script de treinamento ---
    # Se o número de amostras no script for menor que o total no arquivo, reduz o tamanho.
    if COMPLETE_TRAINING == False:  
        if len(df_train) > N_TRAINING_SAMPLES:
            print(f"Reduzindo df_train de {len(df_train)} para {N_TRAINING_SAMPLES} amostras.")
            df_train = df_train.sample(n=N_TRAINING_SAMPLES, random_state=RANDOM_SEED)
            df_val_attack = df_val_attack.sample(n=N_VAL_SAMPLES, random_state=RANDOM_SEED)
        
    # Preparando os conjuntos de teste
    classes_test = df_test['Label']
    X_test = df_test.drop(['Timestamp', 'Label', 'Attack'], axis='columns')
    y_test = classes_test.apply(lambda c: 0 if c == 'R' else 1)

    # contar numero de benignos e maliciosos no conjunto de teste
    total_malicious = (y_test == 1).sum()
    total_benign = (y_test == 0).sum()

    # Divisão manual em FOLD_SPLIT folds
    folds_list = np.array_split(df_train, FOLD_SPLIT)
    
    for f_idx in range(len(feature_sets)):
        feature_set_name = f'F{f_idx + FEATURE_SET_SAHIFT}'
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
        print(f"--> Melhor para {dataset} {feature_set_name}: F1-Score: {best_f1_for_set:.4f} | Acc: {metrics_for_set['acc']:.4f}, TPR: {metrics_for_set['tpr']:.4f}, FPR: {metrics_for_set['fpr']:.4f}")
        print(f"    com {best_model_for_set.n_estimators} estimadores, contaminação {best_model_for_set.contamination}, bootstrap {best_model_for_set.bootstrap}, max_samples {best_model_for_set.max_samples}.")
        print("-" * 30)
        with open(log_file_path, 'a') as log_file:  
            log_file.write(f"Melhor modelo para {dataset} {feature_set_name}: F1-Score: {best_f1_for_set:.4f} | Acc: {metrics_for_set['acc']:.4f}, TPR: {metrics_for_set['tpr']:.4f}, FPR: {metrics_for_set['fpr']:.4f}\n")
            log_file.write(f"    com {best_model_for_set.n_estimators} estimadores, contaminação {best_model_for_set.contamination}, bootstrap {best_model_for_set.bootstrap}, max_samples {best_model_for_set.max_samples}.\n")
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
    output_models_file = f'models/best_iforest_models_v7_{dataset}.pkl'
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
        print(f'Testando o modelo {feature_set_name} no conjunto de teste...')
        val_preds = model.predict(norm_X_test_loop)
        val_preds[val_preds == 1] = 0
        val_preds[val_preds == -1] = 1

        # Gerar e plotar a matriz de confusão
        cm = confusion_matrix(y_test, val_preds)
        
        # Criar a matriz de percentuais para a cor do heatmap
        cm_percent = np.array([
            [cm[0][0]/total_benign, cm[0][1]/total_benign],
            [cm[1][0]/total_malicious, cm[1][1]/total_malicious]
        ])

        # Criar rótulos para anotação com contagem absoluta e percentual
        group_counts = [f'{value:0.0f}' for value in cm.flatten()]
        group_percentages = [f'{value*100:.2f}%' for value in cm_percent.flatten()]
        annot_labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
        annot_labels = np.asarray(annot_labels).reshape(2,2)

        sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues', ax=ax,
                    xticklabels=['Predito Benigno', 'Predito Malicioso'],
                    yticklabels=['Real Benigno', 'Real Malicioso'])
        ax.set_title(f'Matriz de Confusão (Teste) para iForest_{feature_set_name} - {dataset} Attack\nF1-Score: {f1:.4f}')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')

    # Esconde eixos extras se o número de modelos for ímpar
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    # Imprime totais de benignos e maliciosos no conjunto de teste
    print(f"Total benignos no teste: {total_benign}, Total maliciosos no teste: {total_malicious}")
    print("-" * 30)

    plt.tight_layout()
    plt.savefig(f'plots/confusion_matrices_test_if_v7_{dataset}.png')
    print(f"\nMatrizes de confusão para cada conjunto de features salvas em 'plots/confusion_matrices_test_if_v7_{dataset}.png'")
    # plt.show()
    print("-" * 30)
