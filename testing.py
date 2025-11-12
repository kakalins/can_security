

# Testando o melhor modelo no conjunto de teste
print("-" * 30)
print("Avaliação do Melhor Modelo no Conjunto de Teste:")
test_anomaly_preds = overall_best_model.predict(norm_X_test)
test_anomaly_preds[test_anomaly_preds == 1] = 0
test_anomaly_preds[test_anomaly_preds == -1] = 1

print("-" * 30)
print("Relatório de Classificação no Conjunto de Teste:")
print(classification_report(y_test, test_anomaly_preds)) 
plot_confusion_matrix(y_test, test_anomaly_preds)