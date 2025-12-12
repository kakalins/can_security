# can_security

Os datasets usados foram:
    https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
    https://ocslab.hksecurity.net/Datasets/m-can-intrusion-dataset
    
Execute os scripts abaixo para o preprocessamento dos datasets e separação em conjunto de treino, validação e teste: 
    preprocess_car_hacking_t.py
    preprocess_M_CAN_dataset.py

Execute os scripts abaixo para o treinamento, validação e teste nos modelos iForest, OCSVM: 
    training_if_car_hacking.py
    training_if_M_CAN.py
    training_ocsvm_car_hacking.py
    training_ocsvm_M_CAN.py

Execute os scripts abaixo para o treinamento, validação e teste no modelo Online iForest: 
    online_if_car_hacking_t.py
    online_if_M_CAN.py