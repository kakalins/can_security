import pandas as pd
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Caminho para o arquivo salvo
file_path = 'C:/Users/ricardo.mota/OneDrive - SENAI-PE/Documentos/Doutorado/Disciplina Segurança com IA/can_security/dataset/processed_can_data.pkl'

# Carrega o DataFrame
df = pd.read_pickle(file_path)

# Agora você pode usar o df_loaded diretamente
print("DataFrame carregado com sucesso!")
#print(df.info())
#print(df.head()) 

df_aux = df.copy()
df_aux['Label_binary'] = df['Label'].apply(lambda Label: 'Malicious' if Label != 'T' else 'Benign')

fig = px.histogram(df_aux, x='Label_binary', color='Label_binary', title='Contagem de amostras por classe binária',
             color_discrete_map={'Benign':px.colors.qualitative.Plotly[0],      # Blue
                                 'Malicious':px.colors.qualitative.Plotly[1]}   # Red
             )
fig.show()

fig = px.histogram(df_aux.query('Attack != "Benign"'),
            x='Attack',
            color='Attack',
            category_orders={'Attack':df_aux['Attack'].value_counts().index.to_list()},
            title='Contagem de amostras por classe de ataque'
            )
fig.show()