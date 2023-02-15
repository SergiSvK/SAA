import tweepy
import requests
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
import json
import openpyxl

from google.colab import files

keys_temple = {
    "Twitter": {
        "consumer_key": "<tu-token>",
        "consumer_secret": "<tu-token>",
        "access_token": "<tu-token>",
        "access_token_secret": "<tu-token>",
        "bearer_token": "<tu-token>"
    }
}

# creamos un .json y nos descargamos el archivo con el keys_temple con formato json
with open('keys_temple.json', 'w') as f:
    # jugarmos los datos del diccionario en formato json
    json.dump(keys_temple, f, indent=4)

# download the file with the keys_temple and files.download('keys_temple.json')

# load the keys_temple


uploaded = files.upload()

"""Una vez subido el archivo lo leemos y lo pasamos a las variables. 
La variable "uploaded" es un diccionario donde se espera que contenga los nombres de los archivos subidos, y "keys" es una variable donde se almacenará el contenido del archivo JSON una vez cargado.
"""

for fn in uploaded.keys():
    with open(fn) as f:
        keys = json.load(f)

consumer_key = keys['Twitter']['consumer_key']
consumer_secret = keys['Twitter']['consumer_secret']

access_token = keys['Twitter']['access_token']
access_token_secret = keys['Twitter']['access_token_secret']

bearer_token = keys['Twitter']['bearer_token']

client = tweepy.Client(bearer_token=bearer_token,
                       consumer_key=consumer_key,
                       consumer_secret=consumer_secret,
                       access_token=access_token,
                       access_token_secret=access_token_secret,
                       return_type=requests.Response,
                       wait_on_rate_limit=True)

"""## Hacemos la query a twitter"""

profile = input("Cuenta de twitter a buscar: ")

query = f'from: {profile}'

number_results = input("Número de tweets a buscar: ")

# get max. 100 tweets
tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)

"""## Pasamos la respuesta de twitter a un dataframe"""

# Save data as dictionary
tweets_dict = tweets.json()

# Extract "data" value from dictionary
tweets_data = tweets_dict['data']

# Transform to pandas Dataframe
df = pd.json_normalize(tweets_data)

df['text_en'] = df['text'].apply(lambda x: TextBlob(x).translate(from_lang="es", to="en"))

# añadimos la columna con el sentimiento round(2) para que solo tenga 2 decimales
df['sentimientos'] = df['text_en'].apply(lambda x: TextBlob(x).sentiment.polarity).round(2)

# mostrar gráfica scatter, con las fechas creadas y el sentimiento

# convertimos la columna created_at a datetime, dos ultimos digitos del año, mes y dia
df['created_at'] = pd.to_datetime(df['created_at'], format='%m-%d')

plt.scatter(created_at, df['sentimientos'])

plt.title(f'sentimientos de {profile}')

plt.xlabel('Fecha')
plt.ylabel('sentimientos')

# en la x mostramos la primera fecha y la última
plt.xlim(created_at[0], created_at[-1])

plt.show()


# get url of the tweet
def format_url_twit(profile, id_twit):
    return f'https://twitter.com/{profile}/status/{id_twit.values[0]}'


df_sentimientos = df.copy()
df_sentimientos['sentimientos'] = df_sentimientos['sentimientos'].apply(
    lambda x: str(x) + '%' if '%' not in str(x) else x)
# mostramos el dataframe con el sentimiento y el enlace al tweet
df_sentimientos['url_twit'] = df_sentimientos.apply(lambda x: format_url_twit(profile, x['id']), axis=1)

print(df_sentimientos[['text', 'sentimientos', 'url_twit']])

data_now = pd.to_datetime('now').strftime('%Y%m%d_%H%M%S')

format_name = f'{profile}_sentimientos_{data_now}.xlsx'

# descargar el dataframe con solo las columnas que nos interesan, siento text, sentimientos y url_twit sin el index
# en .xlsx
df_sentimientos[['text', 'sentimientos', 'url_twit']].to_excel(format_name, index=False)

# una vez creado el archivo vamos a formatear la columna de url_twit para que sea un enlace
# y que se pueda abrir en el navegador

wp = openpyxl.load_workbook(format_name)

# seleccionamos la hoja y la columna de url_twit para formatearla a enlace clickable
ws = wp['Sheet1']
ws.column_dimensions['C'].width = 50
ws.column_dimensions['C'].number_format = 'hyperlink'

wp.save(format_name)

# guardamos el archivo con el enlace
df.to_excel(format_name, index=False)

# cuál es el mensaje más negativo
tweets_min = df[df['sentimientos'] == df['sentimientos'].min()].id

for tweet in tweets_min:
    print(format_url_twit(profile, tweet))

negative_sentiments = df[df['sentimientos'] < 0]


def get_sentiments(df_type):
    for i in range(len(df_type)):
        sentiment = df_type.iloc[i]['sentimientos']
        id_twit = df_type.iloc[i]['id']
        url = format_url_twit(profile, id_twit)
        print(f'{sentiment}% {url}')


# cuál es el mensaje más positivo
tweets_max = df[df['sentimientos'] == df['sentimientos'].max()].id

for i in range(len(tweets_max)):
    print(format_url_twit(profile, tweets_max.iloc[i]))
