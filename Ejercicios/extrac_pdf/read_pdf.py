import requests
import spacy
import re
from pdfminer.high_level import extract_text, extract_pages

url_pdf = "https://greenteapress.com/thinkstats2/thinkstats2.pdf"
nlp = spacy.load('en_core_web_sm')

# descargar el archivo de internet
r = requests.get(url_pdf)
with open('thinkstats2.pdf', 'wb') as f:
    f.write(r.content)

# leer el archivo pdf con pdfminer y extraer el texto a un string
text = extract_text('thinkstats2.pdf')


# limpiar el texto de caracteres especiales y de espacios en blanco
text = re.sub(r'[^a-zA-Z0-9 ]', '', text)


# procesar el texto con spacy
doc = nlp(text)

# extraer los tokens que sean simboles de monedas, guardarlos en un diccionario y contarlos
monedas = {}
for token in doc:
    if token.text in ['$', '€', '£']:
        if token.text in monedas:
            monedas[token.text] += 1
        else:
            monedas[token.text] = 1


url_doc = []

# extraer los tokens que sean urls, guardarlos en una lista
for token in doc:
    if token.like_url:
        url_doc.append(token.text)


ent_list = []

# extraer los tokens que sean entidades de monedas, guardarlos en una lista
for ent in doc.ents:
    # Label MONEY
    if ent.label_ == 'MONEY':
        ent_list.append(ent.text)
