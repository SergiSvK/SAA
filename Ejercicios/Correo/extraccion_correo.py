import spacy

nlp = spacy.load('es_core_news_sm')

# leemos el archivo txt
with open('datos_alumnos.txt', 'r') as f:
    texto = f.read()


# procesamos el texto con spacy y saber si el texto contiene un correo o no
doc = nlp(texto)

# recorremos el texto y extraemos los correos
for token in doc:
    if token.like_email:
        print(token.text)