import spacy

nlp = spacy.load("es_core_news_sm")
doc = nlp("El perro es muy lindo. El gato es muy feo. El perro es muy feo. El gato es muy lindo.")

tokens = [token.orth_ for token in doc]

lexical_tokens = [t.lower() for t in tokens if len(t) > 3 and t.isalpha()]

words = [t for t in lexical_tokens if len(t) > 3 and t.isalpha()]

# lematizar

lemmas = [token.lemma_ for token in doc]

lemmas_no_pron = [token.lemma_ for token in doc if token.pos_ != "PRON"]

import nltk
from nltk import SnowballStemmer

spain_stemmer = SnowballStemmer('spanish')

stems = [spain_stemmer.stem(t) for t in words]

text = '''El municipio disfruta de una temperatura agradable todo el año con una media que oscila los 18 °C.  Es un lugar escogido por jubilados extranjeros para pasar el resto de su vida. El color verde es el que predomina en el paisaje.'''


def tokenizar(texto: str) -> list:
    doc = nlp(texto)
    return [token.orth_ for token in nlp(texto)]


def normalizar(tokens) -> list:
    return [t.lower() for t in tokens if len(t) > 3 and t.isalpha()]


# lematizar tiene que recibir una lista de tokens
def lematizar(tokens) -> list:
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]


def streamming(tokens)-> list:
    return [spain_stemmer.stem(t) for t in tokens]


texto_tokenizado = tokenizar(text)

texto_normalizado = normalizar(texto_tokenizado)

texto_normalizado_str = " ".join(texto_normalizado)

texto_lematizado = lematizar(texto_normalizado)
print(texto_lematizado)

texto_stemming = streamming(texto_normalizado)

# partes de una oración

frase_con_preposiciones = "El perro esta sobre la mesa desde hace 3 horas. Y el gato entre cojines"

doc = nlp(frase_con_preposiciones)

for token in doc:
    print(f"{token.text}: {token.pos_} - {spacy.explain(token.pos_)}")

# contar los pronombres, verbos, adjetivos, etc que haya en la frase

count = doc.count_by(spacy.attrs.POS)



