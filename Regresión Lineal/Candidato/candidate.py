import os

import pandas as pd
from word2number import w2n
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("../../Datasets/hiring.csv")

# si se encuentra vacio se reemplaza por 0
df.experience = df.experience.fillna("zero")



# se convierte a numero
df.experience = df.experience.apply(w2n.word_to_num)

# para la nota del test vacia se reemplaza por la media
median_test_score = df.test_score.median()
df.test_score = df.test_score.fillna(median_test_score)

# vamos a crear un modelo de regresion lineal
reg = linear_model.LinearRegression()

# entrenamos el modelo
reg.fit(df[['experience', 'test_score', 'interview_score']], df['salary'])

# vamos a predecir el salario de un candidato con 2 años de experiencia, 9 en el test y 6 en la entrevista
print(reg.predict([[2, 9, 6]]))

# vamos a predecir el salario de un candidato con 12 años de experiencia, 10 en el test y 10 en la entrevista
print(reg.predict([[12, 10, 10]]))

# en que directorio estoy
print(os.getcwd())

# bajar un directorio anterior al actual
os.chdir("../../Ejercicios")