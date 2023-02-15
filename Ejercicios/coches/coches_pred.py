# Descarga el archivo adjunto precios_coches.csv. Este archivo contiene precios de venta de coches para 3 modelos
# diferentes. Primero traza los puntos de datos en un gráfico de dispersión para ver si se puede aplicar un modelo de
# regresión lineal. En caso afirmativo, construye un modelo que pueda responder a las siguientes preguntas,

# Predecir el precio de un Mercedes Benz de 4 años con un kilometraje de 45000.
# Predecir el precio de un BMW X5 de 7 años con un kilometraje de 86000.
# Indica la puntuación (precisión) de tu modelo.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('precios_coches.csv')

# Mostramos el dataset
print(df)

# trazamos el grafico de dispersion

# kilometraje vs precio
plt.scatter(df["Kilometraje"], df["Precio venta"], color='red', marker='+')

# edad vs precio
plt.scatter(df["Edad"], df["Precio venta"], color='blue', marker='+')


# Creamos el hot encoding
dummies = pd.get_dummies(df.Modelo)

# Concatenamos el hot encoding con el dataset
merged = pd.concat([df, dummies], axis='columns')

# Eliminamos la columna Modelo
final = merged.drop(['Modelo'], axis='columns')

# Dummy variable trap
final = final.drop(['Mercedes Benz C class'], axis='columns')

# Características y etiquetas, X y y respectivamente
X = final.drop(['Precio venta'], axis='columns')  # Características

# Variable objetivo
y = final['Precio venta']  # Etiquetas

model = LinearRegression()

# Entrenamos el modelo
model.fit(X, y)

# Predecimos el precio de un Mercedes Benz de 4 años con un kilometraje de 45000.
print(model.predict([[45000, 4, 0, 0]]))

# Predecir el precio de un BMW X5 de 7 años con un kilometraje de 86000.
print(model.predict([[86000, 7, 0, 1]]))

# score
print(model.score(X, y))



