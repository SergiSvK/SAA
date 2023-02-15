import pandas as pd

# Cargamos el dataset
df = pd.read_csv('data.csv')

# Mostramos el dataset
print(df)

# Creamos el hot encoding
dummies = pd.get_dummies(df.town)

print(dummies)

# Concatenamos el hot encoding con el dataset
merged = pd.concat([df, dummies], axis='columns')

print(merged)

# Eliminamos la columna town
final = merged.drop(['town'], axis='columns')
print(final)

# Dummy variable trap

final = final.drop(['west windsor'], axis='columns')
print(final)

# Características y etiquetas, X y y respectivamente
X = final.drop(['price'], axis='columns')  # Características
print(X)

# Variable objetivo
y = final.price  # Etiquetas
print(y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Entrenamos el modelo
model.fit(X, y)

# Predecimos el precio de una casa en monroe township
model.predict(X)

# score
model.score(X, y)

# Predecimos el precio de una casa en robbinsville
model.predict([[3400, 0, 0]])
