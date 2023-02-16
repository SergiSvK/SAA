import pandas as pd

from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("../../Datasets/movies.csv")
print(df)

# nulos
print(df.isnull().sum())

# Seleccionar las columnas numéricas sólo.
df = df.select_dtypes(include=['float64', 'int64'])

# Eliminar los valores nulos y reemplazarlos con 0, pista, usar los metodos select_dtypes y fillna
df = df.select_dtypes(include=['float64', 'int64']).fillna(0)

# Crear un modelo de regresion lineal (con scikit-learn) y entrenarlo en el dataset numéricos. La variable objetivo
# es ventas Asignar una columna nueva con las predicciones al dataset original llamada ventas_pred

reg = linear_model.LinearRegression()
reg.fit(df.drop("ventas", axis="columns"), df.ventas)

df["ventas_pred"] = reg.predict(df.drop("ventas", axis="columns"))

# Comparar las ventas reales
print(df[["ventas", "ventas_pred"]])

# Graficar las ventas reales vs las predichas
plt.scatter(df.ventas, df.ventas_pred)
plt.xlabel("ventas reales")
plt.ylabel("ventas predichas")
plt.show()

# hist to csv
df.to_csv("data/movies_pred.csv", index=True)

# nombre de las columnas
print(df.columns)


