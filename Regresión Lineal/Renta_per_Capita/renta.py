from datetime import time, datetime

import pandas as pd

from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("../../Datasets/renta_per_capita.csv")
print(df)

plt.xlabel("Año")
plt.ylabel("US$")
plt.scatter(df.year, df.renta, color="red", marker="+")
plt.show()

# creamos datos de entrenamiento para sacar la recta de regresión sin saber los años
reg = linear_model.LinearRegression()
reg.fit(df[["year"]], df.renta)

# predicimos la renta a partir del ultimo año de la tabla hasta el año actual
# para ello creamos una tabla con los años que queremos predecir

ultimo_anyo_conocido = df.year.tail(1).values[0]

anyo_actual = datetime.now().year

anyo_extras = 2


# apartir del ultimo año conocido hasta el año actual mas 2 años extras creamos una tabla con los años
# que queremos predecir
anyos = pd.DataFrame({"year": range(ultimo_anyo_conocido, anyo_actual + anyo_extras)})


df2 = pd.DataFrame({"year": [2020, 2021, 2022]})
print(df2)

# predecimos la renta para los años que hemos creado
p = reg.predict(df2)


df2["renta"]= p


df3 = pd.concat([df, df2])


