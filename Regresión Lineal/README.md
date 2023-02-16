# Regresión Lineal

## ✍️ ¿Qué és?
> La regresión lineal es un método estadístico que permite predecir el valor de una variable dependiente (Y) a 
> partir de una variable independiente (X). La variable dependiente es la que se quiere predecir y la variable 
> independiente es la que se utiliza para predecir la variable dependiente.


## 📖 Funcionamiento

```python

from sklearn import linear_model

reg = linear_model.LinearRegression()

# Entrenamiento del modelo con los datos de entrenamiento
reg.fit(X_train, y_train)

# Predicción de los datos de test
y_pred = reg.predict(X_test)

# score
reg.score(X_test, y_test)

```