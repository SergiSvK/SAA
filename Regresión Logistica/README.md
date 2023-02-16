# Regresión Logística

## ✍️ ¿Qué és?
> Aprendizaje supervisado, clasificación binaria (dos clases)
> - Clasificación binaria: 0 o 1, positivo o negativo, verdadero o falso, etc.
> Se entrena a partir de un conjunto de datos de entrenamiento, donde se conocen las etiquetas de clase de cada ejemplo.
> 
## 📖 Funcionamiento

```python
from sklearn.linear_model import LogisticRegression

# clarification binaria
model = LogisticRegression()

# Entrenamiento
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Probabilidad de cada clase
model.predict_proba(X_test)

# Puntuación
model.score(X_test, y_test)
```

> En la regresión logística, la función sigmoide se utiliza para transformar el resultado de la combinación lineal de 
> las variables independientes en una probabilidad entre 0 y 1.


> El `coef` se utilizan para calcular la probabilidad de que la variable objetivo sea igual a 1 para una combinación dada 
> de valores de las variables independientes.

> El `intercept` se utiliza para calcular la probabilidad de que la variable objetivo sea igual a 1 para una combinación


```python
# intercep
model.intercept_

# coeficiente
model.coef_
```

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(data, coef, intercept):
    z = coef * data + intercept
    y = sigmoid(z)
    return y


prediction_function(data, model, model.intercept_)
```