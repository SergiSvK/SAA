# Dummy Variables

## ✍️ ¿Qué és?
> Las variables dummy son variables que toman valores de 0 o 1 para indicar la presencia o ausencia de un efecto 
> categórico con múltiples niveles. Se utilizan para incluir variables categóricas en un modelo de regresión cuando estas no son numéricas.


## 📖 Funcionamiento

### Uso de Dummies con Scikit-Learn

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Creamos un dataframe de ejemplo
data = pd.DataFrame({
    'precio': [200000, 300000, 250000, 400000],
    'ubicacion': ['norte', 'sur', 'este', 'sur'],
    'habitaciones': [3, 4, 2, 5],
    'piscina': [1, 0, 1, 1]
})
```

La variable "ubicación" es una variable categórica con 3 niveles: norte, sur y este. Para incluir esta variable en un 
modelo de regresión, debemos convertirla en variables dummy. Para ello, utilizaremos la clase OneHotEncoder de la 
librería sklearn.


```python
# Creamos una instancia de la clase OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Convertimos la variable "ubicacion" en variables dummy
dummy = pd.DataFrame(encoder.fit_transform(data[['ubicacion']]).toarray(), columns=encoder.get_feature_names(['ubicacion']))

# Combinamos el dataframe original con las variables dummy
data = pd.concat([data, dummy], axis=1)

# Eliminamos la variable "ubicacion" original y una de las categorías dummy para evitar la multicolinealidad
data = data.drop(['ubicacion', 'ubicacion_norte'], axis=1)

# Mostramos el dataframe final
print(data)

```

### Uso de Dummies con Pandas

Pandas también proporciona una función incorporada llamada `pd.get_dummies()` que se utiliza para crear variables dummy a 
partir de variables categóricas en un dataframe. La sintaxis de `pd.get_dummies()` es la siguiente:

```python
import pandas as pd

data = pd.DataFrame({
    'precio': [200000, 300000, 250000, 400000],
    'ubicacion': ['norte', 'sur', 'este', 'sur'],
    'habitaciones': [3, 4, 2, 5],
    'piscina': [1, 0, 1, 1]
})

# Convertimos la variable "ubicacion" en variables dummy
dummy = pd.get_dummies(data, columns=['ubicacion'], prefix='ubicacion', prefix_sep='_', drop_first=True)

# Combinamos el dataframe original con las variables dummy
data = pd.concat([data, dummy], axis=1)

# Eliminamos la variable "ubicacion" original
data = data.drop(['ubicacion'], axis=1)

# Mostramos el dataframe final
print(data)
```

