# Entrenamiento y test

## ✍️ ¿Qué és?
> El conjunto de datos de entrenamiento y test es una técnica de validación de modelos de aprendizaje automático 
> que se utiliza para evaluar el rendimiento de un modelo en datos no vistos.


## 📖 Funcionamiento

```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```