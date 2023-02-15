# Árboles de Decisiones

## ✍️ ¿Qué és?
> Un árbol de decisión es un modelo predictivo que utiliza datos 
> históricos para identificar patrones y relaciones entre variables 
> y así poder tomar decisiones basadas en ellos.


## 📖 Funcionamiento

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/Datasets/salaries.csv')

df.head()
```

Limpiamos los datos y formateamos los datos de entrada

```python
# Revisión de datos nulos
df.isnull().sum()

# eliminamos las columnas que no nos sirven passangerid, name, ticket, cabin, embarked
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

# los nulos de la edad los reemplazamos por la media de la edad
df['Age'] = df['Age'].fillna(df['Age'].mean())

# guardamos los datos de entrada en una variable inputs, sin la columna de survived
inputs = df.drop('Survived', axis='columns')

# LabelEncoder para convertir los datos de texto a números
le_sex = LabelEncoder()

# convertimos la columna de sexo a números
inputs['Sex_j'] = le_sex.fit_transform(inputs['Sex'])

# eliminamos la columna de sexo (Texto)
inputs = inputs.drop('Sex', axis='columns')
```

Una vez los datos limpios y formateados, podemos entrenar el modelo

```python
# separar los datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(inputs, df['Survived'], test_size=0.2)


# crear el modelo
model = tree.DecisionTreeClassifier()

# entrenamiento del modelo
model.fit(X_train, y_train)

# score del modelo de entrenamiento
model.score(X_test, y_test)
```