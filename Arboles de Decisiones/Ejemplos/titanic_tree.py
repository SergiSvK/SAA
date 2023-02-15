# Arbol de decisiones

from google.colab import drive
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Datasets/titanic.csv')

df.head()

df.isnull().sum()

# eliminamos las columnas que no nos sirven passangerid, name, ticket, cabin, embarked

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

# los nulos de la edad los reemplazamos por la media de la edad
df['Age'] = df['Age'].fillna(df['Age'].mean())

inputs = df.drop('Survived', axis='columns')

le_sex = LabelEncoder()

inputs['Sex_j'] = le_sex.fit_transform(inputs['Sex'])

inputs = inputs.drop('Sex', axis='columns')

# separar los datos de entrenamiento y de prueba


X_train, X_test, y_train, y_test = train_test_split(inputs, df['Survived'], test_size=0.2)


model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

model.score(X_test, y_test)
