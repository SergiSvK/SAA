from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Datasets/salaries.csv')

inputs = df.drop('salary_more_then_100k', axis='columns')

target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder

# no es necesario hacer one hot encoding ni dummy variables

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

print(inputs)

inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(inputs_n, target)
model.score(inputs_n, target)

# salario de google, ingeniero, con bacherlor > 100k

model.predict([[2, 1, 0]])


# Arbol de decisiones

from google.colab import drive

drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/Datasets/titanic.csv')

df.head()

df.isnull().sum()

# eliminamos las columnas que no nos sirven passangerid, name, ticket, cabin, embarke

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

# los nulos de la edad los reemplazamos por la media de la edad
df['Age'] = df['Age'].fillna(df['Age'].mean())

inputs = df.drop('Survived', axis='columns')

from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()

inputs['Sex_j'] = le_sex.fit_transform(inputs['Sex'])

inputs = inputs.drop('Sex', axis='columns')

# separar los datos de entrenamiento y de prueba

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs, df['Survived'], test_size=0.2)

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)

model.score(X_test, y_test)











