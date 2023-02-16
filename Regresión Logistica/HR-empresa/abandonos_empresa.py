
from google.colab import drive
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt

# importar el data set

df = pd.read_csv('/content/drive/MyDrive/Datasets/HR_comma_sep.csv')

print(df.columns)

for i in df.columns:
    plt.title(i)
    pd.crosstab(df[i], df.left).plot(kind='bar')
    plt.show()

# filtramos los datos que necesitamos para el analisis

df = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
         'promotion_last_5years', 'left']]

# target es left

# realizamos one hot encoding para las variables categoricas de Departamento y salario

# salario es ordinal
# department es nominal


# variables X e y
X = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
        'promotion_last_5years']]
y = df['left']

# dividimos el dataset en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
preprocessor = make_column_transformer(
    (ohe, ['salary', 'sales']),
    remainder='passthrough'
)

# aplicamos el preprocesador a los datos de train y test con fit_transform y transform

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.fit_transform(X_test)

# aplicamos el modelo de regresion logistica

model = LogisticRegression()

model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)

# score del modelo

model.score(X_test_transformed, y_test)

