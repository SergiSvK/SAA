# practica 01

from google.colab import drive
from sklearn.impute import SimpleImputer

drive.mount('/content/drive/Dataset/')

# importamos linear regression
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('/content/drive/Dataset/One_hot_encoder.csv')

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

one = OneHotEncoder()

ct = make_column_transformer((one, ['town']), remainder='passthrough')

listos = ct.fit_transform(df)

X, y = listos[:, :-1], listos[:, -1]

model = LinearRegression()
model.fit(X, y)

model.predict(X)

model.predict([[1, 0, 0, 2800]])

# practica 02

data = {"Fare": [7.25, 71.83, 8.05, 53.1, 8.05],
        "Embarked": ['S', 'C', 'S', 'S', 'Q'],
        "Age": [22, 38, 26, 35, 35, "NaN"],
        "Sex": ['F', 'M', 'M', 'F', 'F']
        }


df = pd.DataFrame(data)

X = df[data.keys()]

ohe = OneHotEncoder()
imp = SimpleImputer()

ct = make_column_transformer((ohe, ['Embarked']),
                             (imp, ['Age']),
                             remainder='passthrough')

ct.fit_transform(X)

# OneHotEncoder ordinales

# datos de las camisas

ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(X[["Shape"]])









