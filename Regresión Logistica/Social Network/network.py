import pandas as pd
from google.colab import drive
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Datasets/Social_Network_Ads.csv')

df = df.drop(['User ID'], axis=1)

# graficamos los datos

import matplotlib.pyplot as plt

for i in df.columns:
    pd.crosstab(df[i], df['Purchased']).plot(kind='bar')


X = df

y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()

preprocessor = make_column_transformer(
    (ohe, ['Gender', 'sales']),
    remainder='passthrough'
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)

# score del modelo
model.score(X_test_transformed, y_test)

