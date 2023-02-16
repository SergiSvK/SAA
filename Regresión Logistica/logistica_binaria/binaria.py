import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import math
# drive mount
from google.colab import drive

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Datasets/insurance_data.csv')

# mostrar columnas
print(df.columns)

plt.scatter(df['age'], df['bought_insurance'], marker='+', color='red')

X = df['age']

y = df['bought_insurance']

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# clarification binaria
model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

model.predict_proba(X_test)

model.score(X_test, y_test)

# intercep
model.intercept_

# coeficiente
model.coef_


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# 0.5 = 1 / (1 + e^(-x))


def prediction_function(age, coef, intercept):
    z = coef * age + intercept
    y = sigmoid(z)
    return y


age: int = 25

prediction_function(age, model.coef_[0][0], model.intercept_[0])
