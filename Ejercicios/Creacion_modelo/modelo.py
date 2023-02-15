import pandas as pd
from sklearn import linear_model
import pickle
import joblib

df = pd.read_csv("homeprices.csv")

df.head()

model = linear_model.LinearRegression()
model.fit(df[['area']], df.price)

model.coef_

model.intercept_

model.predict([[5000]])

with open('model_pickle', 'wb') as file:
    pickle.dump(model, file)

# cargamos el modelo
with open('model_pickle', 'rb') as file:
    mp = pickle.load(file)

mp.coef_
mp.intercept_

mp.predict([[5000]])

# Guardamos el modelo usando joblib

joblib.dump(model, 'model_joblib')

#cargamos el modelo

mj = joblib.load('model_joblib')
