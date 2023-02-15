# mount drive
import pandas as pd
from google.colab import drive
import matplotlib.pyplot as plt

drive.mount('/content/drive')

# import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/Datasets/carprices.csv')

# motrar columnas
print(df.columns)

# kilometraje del coche frente al precio de venta
# %matplotlib inline

plt.scatter(df['Mileage'], df['Sell Price($)'])

# años del coche frente al precio de venta

plt.scatter(df['Age(yrs)'], df['Sell Price($)'])

X = df[['Mileage', 'Age(yrs)']]

y = df['Sell Price($)']

# train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# linear regression
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)

clf.predict(X_test)

clf.score(X_test, y_test)
