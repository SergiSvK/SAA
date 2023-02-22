from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


iris = load_iris()


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logreg.score(X_test, y_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, logreg.predict(X_test))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(cm, figsize=(10, 10))
sns.heatmap(cm, annot=True)

plt.xlabel('Predicted')
plt.ylabel('Actual')

# convertir el dataset en un dataframe

import pandas as pd

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# que porcentaje de valores hay de cada especie

df['target'] = iris.target

df['target'].value_counts()

# en porcentaje
df['target'].value_counts(normalize=True)

# predecir los 10 primeros iteam

logreg.predict(X_test[:10])


# predecio de los 10 primeros iteam y ver la probabilidad de cada uno

logreg.predict_proba(X_test[:10])

# precision del modelo

logreg.score(X_test, y_test)
