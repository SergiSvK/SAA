
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
digits = load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])

dir(digits)

print(digits.data[0])

# crear y entrenar el modelo

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

model = LogisticRegression()

model.fit(X_train, y_train)

model.score(X_test, y_test)

# predecir
plt.matshow(digits.images[67])

model.predict([digits.data[67]])

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, model.predict(X_test))

# Visualizar la confusion matrix
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')



