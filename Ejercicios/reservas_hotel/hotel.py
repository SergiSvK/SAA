import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression


df = pd.read_csv('/content/drive/MyDrive/Datasets/Hotel_Reservations.csv')


# nombres de las columnas
print(df.columns)

# eliminar la columna, Booking_ID
df.drop('Booking_ID', axis=1, inplace=True)

# convertir "booking_status" a 0 y 1 con replace
df['booking_status'] = df['booking_status'].replace({'Canceled': 0, 'Not_Canceled': 1})
df.head()


# Crear una instancia de OneHotEncoder para las variables categóricas ordinales
oe = OrdinalEncoder()

# Crear una instancia de OneHotEncoder para las variables categóricas nominales
ohe = OneHotEncoder()

# La diferencia entre usar el make_columns_tranformer es que uno es una función
# par crear un objeto con menos líneas
preprocessor = make_column_transformer(
    (oe, ['type_of_meal_plan', 'room_type_reserved']),
    (ohe, ['market_segment_type']),
    remainder='passthrough'
)

# Variable objetivo estado de la reserva
X = df.drop("booking_status", axis=1)
y = df["booking_status"]

# Dividision de los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()

# Utilizar el preprocessor para transformar los datos antes del entrenamiento
# tanto en el entrenamiento como en el test
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Entrenar el modelo utilizando los datos transformados
model.fit(X_train_transformed, y_train)

y_pred = model.predict(X_test_transformed)

# Score del modelo

model.score(X_test_transformed, y_test)