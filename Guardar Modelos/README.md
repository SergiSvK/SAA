# Guardar Modelos

## ✍️ ¿Qué és?
> Guardar modelos permite utilizar el modelo entrenado en nuevas 
> predicciones o en futuras mejoras del modelo.


## 📖 Funcionamiento

Guardar el modelo
```python
import joblib
# Guardamos el modelo usando joblib

# Se le pasa el modelo y el nombre del fichero
joblib.dump(model, 'model_joblib')
```

Cargar el modelo
```python
#cargamos el modelo
mj = joblib.load('model_joblib')
```
