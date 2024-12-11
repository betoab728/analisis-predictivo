import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from data_processing import get_health_data  # Importa la función que obtiene los datos

# Ruta para guardar el modelo
MODEL_PATH = 'models/growth_model.pkl'

def train_growth_model(data):
    """
    Entrena un modelo de regresión lineal para predecir el peso del ganado.
    
    Parámetros:
        data (DataFrame): Un DataFrame que contiene las columnas 'dias', 'temperatura' y 'peso'.

    Retorna:
        model: El modelo entrenado.
    """
    # Extraer las características (X) y la variable objetivo (y)
    X = data[['dias', 'temperatura']].values
    y = data['peso'].values

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X, y)

    # Guardar el modelo entrenado
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

    return model

def load_growth_model():
    """
    Carga el modelo de regresión lineal desde un archivo.

    Retorna:
        model: El modelo cargado.
    """
    try:
        model = joblib.load(MODEL_PATH)
        print("Modelo cargado exitosamente.")
        return model
    except FileNotFoundError:
        print("Modelo no encontrado. Entrena el modelo primero.")
        return None

def predict_growth(especie, dias, temperatura):
    """
    Realiza una predicción del peso del ganado dado los días y la temperatura.

    Parámetros:
        especie (str): Especie del ganado (pavo, pollo, pato, etc.)
        dias (int): Días desde la primera fecha registrada.
        temperatura (float): Temperatura corporal del ganado.

    Retorna:
        float: Predicción del peso.
    """
    model = load_growth_model()
    if model is None:
        raise Exception("El modelo no está entrenado. Por favor, entrena el modelo antes de predecir.")
    
    # Preparar los datos de entrada
    X_input = np.array([[dias, temperatura]])
    
    # Hacer la predicción
    peso_predicho = model.predict(X_input)[0]
    return peso_predicho
