import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from data_processing import get_health_data

MODEL_PATH = 'models/'

def train_models():
    """
    Entrena un modelo de regresión lineal para cada tipo de ganado.
    """
    data = get_health_data()
    tipos_ganado = data['idtipoganado'].unique()

    for tipo in tipos_ganado:
        print(f"Entrenando modelo para el tipo de ganado: {tipo}")
        train_model_for_species(data, tipo)

def train_model_for_species(data, tipo_ganado):
    """
    Entrena un modelo para un tipo específico de ganado.
    """
    # Filtrar los datos por tipo de ganado
    data_tipo = data[data['idtipoganado'] == tipo_ganado]

    if data_tipo.empty:
        print(f"No hay datos suficientes para el tipo de ganado: {tipo_ganado}")
        return

    X = data_tipo[['dias']].values
    y = data_tipo['peso'].values

    model = LinearRegression()
    model.fit(X, y)

    # Guardar el modelo
    model_filename = f'{MODEL_PATH}{tipo_ganado}_growth_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Modelo para tipo de ganado {tipo_ganado} guardado en {model_filename}")

def predict_for_ganado(id_ganado, dias):
    """
    Realiza una predicción del peso para un ganado específico.
    """
    data = get_health_data()
    ganado_data = data[data['idganado'] == id_ganado]

    if ganado_data.empty:
        raise Exception(f"No se encontraron datos para el ganado con ID: {id_ganado}")

    tipo_ganado = ganado_data['idtipoganado'].iloc[0]
    model = load_model_for_species(tipo_ganado)

    if model is None:
        raise Exception(f"No se ha entrenado un modelo para el tipo de ganado con ID: {tipo_ganado}")

    X_input = np.array([[dias]])
    peso_predicho = model.predict(X_input)[0]

    # Generar la respuesta en el formato solicitado
    respuesta = {
        "idganado": int(id_ganado),
        "nombre": ganado_data['nombre'].iloc[0],
        "peso": round(peso_predicho, 2),
        "idtipoganado": int(tipo_ganado)
    }

    return respuesta

def load_model_for_species(tipo_ganado):
    """
    Carga un modelo de predicción para un tipo específico de ganado.
    """
    model_filename = f'{MODEL_PATH}{tipo_ganado}_growth_model.pkl'

    try:
        model = joblib.load(model_filename)
        return model
    except FileNotFoundError:
        print(f"No se encontró un modelo para el tipo de ganado: {tipo_ganado}")
        return None

if __name__ == '__main__':
    # Entrenar modelos para cada tipo de ganado
    train_models()

    # Ejemplo de predicción
    try:
        id_ganado = 1  # ID del ganado específico
        dias = 100  # Días desde el primer registro
        prediccion = predict_for_ganado(id_ganado, dias)
        print("Predicción:", prediccion)
    except Exception as e:
        print("Error:", e)
