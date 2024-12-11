from flask import Flask, request, jsonify
from predict_model import train_models, predict_for_ganado
from data_processing import get_health_data

app = Flask(__name__)

# Cargar los datos al iniciar el servidor
df = get_health_data()

# Endpoint para entrenar modelos
@app.route('/train', methods=['POST'])
def train():
    try:
        # Entrenar modelos con los datos cargados
        train_models()
        return jsonify({"message": "Modelos entrenados con éxito."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para predecir el peso de un ganado específico
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del cuerpo de la solicitud
        data = request.json
        id_ganado = data.get("idganado")
        dias = data.get("dias")

        if id_ganado is None or dias is None:
            return jsonify({"error": "Por favor, proporciona 'idganado' y 'dias' en el cuerpo de la solicitud."}), 400

        # Realizar la predicción
        prediction = predict_for_ganado(id_ganado, dias)

        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para verificar el estado del servidor
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Servidor en funcionamiento"}), 200

if __name__ == '__main__':
    print("Servidor iniciando con los datos:")
    print(df.head())  # Muestra un resumen inicial de los datos
    app.run(debug=True, host='0.0.0.0', port=5000)
