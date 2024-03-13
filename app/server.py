from flask import Flask, request
import pickle
import numpy as np


app = Flask(__name__)

with open('app/models/realty_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)


# домашняя страница 
@app.route('/')
def index():
    return "Сервер запущен!"


# страница с для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    features = np.array(request.json).reshape(1, -1)
    prediction = model.predict(features)
    
    return prediction[0]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 