from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))

# Define a function to preprocess the input data


def preprocess_input(n, p, k, temperature, humidity, ph, rainfall):
    data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    df = pd.DataFrame(
        data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    return df

# Define the home page


@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction page


@app.route('/submit', methods=['POST'])
def predict():
    # Get the user inputs
    n = float(request.form['n'])
    p = float(request.form['p'])
    k = float(request.form['k'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Preprocess the input data
    input_data = preprocess_input(n, p, k, temperature, humidity, ph, rainfall)

    # Get the predicted crop
    prediction = model.predict(input_data)

    # Return the prediction as response
    return render_template('index.html', prediction=prediction[0])


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
