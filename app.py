from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
print(tf.__version__)


app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path + '/static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    print("Index route was accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = request.form['input']
    input_values = list(map(float, input_data.split(',')))
    
    # Prepare the data for prediction (adjust shape if necessary)
    input_array = np.array(input_values).reshape(1, 1, len(input_values))
    
    # Make prediction
    predicted_price = model.predict(input_array)[0][0]
    
    return render_template('results.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)





