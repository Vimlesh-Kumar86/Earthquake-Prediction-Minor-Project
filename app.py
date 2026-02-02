from flask import Flask, request, render_template
import pickle
import numpy as np
import datetime

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("earthquake_decision_tree_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = request.form

        # Validate and process inputs
        # magnitude = float(data['magnitude'])
        depth = float(data['depth'])
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        event_time = float(datetime.datetime.strptime(data['time'], '%Y-%m-%d %H:%M:%S').timestamp())

        # Create an input array
        input_features = np.array([[depth, latitude, longitude, event_time]])

        # Make a prediction
        prediction = model.predict(input_features)
        output = "Earthquake" if prediction[0] == 1 else "No Earthquake"

        # Render result
        return render_template('index.html', prediction_text=f'Result: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
