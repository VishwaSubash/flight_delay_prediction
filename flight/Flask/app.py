from flask import Flask, request, render_template
import joblib

# Load trained machine learning model
clf = joblib.load('flight_delay_prediction_model.joblib')

# Initialize Flask app
app = Flask(__name__)

# Define app routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from web form
    departure_time = request.form.get('departure_time')
    arrival_time = request.form.get('arrival_time')
    carrier = request.form.get('carrier')
    distance = request.form.get('distance')

    # Make prediction using machine learning model
    prediction = clf.predict([[departure_time, arrival_time, carrier, distance]])

    # Render prediction result in HTML template
    return render_template('prediction.html', prediction=prediction[0])

# Run app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
