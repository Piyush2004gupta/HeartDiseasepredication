from flask import Flask, render_template, request
import pickle
import numpy as np

app3 = Flask(__name__)

# Load trained model, scaler, and encoders
model = pickle.load(open('models/flight_price_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler5.pkl', 'rb'))
encoders = pickle.load(open('models/encoders.pkl', 'rb'))

@app3.route('/')
def home():
    return render_template('home3.html')

@app3.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Fetch form inputs
            airline = request.form['airline']
            flight = request.form['flight']
            source_city = request.form['source_city']
            departure_time = request.form['departure_time']
            stops = request.form['stops']
            arrival_time = request.form['arrival_time']
            destination_city = request.form['destination_city']
            travel_class = request.form['class']
            duration = float(request.form['duration'])
            days_left = int(request.form['days_left'])

            # Encode categorical inputs
            airline_enc = encoders['airline'].transform([airline])[0]
            flight_enc = encoders['flight'].transform([flight])[0]
            source_city_enc = encoders['source_city'].transform([source_city])[0]
            departure_time_enc = encoders['departure_time'].transform([departure_time])[0]
            stops_enc = encoders['stops'].transform([stops])[0]
            arrival_time_enc = encoders['arrival_time'].transform([arrival_time])[0]
            destination_city_enc = encoders['destination_city'].transform([destination_city])[0]
            class_enc = encoders['class'].transform([travel_class])[0]

            # Standardize duration
            duration_scaled = scaler.transform([[duration]])[0][0]

            # Arrange features in correct order
            features = np.array([
                airline_enc, flight_enc, source_city_enc, departure_time_enc,
                stops_enc, arrival_time_enc, destination_city_enc, class_enc,
                duration_scaled, days_left
            ]).reshape(1, -1)

           
            predicted_price = model.predict(features)[0]
            output = round(predicted_price, 2)

            return render_template('home3.html', prediction_text=f"Estimated Flight Price: ₹ {output}")

        except Exception as e:
            return render_template('home3.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app3.run(debug=True)
