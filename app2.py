from flask import Flask, render_template, request
import pickle
import numpy as np

app2 = Flask(__name__)

# Load models
logistic = pickle.load(open('models/logistic.pkl', 'rb'))
scaler2 = pickle.load(open('models/scaler2.pkl', 'rb'))

@app2.route('/predictdatapoint', methods=['POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extract values from form
            age = float(request.form.get('age'))
            sex = float(request.form.get('sex'))
            cp = float(request.form.get('cp'))
            trestbps = float(request.form.get('trestbps'))
            chol = float(request.form.get('chol'))
            fbs = float(request.form.get('fbs'))
            restecg = float(request.form.get('restecg'))
            thalach = float(request.form.get('thalach'))
            exang = float(request.form.get('exang'))
            oldpeak = float(request.form.get('oldpeak'))
            slope = float(request.form.get('slope'))
            ca = float(request.form.get('ca'))
            thal = float(request.form.get('thal'))

            # Create input array
            features = np.array([[age, sex, cp, trestbps, chol, fbs,
                                  restecg, thalach, exang, oldpeak, slope, ca, thal]])

            # Scale and predict
            scaled_features = scaler2.transform(features)
            result = logistic.predict(scaled_features)[0]

            # Show result
            message = "Heart Disease Detected (1)" if result == 1 else "No Heart Disease (0)"
            return render_template('home2.html', result=message)

        except Exception as e:
            return render_template('home2.html', result=f"Error: {e}")
    
    return render_template('home2.html', result=None)

# Home page route
@app2.route('/')
def home():
    return render_template('home2.html')

if __name__ == "__main__":
    app2.run(debug=True, host="0.0.0.0")
