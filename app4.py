from flask import Flask, render_template, request
import pickle
import numpy as np

app4 = Flask(__name__)

# Load model and scaler
model = pickle.load(open('models/attrition_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler6.pkl', 'rb'))

# Manual encoding dictionary
enc = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
    "Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4,
    "Low": 1, "Medium": 2, "High": 3, "Very High": 4, "Average": 2,
    "Single": 0, "Married": 1, "Divorced": 2,
    "Entry": 0, "Mid": 1, "Senior": 2,
    "Small": 0, "Medium": 1, "Large": 2,
    "Education": 0, "Healthcare": 1, "Technology": 2, "Media": 3
}

@app4.route('/')
def home():
    return render_template('home4.html')

@app4.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all 23 inputs
        age = int(request.form['age'])
        gender = enc[request.form['gender']]
        years = int(request.form['years_at_company'])
        job_role = enc[request.form['job_role']]
        income = int(request.form['monthly_income'])
        work_life = enc[request.form['work_life_balance']]
        satisfaction = enc[request.form['job_satisfaction']]
        rating = enc[request.form['performance_rating']]
        promotions = int(request.form['number_of_promotions'])
        overtime = enc[request.form['overtime']]
        distance = int(request.form['distance_from_home'])
        education = int(request.form['education_level'])
        marital = enc[request.form['marital_status']]
        dependents = int(request.form['number_of_dependents'])
        job_level = enc[request.form['job_level']]
        company_size = enc[request.form['company_size']]
        tenure = int(request.form['company_tenure'])
        remote = enc[request.form['remote_work']]
        leadership = enc[request.form['leadership_opportunities']]
        innovation = enc[request.form['innovation_opportunities']]
        reputation = enc[request.form['company_reputation']]
        recognition = enc[request.form['employee_recognition']]
        flexibility = enc[request.form['workplace_flexibility']]

        input_data = [
            age, gender, years, job_role, income, work_life, satisfaction,
            rating, promotions, overtime, distance, education, marital,
            dependents, job_level, company_size, tenure, remote, leadership,
            innovation, reputation, recognition, flexibility
        ]

        # Scale and predict
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]

        result = "Left" if prediction == 1 else "Stayed"
        return render_template('home4.html', prediction=result)

    except Exception as e:
        return f"Something went wrong: {str(e)}"

if __name__ == '__main__':
    app4.run(debug=True)
