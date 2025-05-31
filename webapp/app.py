from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('calorie_model.pkl')
except FileNotFoundError:
    print("Error: calorie_model.pkl not found. Please ensure the model file exists.")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            gender = 1 if request.form.get('Gender') == 'Male' else 0
            age = float(request.form.get('Age'))
            height = float(request.form.get('Height'))
            weight = float(request.form.get('Weight'))
            duration = float(request.form.get('Duration'))
            heart_rate = float(request.form.get('Heart_Rate'))
            body_temp = float(request.form.get('Body_Temp'))

            # Create feature array
            features = np.array([[age, height, weight, duration, heart_rate, body_temp, gender]])
            
            # Make prediction
            prediction = model.predict(features)[0]

            # Render result template
            return render_template('result.html', prediction=round(prediction, 2))
        
        except (KeyError, ValueError) as e:
            error = "Please provide valid input for all fields."
            return render_template('index.html', error=error)
    
    # For GET request, render the input form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)