from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('fish_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture form data
        weight = request.form.get('weight')
        length1 = request.form.get('length1')
        length2 = request.form.get('length2')
        length3 = request.form.get('length3')
        height = request.form.get('height')
        width = request.form.get('width')
        
        # Check if any field is empty
        if not all([weight, length1, length2, length3, height, width]):
            return render_template('index.html', prediction_text='Please fill out all fields with valid numbers.')
        
        # Convert form data to float and create features list
        features = [float(weight), float(length1), float(length2), float(length3), float(height), float(width)]
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        return render_template('index.html', prediction_text='Predicted Fish Species: {}'.format(prediction[0]), request=request)
    except ValueError:
        return render_template('index.html', prediction_text='Please enter valid numbers.')
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
