from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('../models/insurance_model.pkl')
model_columns = joblib.load('../models/model_columns.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_dict = {
        'age': data['age'],
        'bmi': data['bmi'],
        'children': data['children'],
        'sex_male': 1 if data['sex'] == 'male' else 0,
        'smoker_yes': 1 if data['smoker'] == 'yes' else 0,
        'region_northwest': 1 if data['region'] == 'northwest' else 0,
        'region_southeast': 1 if data['region'] == 'southeast' else 0,
        'region_southwest': 1 if data['region'] == 'southwest' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]

    return jsonify({'predicted_cost': round(prediction, 2)})
if __name__ == '__main__':
    app.run(debug=True, port=5000)