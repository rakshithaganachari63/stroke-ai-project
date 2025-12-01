from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load trained model
model = joblib.load("stroke_model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        gender = data["gender"]
        age = float(data["age"])
        systolic_bp = float(data["systolic_bp"])
        heart_disease = data["heart_disease"]
        ever_married = data["ever_married"]
        work_type = data["work_type"]
        residence_type = data["Residence_type"]
        avg_glucose_level = float(data["avg_glucose_level"])
        bmi = float(data["bmi"])
        smoking_status = data["smoking_status"]

        # --- 1. Build input row for ML model (same columns as dataset) ---
        input_df = pd.DataFrame([{
            "gender": gender,
            "age": age,
            "hypertension": 1 if systolic_bp >= 140 else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status
        }])

        # ML probability of stroke
        proba = float(model.predict_proba(input_df)[0][1])

        # --- 2. Rule-based risk score so you can show Low/Moderate/High ---
        risk_score = 0

        # age
        if age > 60:
            risk_score += 3
        elif age > 45:
            risk_score += 2
        elif age > 30:
            risk_score += 1

        # blood pressure
        if systolic_bp >= 140:
            risk_score += 4

        # heart disease
        if heart_disease == "Yes":
            risk_score += 5

        # glucose
        if avg_glucose_level > 140:
            risk_score += 4
        elif avg_glucose_level > 100:
            risk_score += 2

        # BMI
        if bmi > 30:
            risk_score += 3
        elif bmi >= 25:
            risk_score += 1

        # smoking
        if smoking_status.lower() in ["smokes", "formerly smoked", "formerly_smoked"]:
            risk_score += 2

        # work type (children / never worked usually lower risk)
        if work_type.lower() in ["children", "never_worked"]:
            risk_score -= 1

        # --- 3. Convert risk_score into Low / Moderate / High ---
        if risk_score >= 10:
            risk_level = "High Risk"
        elif risk_score >= 5:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        return jsonify({
            "stroke_probability": round(proba, 4),
            "risk_score": risk_score,
            "risk_level": risk_level
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
