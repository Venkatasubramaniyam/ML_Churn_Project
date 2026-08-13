from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
from openai import OpenAI

app = Flask(__name__)

# Load ML model and scaler
model = pickle.load(open("models/model_Log.sav", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# OpenAI client. The API key must be configured as an environment variable
# in Render, not hard-coded in this source code.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHATBOT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """
You are the AI assistant for a Telecom Customer Churn Prediction website.

Your job is to help users understand:
- customer churn
- the inputs used by this application: pincode, estimated salary,
  calls made, SMS sent, and data used
- the meaning of "Churn Detected" and "No Churn"
- practical ways telecom companies can reduce customer churn

Keep answers concise, clear, and business-friendly.

Do not claim that a specific customer will churn unless the application's
prediction result is explicitly provided to you. Do not invent model
accuracy, probabilities, customer information, or business facts.

If asked about something unrelated to telecom churn or this application,
politely explain that you are focused on helping with the churn prediction
application.
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    pincode = float(request.form["pincode"])
    estimated_salary = float(request.form["estimated_salary"])
    calls_made = float(request.form["calls_made"])
    sms_sent = float(request.form["sms_sent"])
    data_used = float(request.form["data_used"])

    data = pd.DataFrame(
        np.array([[pincode, estimated_salary, calls_made, sms_sent, data_used]]),
        columns=[
            "pincode",
            "estimated_salary",
            "calls_made",
            "sms_sent",
            "data_used",
        ],
    )

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        result = "Churn Detected"
    else:
        result = "No Churn"

    return render_template("result.html", prediction=result)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({
            "error": "OPENAI_API_KEY is not configured on the server."
        }), 500

    try:
        response = client.responses.create(
            model=CHATBOT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=message,
            max_output_tokens=300,
        )

        return jsonify({"response": response.output_text})

    except Exception as exc:
        app.logger.exception("Chatbot error")
        return jsonify({
            "error": "The chatbot is temporarily unavailable. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
