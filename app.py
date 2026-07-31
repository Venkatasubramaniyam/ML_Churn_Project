from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("models/model_Log.sav", "rb"))

# Load scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))


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

    #data = np.array([[bgr, bu, sc, pcv, wc]])

    data = pd.DataFrame(
    np.array([[pincode, estimated_salary, calls_made, sms_sent, data_used]]),
    columns=["pincode", "estimated_salary", "calls_made", "sms_sent", "data_used"]
)


    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        result = "Churn Detected"
    else:
        result = "No Churn"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)