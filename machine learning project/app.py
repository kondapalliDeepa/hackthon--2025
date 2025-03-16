from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_page")
def predict_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
       
        features = [float(request.form[key]) for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]

       
        input_scaled = scaler.transform([features])


        prediction = model.predict(input_scaled)[0]

        return render_template("index.html", prediction_text=f"Predicted Species: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
