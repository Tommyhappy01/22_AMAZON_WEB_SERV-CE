from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open("mymodel.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = {
        "TV":[request.form["TV"]],
        "radio":[request.form["radio"]],
        "newspaper":[request.form["newspaper"]]
        }
    df = pd.DataFrame(data)
    res = model.predict(df) # f"{res:.2f}"
    return render_template("index.html", prediction_text=f"${res[0]:.2f}")

@app.route("/predict_api", methods = ["POST"])
def predict_api():
    data = request.json
    df = pd.DataFrame(data)
    res = model.predict(df.values)
    return {"response":str(res)}

if __name__ == "__main__":
    app.run(debug=True)