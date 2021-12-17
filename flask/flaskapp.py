from flask import Flask, request, render_template

import pandas as pd
import joblib

model = joblib.load(open("model2.pkl","rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return "flask server is running"

@app.route("/api", methods = ["GET", "POST"])
def api():
    if request.method == "GET":
        return "welcome to my API server, Tommy"
    if request.method == "POST":
        data = request.json
        df = pd.DataFrame(data)
        res = model.predict(df.values)
        return {"response":str(res)}
@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        data = {
            "TV":[request.form["TV"]],
            "radio":[request.form["radio"]],
            "newspaper":[request.form["snewspaper"]]
            }
        print("debug starts here ==================")
        print(data)
        df = pd.DataFrame(data)
        print(df)
        res = model.predict(df) # f"{res:.2f}"
        return render_template("index.html", result=f"${res[0]:.2f}")
if __name__ == "__main__":
    app.run(debug=True, port=81)

