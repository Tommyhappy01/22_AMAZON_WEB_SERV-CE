from flask import Flask, request

app  = Flask(__name__)

@app.route("/")
def home():
    return f"""
<h1>Welcome to my homepage,Tommy</h1>
Flask Server is UP"""

@app.route("/api")
def api():
    name = request.args.get("name")
    age = int(request.args.get("age"))
    return f"your name is {name} and your age is {age}"

@app.route("/<num>")
def number(num):
    return f"you enter {num}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    