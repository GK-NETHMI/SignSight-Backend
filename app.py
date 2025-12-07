from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is working!"})

@app.route("/hello/<name>", methods=["GET"])
def hello(name):
    return jsonify({"message": f"Hello, {name}!"})

@app.route("/add", methods=["POST"])
def add():
    data = request.get_json()
    a = data.get("a", 0)
    b = data.get("b", 0)
    return jsonify({"result": a + b})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# To run the app, use the command: python app.py