from flask import Flask, request, Response
from srv.core import Core

core = Core()
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_covid():
    imagefile = request.files["image"]
    path = "temp/detect.png"
    imagefile.save(path)
    data = core.work(path)
    response = Response(data)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

def run(host="0.0.0.0", port=5000):
    app.run(host, port)