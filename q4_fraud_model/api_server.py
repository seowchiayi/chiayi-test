from flask import Flask, request, jsonify
from q4_fraud_model import fraudmodel
import pandas as pd

app = Flask(__name__)


@app.route("/health", methods=['GET'])
def get():
    return "model is up!"


@app.route("/predict", methods=['POST'])
def post():
    res = {}

    try:
        csv_file = request.files["csv"]
        get_csv_file = pd.read_csv(csv_file)
        acc_dict = fraudmodel.predict(get_csv_file)

        res.update(acc_dict)

    except Exception as e:
        return e.with_traceback()

    return jsonify(acc_dict)


if __name__ == "__main__":
    app.run(port=8010)
