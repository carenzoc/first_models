import numpy as np
from flask import Flask, request, render_template
import pickle
from flask_cors import CORS
import jsonify

app = Flask(__name__)
CORS(app)

model = pickle.load(open("./ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = prediction[0]

        countries = ["Australia", "Canada", "Germany", "UK", "US"]

        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    except Exception as e:
        return jsonify({"status":False,"message":str(e)})

if __name__ == "__main__":
    app.run(debug=True)