import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model1.bin'
vectorizer_file = 'dv.bin'

with open (model_file, 'rb') as md_file:
    with open (vectorizer_file, 'rb') as dv_file:
        dv = pickle.load(dv_file)
        model = pickle.load(md_file)

app = Flask('predict')

@app.route('/predict', methods = ['POST'])

def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]

    result = {
        'approval_probability': y_pred
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)