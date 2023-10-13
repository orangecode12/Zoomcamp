import pickle
from flask import Flask


model_file = 'model1.bin'
vectorizer_file = 'dv.bin'

with open (model_file, 'rb') as md_file:
    with open (vectorizer_file, 'rb') as dv_file:
        dv = pickle.load(dv_file)
        model = pickle.load(md_file)

customer = {"job": "retired", "duration": 445, "poutcome": "success"}

def predict(customer):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    return y_pred

print("The probability is: ", y_pred)