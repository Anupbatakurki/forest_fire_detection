
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle

application = Flask(__name__) 
app = application 

ridgecv_model = pickle.load(open('models/ridgecv.pkl','rb'))
scaler_model = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods= ['GET','POST'])
def predict():
    if request.method =="POST":
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridgecv_model.predict(new_data_scaled)

        return render_template('predict.html',result=result[0])
    else:
        return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)