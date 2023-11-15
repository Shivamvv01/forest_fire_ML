import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app= Flask(__name__)

# import Ridge regresssor model and standrascale model
ridge_model=pickle.load(open('Model/ridge.pkl','rb'))
Standar_scaler=pickle.load(open('Model/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predicate",methods=["GET","POST"])
def predict_datapoint():
    if request.method== "POST":
        Tempreture=float(request.form.get("Tempreture"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        new_data_scale=Standar_scaler.transform([[Tempreture,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scale)
        return render_template("home.html",result=result)

    else:
        return render_template("home.html")
        
if __name__== "__main__" :
    app.run(debug=True)
    