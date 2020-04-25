from flask import Flask,render_template,url_for
from flask_bootstrap import Bootstrap
from flask import request
import eda as ed
import mla
import heartriskscript
import heartrisks
app=Flask(__name__)
Bootstrap(app)
@app.route('/')

def index():
    return render_template('index.html')



@app.route('/eda')
def eda():
	return ed.funcy()

@app.route('/ml')
def ml():
	return mla.func()

@app.route('/heartriskform')
def heartriskform():
	return render_template('heartriskform.html')	

@app.route('/heartriskscript', methods=['POST','GET'])
def heartriskscript():
	if request.method== "POST":
		Age = int(request.form["Age"])
		Gender = int(request.form["Gender"])
		ChestPain = int(request.form["ChestPain"])
		RestingBP= int(request.form["RestingBP"])
		Cholestrol =int(request.form["Cholestrol"])
		FastingBloodSugar=int(request.form["FastingBloodSugar"])
		RestingECG=int(request.form["RestingECG"])
		MaxHeartRate=int(request.form["MaxHeartRate"])
		ExerciseIndusedAngina=int(request.form["ExerciseIndusedAngina"])
		Oldpeak=float(request.form["Oldpeak"])
		Slope=int(request.form["Slope"])
		MajorVessels=int(request.form["MajorVessels"])
		Thalassemia=int(request.form["Thalassemia"])
	pred_array=[]
	pred_array.extend([Age,Gender,ChestPain,RestingBP,Cholestrol,FastingBloodSugar,RestingECG,MaxHeartRate,ExerciseIndusedAngina,Oldpeak,Slope,MajorVessels,Thalassemia])
	pred_arr=[]
	pred_arr.append(pred_array)
	import numpy as np 
	import pandas as pd 
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	import warnings
	warnings.filterwarnings("ignore")
	df=pd.read_csv("heart.csv")
	df.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchieved','ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']
	X_data = df.drop(columns=['Target'], axis=1)
	Y = df['Target']
	x_train, x_test, y_train, y_test = train_test_split(X_data,Y,test_size = 0.2,random_state=42)
	rfc = RandomForestClassifier(n_estimators = 50, max_depth = 7)
	rfc.fit(x_train, y_train)
	df2=pd.DataFrame(pred_arr,columns=['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchieved','ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia'])
	rfc_pred = rfc.predict(df2)
	if rfc_pred[0]==1:
		return render_template('predictionran.html')
	if rfc_pred[0]==0:
		return render_template('predictionran1.html')	