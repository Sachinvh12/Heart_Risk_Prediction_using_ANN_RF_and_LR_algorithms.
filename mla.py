def func():
	import numpy as np 
	import pandas as pd 
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.metrics import confusion_matrix
	from sklearn.model_selection import train_test_split
	from flask import Flask,render_template
	from flask_bootstrap import Bootstrap
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score

	df=pd.read_csv("heart.csv")

	df.columns = ['Age', 'Gender',	'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchieved',
       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']

	
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier

	X_data = df.drop(columns=['Age','Target'], axis=1)
	Y = df['Target']

	Y = ((Y - np.min(Y))/ (np.max(Y) - np.min(Y))).values
	X = ((X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))).values

	x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)



	
	lr = LogisticRegression()
	lr.fit(x_train,y_train)
	lr_pred = lr.predict(x_test)


	lr_accuracy = accuracy_score(y_test, lr_pred)
	lr_precision = precision_score(y_test, lr_pred)
	lr_recall = recall_score(y_test, lr_pred)
	lr_res=[]
	lr_res.append(lr_accuracy*100)
	lr_res.append(lr_precision*100)
	lr_res.append(lr_recall*100)
	cm = confusion_matrix(y_test,lr_pred)
	conf=sns.heatmap(cm,annot=True);
	figure = conf.get_figure()    
	figure.savefig('conflr.png', dpi=400,bbox_inches='tight',transparent=True)
	plt.clf()
    


	rfc = RandomForestClassifier(n_estimators = 50, max_depth = 3)

	rfc.fit(x_train, y_train)
	rfc_pred = rfc.predict(x_test)
	rfc_accuracy = accuracy_score(y_test, rfc_pred)
	rfc_precision = precision_score(y_test, rfc_pred)
	rfc_recall = recall_score(y_test, rfc_pred)
	rfc_res=[]
	rfc_res.append(rfc_accuracy*100)
	rfc_res.append(rfc_precision*100)
	rfc_res.append(rfc_recall*100)

	print('Random Forest Classifier Accuracy: {:.2f}%'.format(rfc_accuracy*100))
	cm = confusion_matrix(y_test,rfc_pred)
	conf=sns.heatmap(cm,annot=True);
	figure = conf.get_figure()    
	figure.savefig('confrf.png', dpi=400,bbox_inches='tight',transparent=True)
	plt.clf()
    
	
	df2=pd.read_csv("heart.csv")
	chest_pain=pd.get_dummies(df2['cp'],prefix='cp',drop_first=True)
	df2=pd.concat([df2,chest_pain],axis=1)
	df2.drop(['cp'],axis=1,inplace=True)
	sp=pd.get_dummies(df2['slope'],prefix='slope')
	th=pd.get_dummies(df2['thal'],prefix='thal')
	rest_ecg=pd.get_dummies(df2['restecg'],prefix='restecg')
	frames=[df2,sp,th,rest_ecg]
	df2=pd.concat(frames,axis=1)
	df2.drop(['slope','thal','restecg'],axis=1,inplace=True)
	X = df2.drop(['target'], axis = 1)
	y = df2.target.values
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)


	import tensorflow as tf
	import keras
	from keras.models import Sequential
	from keras.layers import Conv2D, MaxPooling2D
	from keras.layers import Activation, Dropout, Flatten, Dense
	from keras.layers import Dense
	import warnings



	classifier = Sequential()

# Adding the input layer and the first hidden layer
	classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu', input_dim = 22))

# Adding the second hidden layer
	classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu'))
	
# Adding the output layer
	classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])




	classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


	y_pred = classifier.predict(X_test)


	ann_accuracy=accuracy_score(y_test, y_pred.round())
	ann_precision = precision_score(y_test, y_pred.round())
	ann_recall = recall_score(y_test,y_pred.round())
	ann_res=[]
	ann_res.append(ann_accuracy*100)
	ann_res.append(ann_precision*100)
	ann_res.append(ann_recall*100)

	cm = confusion_matrix(y_test, y_pred.round())
	conf=sns.heatmap(cm,annot=True);
	figure = conf.get_figure()
	figure.savefig('confann.png', dpi=400,bbox_inches='tight',transparent=True)
	plt.clf()
	
	
	data=[]
	data.append(lr_res) 
	data.append(rfc_res) 
	data.append(ann_res)  
	func = lambda x: round(x,2)
	data = [list(map(func, i)) for i in data]
	return render_template('mlran.html',data=data)

