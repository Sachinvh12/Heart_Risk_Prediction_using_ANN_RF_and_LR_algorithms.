def func():
	import numpy as np 
	import pandas as pd 
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.metrics import confusion_matrix
	from sklearn.model_selection import train_test_split


	df=pd.read_csv("heart.csv")

	df.columns = ['Age', 'Gender',	'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchieved',
       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']

	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier

	X_data = df.drop(columns=['Age','Target'], axis=1)
	Y = df['Target']

	Y = ((Y - np.min(Y))/ (np.max(Y) - np.min(Y))).values
	X = ((X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))).values

	x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)



	from sklearn.metrics import accuracy_score
	lr = LogisticRegression()
	lr.fit(x_train,y_train)
	lr_pred = lr.predict(x_test)


	lr_accuracy = accuracy_score(y_test, lr_pred)

	cm = confusion_matrix(y_test,lr_pred)
	conf=sns.heatmap(cm,annot=True);
	figure = conf.get_figure()    
	figure.savefig('conflr.png', dpi=400,bbox_inches='tight',transparent=True)
	plt.clf()
    


	rfc = RandomForestClassifier(n_estimators = 50, max_depth = 3)

	rfc.fit(x_train, y_train)
	rfc_pred = rfc.predict(x_test)
	rfc_accuracy = accuracy_score(y_test, rfc_pred)
 
	print('Random Forest Classifier Accuracy: {:.2f}%'.format(rfc_accuracy*100))
	cm = confusion_matrix(y_test,rfc_pred)
	conf=sns.heatmap(cm,annot=True);
	figure = conf.get_figure()    
	figure.savefig('confrf.png', dpi=400,bbox_inches='tight',transparent=True)
	plt.clf()
    
    

	return "Ran your models"
