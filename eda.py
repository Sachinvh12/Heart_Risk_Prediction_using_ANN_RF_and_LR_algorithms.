def funcy():
    from flask import Flask,render_template
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")

    


    bg_color = (0.25, 0.25, 0.25)
    sns.set(rc={"font.style":"normal",
                "axes.facecolor":bg_color,
                "figure.facecolor":bg_color,
                "text.color":"white",
                "xtick.color":"white",
                "ytick.color":"white",
                "axes.labelcolor":"white",
                "axes.grid":False,
                'axes.labelsize':25,
                'figure.figsize':(10.0,5.0),
                'xtick.labelsize':15,
                'ytick.labelsize':10})



    df=pd.read_csv("heart.csv")
    df2=pd.read_csv("heart.csv")



    df.head()



    df.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchieved',
           'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']



    df.head()

    df.tail()



    df.shape



    df.isnull().sum()


    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))

    coerr=sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,mask=mask,cmap='summer_r',vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    figure = coerr.get_figure()    
    figure.savefig('static/img/coerr.png', dpi=400,bbox_inches='tight',transparent=True)
    plt.clf()
    

    typical_angina_cp = [k for k in df['ChestPain'] if k ==0]
    atypical_angina_cp = [k for k in df['ChestPain'] if k ==1]
    non_anginal_cp = [k for k in df['ChestPain'] if k ==2]
    none_cp = [k for k in df['ChestPain'] if k ==3]

    typical_angina_cp_total = len(typical_angina_cp)*100/len(df)
    atypical_angina_cp_total = len(atypical_angina_cp)*100/len(df)
    non_anginal_cp_total = len(non_anginal_cp)*100/len(df)
    none_cp_total = len(none_cp)*100/len(df)

    labels=['Typical angina','Atypical angina','Non-anginal','Asymptomatic']
    values = [typical_angina_cp_total,atypical_angina_cp_total,non_anginal_cp_total,none_cp_total]

    plt.pie(values,labels=labels,autopct='%1.2f%%')

    plt.title("Chest Pain Type Percentage")    
    plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)

    plt.savefig("static/img/cptp.png",bbox_inches='tight',transparent=True)
    plt.clf()


    target=[];
    for k in df['Target']:
        if k > 0:
            target.append(1)
        else:
            target.append(0)

    ax = sns.countplot(x=target,palette='bwr')

    plt.title("People who have heart disease")
    plt.ylabel("")
    plt.yticks([])
    plt.xlabel("")

    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))
    ax.set_xticklabels(["Healthy Heart","Heart Disease"]);


    plt.savefig("static/img/hhcp.png",bbox_inches='tight',transparent=True)
    plt.clf()


    heart_health=[]
    for k in df['Target']:
        if k == 0:
            heart_health.append('Healthy Heart')
        elif k == 1:
            heart_health.append('Heart Disease')
    plt.title("Heart-Health Vs Chest Pain Type")
    ax = sns.countplot(x='ChestPain',hue=heart_health,data=df,palette='bwr') 
    plt.ylabel("")
    plt.yticks([])
    plt.xlabel("")
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+0.5))
        ax.set_xticklabels(['Typical Angina','Atypical Angina','Non-Anginal','Asymptomatic']);
    plt.savefig("static/img/pwhhd.png",bbox_inches='tight',transparent=True)
    plt.clf()


    bp=[]
    for k in df['RestingBloodPressure']:
        if (k > 130):
            bp.append(1) 
        else:
            bp.append(0) 

    ax = sns.countplot(x=bp,palette='bwr')

    plt.title("Resting Blood Pressure Count")
    plt.ylabel("")
    plt.yticks([])
    plt.xlabel("")

    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.35, p.get_height()+0.5))
    
    ax.set_xticklabels(["Normal BP","Abnormal BP"]);
    plt.savefig("static/img/rbpc.png",bbox_inches='tight',transparent=True)
    plt.clf()






    age_group=[]
    for k in df['Age']:
        if (k >=29) & (k<40):
            age_group.append(0)
        elif (k >=40)&(k<55):
            age_group.append(1)
        else:
            age_group.append(2)
    df['Age-Group'] = age_group
    plt.title("Heart-Health Vs Age group")
    ax = sns.countplot(x=age_group,hue=heart_health,palette='bwr')

    plt.ylabel("")
    plt.yticks([])
    plt.xlabel("")
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.15, p.get_height()+0.5))
    
    ax.set_xticklabels(['Young (29-40)','Mid-Age(40-55)','Old-Age(>55)']);
    plt.savefig("static/img/hhag.png",bbox_inches='tight',transparent=True)
    plt.clf()
    

    return render_template('edaran.html')

