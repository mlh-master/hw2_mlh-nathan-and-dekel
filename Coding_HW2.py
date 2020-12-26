import  numpy as np
import pandas as pd
import numpy.random as rd
import matplotlib as mpl
import seaborn as sns #for the countplots!
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

A=pd.read_csv('HW2_data.csv')
#We already encode the data, as it will be more convenient to handle.
A.replace(['Yes','No'],[1,0],inplace=True)
A.replace(['Positive','Negative'],[1,0],inplace=True)
A.replace(['Male','Female'],[1,0],inplace=True)


#Are they missing values? In which column?
dicNa={}
for column in A.columns:
    dicNa[column]=A[column].isna().sum()

#solution 1: remove the rows
#A.dropna(inplace=True)

#solution 2: inputation with Bernoulli random variable
for column in A.columns:
    if dicNa[column]!=0:
        p=(A[column].sum())/(A.shape[0]-dicNa[column])
        A[column].apply(lambda x: rd.binomial(1,p,1) if (np.isnan(x)) else x)

#Train-Test Split: is the data imbalanced?
X = pd.concat([A.iloc[:,:16],A.iloc[:, 17]],axis=1)
Y = A.iloc[:, 16]

Y.value_counts().plot(kind="pie", labels=['Positive','Negative'], colors = ['steelblue', 'salmon'], autopct='%1.1f%%')
plt.show()
#Yes! So we have to stratify.

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 10, stratify=Y)

#Exploration of the Data:
#Let's show that the distribution of the features is similar between test and train.
#First, we can plot the histograms for Age:

bins = 20
feat = 'Age'
plt.figure()
plt.hist(X_train[feat], bins, density=True, alpha=0.5, label='Train')
plt.hist(x_test[feat], bins, density=True, alpha=0.5, label='Test')
plt.xlabel(feat)
plt.ylabel('Probability')
plt.legend(loc='upper right')
plt.show()

#Then, we'll build a table that summarizes the distribution of the binary variables:
PTrain,PTest=[],[]
for coltrain in X_train.columns[1:]:
    PTrain.append(100*X_train[coltrain].sum()/(X_train.shape[0]))
for coltest in x_test.columns[1:]:
    PTest.append(100*x_test[coltest].sum()/(x_test.shape[0]))
Delta=[PTrain[i]-PTest[i] for i in range(X_train.shape[1]-1)]
percentages=np.round(([PTrain,PTest,Delta]),1).T
tab=np.concatenate((np.transpose([X_train.columns[1:]]),percentages),axis=1)
distrib=pd.DataFrame(data=tab, columns=['Positive Feature','Train %','Test %','Delta %'])

#Visualisation of the Data:


#COUNTPLOT!


