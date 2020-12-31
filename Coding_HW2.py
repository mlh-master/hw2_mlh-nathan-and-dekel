import  numpy as np
import pandas as pd
import numpy.random as rd
import matplotlib as mpl
import seaborn as sns #for the countplots!
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#for the evaluation metrics:
from sklearn.metrics import log_loss
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import hinge_loss





A=pd.read_csv('HW2_data.csv')
#We binary encode the data, as it will be more convenient to handle.
A.replace(['Yes','No'],[1,0],inplace=True)
A.replace(['Positive','Negative'],[1,0],inplace=True)
A.replace(['Male','Female'],[1,0],inplace=True)



#Are they missing values? In which column?
dicNa={}
for column in A.columns:
    dicNa[column]=A[column].isna().sum()

#solution 1: remove the rows
A.dropna(inplace=True)
A.reset_index(drop=True,inplace=True)


#solution 2: inputation with Bernoulli random variable
# for column in A.columns:
#     if dicNa[column]!=0:
#         p=(A[column].sum())/(A.shape[0]-dicNa[column])
#         A[column].apply(lambda x: rd.binomial(1,p,1) if (np.isnan(x)) else x)

#Train-Test Split: is the data imbalanced?
B= pd.concat([A.iloc[:,:16],A.iloc[:, 17]],axis=1)
#Xwa=pd.DataFrame(enc.fit_transform(B.loc[:,'Gender':'Family History']).toarray())
#Xwa.columns=enc.get_feature_names(B.columns[1:])
#
# X=pd.DataFrame(A.iloc[:,0]).join(Xwa)
X=B
Y = A.iloc[:, 16]

Y.value_counts().plot(kind="pie", labels=['Positive','Negative'], colors = ['steelblue', 'salmon'], autopct='%1.1f%%')
plt.show()
#Yes! So we have to stratify.

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 10, stratify=Y)
X_train.reset_index(drop=True,inplace=True)
x_test.reset_index(drop=True,inplace=True)

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder(sparse=False)
enc.fit(X_train.iloc[:,1:])
X_train_enc=pd.DataFrame(X_train.iloc[:,0]).join(pd.DataFrame(enc.transform(X_train.iloc[:,1:]),columns=enc.get_feature_names(X.columns[1:])))
x_test_enc=pd.DataFrame(x_test.iloc[:,0]).join(pd.DataFrame(enc.transform(x_test.iloc[:,1:]),columns=enc.get_feature_names(X.columns[1:])))


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

n_splits = 4
max_iter = 2000
skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
#Before chosing our models, let's scale our data!
scaler = StandardScaler()
x_tr = scaler.fit_transform(X_train_enc)
x_tst = scaler.transform(x_test_enc)

#------------LOGISTIC REGRESSION-----------------
from sklearn.linear_model import LogisticRegression

#Tuning our Hyperparameter lambda and the regularization scheme:
solver = 'liblinear'
log_reg = LogisticRegression(random_state=5, max_iter=max_iter,solver=solver)
lmbda = np.array([0.01, 0.01, 1, 10, 100, 1000])
pipe = Pipeline(steps=[('scale', StandardScaler()), ('logistic', log_reg)])
clf = GridSearchCV(estimator=pipe, param_grid={'logistic__C': 1/lmbda, 'logistic__penalty': ['l1','l2']},
                   scoring=['accuracy','f1','precision','recall','roc_auc'], cv=skf,
                   refit='roc_auc', verbose=3, return_train_score=True)
clf.fit(X_train_enc, Y_train)
#let's take the best estimator according to our AUC metric:
best_log_reg=clf.best_estimator_
#we also keep track of the best params to use them with PCA at section 7:
best_lambda,best_pen=clf.best_params_['logistic__C'],clf.best_params_['logistic__penalty']


#After we chose our hyperparameter lambda, we can test our model performances.
#We'll report the performance statistics on both train and test sets (AUC, F1, LOSS, ACC) as asked.
def predict_and_report(model,estimator,X,y):
    #model='LR' or 'SVM'
    #X=X_train_enc or x_test_enc
    #y=Y_train or y_test
    y_pred= estimator.predict(X)
    y_pred_proba= estimator.predict_proba(X)
    TN = confusion_matrix(y, y_pred)[0, 0]
    FP = confusion_matrix(y, y_pred)[0, 1]
    FN = confusion_matrix(y, y_pred)[1, 0]
    TP = confusion_matrix(y, y_pred)[1, 1]
    ACC = (TP + TN) / (TP + TN + FP + FN)
    PPV = TP / (TP + FP)
    Se = TP / (TP + FN)
    F1 = 2 * Se * PPV / (Se + PPV)
    if model=='LR':
        print("Loss is {:.2f}".format(log_loss(y, y_pred_proba)))
    if model=='SVM':
        print("Loss is {:.2f}".format(hinge_loss(y,estimator.decision_function(X))))
    plot_confusion_matrix(estimator,X,y, cmap=plt.cm.Blues)
    plt.show()
    print('Accuracy is {:.2f}'.format(ACC))
    print('F1 is {:.2f}'.format(F1))
    print('AUROC is {:.2f}'.format(roc_auc_score(y, y_pred_proba[:, 1])))


# #Fit
#log_reg = LogisticRegression(random_state=5, penalty=pen, C = 1/lbda, max_iter=max_iter,solver=solver)
# log_reg.fit(x_tr, Y_train)
# #Predict and Report
# #Train & Test losses
# y_pred_proba_train = log_reg.predict_proba(x_tr)
# y_pred_proba_test = log_reg.predict_proba(x_tst)
# print("Train loss is {:.2f}".format(log_loss(Y_train,y_pred_proba_train)))
# print("Test loss is {:.2f}".format(log_loss(y_test,y_pred_proba_test)))
# y_pred_train = log_reg.predict(x_tr)
# y_pred_test = log_reg.predict(x_tst)
#
# #Plot of the Confusion Matrix
# plot_confusion_matrix(log_reg, x_tst, y_test, cmap=plt.cm.Blues)
# plt.show()
#
# #ACC and F1
# def ACC_F1(y_true,y_pred):
#     TN = confusion_matrix(y_true, y_pred)[0, 0]
#     FP = confusion_matrix(y_true, y_pred)[0, 1]
#     FN = confusion_matrix(y_true, y_pred)[1, 0]
#     TP = confusion_matrix(y_true, y_pred)[1, 1]
#     ACC=(TP+TN)/(TP+TN+FP+FN)
#     PPV=TP/(TP+FP)
#     Se=TP/(TP+FN)
#     F1=2*Se*PPV/(Se+PPV)
#     print('Accuracy is {:.2f}'.format(ACC))
#     print('F1 is {:.2f}'.format(F1))
# ACC_F1(Y_train,y_pred_train)
# ACC_F1(y_test,y_pred_test)
#
# #AUC
# print('AUROC is {:.2f}'.format(roc_auc_score(Y_train, y_pred_proba_train[:,1])))
# print('AUROC is {:.2f}'.format(roc_auc_score(y_test, y_pred_proba_test[:,1])))

predict_and_report(model='LR',estimator=best_log_reg,X=X_train_enc,y=Y_train)
predict_and_report(model='LR',estimator=best_log_reg,X=x_test_enc,y=y_test)

#-------------LINEAR SVM--------------
from sklearn.svm import SVC

svc = SVC(probability=True)
C = np.array([0.001, 0.01, 1, 10, 100, 1000])
solver='liblinear'
pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
svm_lin = GridSearchCV(estimator=pipe,
             param_grid={'svm__kernel':['linear'], 'svm__C':C},
             scoring=['accuracy','f1','precision','recall','roc_auc'],
             cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_lin.fit(X_train_enc, Y_train)
best_lin_svm=svm_lin.best_estimator_
best_C_l=svm_lin.best_params_['svm__C']

predict_and_report(model='SVM',estimator=best_lin_svm,X=X_train_enc,y=Y_train)
predict_and_report(model='SVM',estimator=best_lin_svm,X=x_test_enc,y=y_test)

#--------------NON-LINEAR SVM-----------------------------------------

svc = SVC(probability=True)
C = np.array([0.01,1, 10, 100])
pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
svm_nonlin = GridSearchCV(estimator=pipe,param_grid={'svm__kernel':['rbf','poly'], 'svm__C':C, 'svm__degree':[3],'svm__gamma':['auto','scale']},
             scoring=['accuracy','f1','precision','recall','roc_auc'], cv=skf, refit='roc_auc', verbose=3, return_train_score=True)
svm_nonlin.fit(X_train_enc, Y_train)
best_C_nl,best_gamma,best_kernel=svm_nonlin.best_params_['svm__C'],svm_nonlin.best_params_['svm__gamma'],svm_nonlin.best_params_['svm__kernel']

best_nlin_svm=svm_nonlin.best_estimator_
predict_and_report(model='SVM',estimator=best_nlin_svm,X=X_train_enc,y=Y_train)
predict_and_report(model='SVM',estimator=best_nlin_svm,X=x_test_enc,y=y_test)


#----------------RANDOM FOREST NETWORK FOR FEATURE SELECTION--------------------------------

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=4, random_state=0)
rfc.fit(x_tr, Y_train)
importance = rfc.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()



#------------DIMENSIONALITY REDUCTION---------------------------------

from sklearn.decomposition import PCA

#Perform PCA
n_components = 2
pca=PCA(n_components=n_components,whiten=True)
x_tr_pca=pca.fit_transform(x_tr)
x_tst_pca=pca.transform(x_tst)

#Plot data in 2D
def plt_2d_pca(X_pca,y):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='b')
    ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='r')
    ax.legend(('Negative','Positive'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('2D PCA')
    plt.show()
plt_2d_pca(x_tst_pca,y_test)

#Training of the models on the reduced set (with the same hyperparameters we tuned before)

#Logistic Regression:
lr = LogisticRegression(random_state=5, penalty=best_pen, C = 1/best_lambda, max_iter=max_iter,solver=solver)
pipe_pca_lr = Pipeline(steps=[('scale', StandardScaler()),('pca', pca), ('logistic', lr)])
pipe_pca_lr.fit(X_train_enc, Y_train)
predict_and_report(model='LR',estimator=pipe_pca_lr,X=X_train_enc,y=Y_train)
predict_and_report(model='LR',estimator=pipe_pca_lr,X=x_test_enc,y=y_test)

#Linear SVM:
lsvm=SVC(C=best_C_l,kernel='linear',probability=True)
pipe_pca_lsvm = Pipeline(steps=[('scale', StandardScaler()),('pca', pca), ('svm', lsvm)])
pipe_pca_lsvm.fit(X_train_enc, Y_train)
predict_and_report(model='SVM',estimator=pipe_pca_lsvm,X=X_train_enc,y=Y_train)
predict_and_report(model='SVM',estimator=pipe_pca_lsvm,X=x_test_enc,y=y_test)

#Non-linear SVM:
nlsvm=SVC(C=best_C_nl,kernel=best_kernel,gamma=best_gamma,probability=True)
pipe_pca_nlsvm = Pipeline(steps=[('scale', StandardScaler()),('pca', pca), ('svm', nlsvm)])
pipe_pca_nlsvm.fit(X_train_enc, Y_train)
predict_and_report(model='SVM',estimator=pipe_pca_nlsvm,X=X_train_enc,y=Y_train)
predict_and_report(model='SVM',estimator=pipe_pca_nlsvm,X=x_test_enc,y=y_test)








