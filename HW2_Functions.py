from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#for the evaluation metrics:
from sklearn.metrics import log_loss
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import hinge_loss


#-----------Compare Train-Test distributions----------------------

def compare_distrib(X_train,X_test):
    PTrain, PTest = [], []
    for coltrain in X_train.columns[1:]:
        PTrain.append(100 * X_train[coltrain].sum() / (X_train.shape[0]))
    for coltest in X_test.columns[1:]:
        PTest.append(100 * X_test[coltest].sum() / (X_test.shape[0]))
    Delta = [abs(PTrain[i] - PTest[i]) for i in range(X_train.shape[1] - 1)]
    percentages = np.round(([PTrain, PTest, Delta]), 1).T
    tab = np.concatenate((np.transpose([X_train.columns[1:]]), percentages), axis=1)
    distrib = pd.DataFrame(data=tab, columns=['Positive Feature', 'Train %', 'Test %', 'Delta %'])
    return distrib

#-----------Make predictions & report the performances: -----------

def predict_and_report(estimator,X,y,set=None,model=None):
    y_pred= estimator.predict(X)
    y_pred_proba= estimator.predict_proba(X)
    TN = confusion_matrix(y, y_pred)[0, 0]
    FP = confusion_matrix(y, y_pred)[0, 1]
    FN = confusion_matrix(y, y_pred)[1, 0]
    TP = confusion_matrix(y, y_pred)[1, 1]
    ACC = (TP + TN) / (TP + TN + FP + FN)
    PPV = TP / (TP + FP)
    NPV=TN / (TN + FN)
    Se = TP / (TP + FN)
    SP=TP / (TP + FN)
    F1 = 2 * Se * PPV / (Se + PPV)
    if model=='LR':
        #print("Loss is {:.2f}".format(log_loss(y, y_pred_proba)))
        LOSS=log_loss(y, y_pred_proba)
    if model=='SVM':
        #print("Loss is {:.2f}".format(hinge_loss(y,estimator.decision_function(X))))
        LOSS=hinge_loss(y,estimator.decision_function(X))
    plot_confusion_matrix(estimator,X,y, cmap=plt.cm.Blues)
    plt.title(set)
    plt.show()
    Scores=[Se,SP,PPV,NPV,ACC,F1,roc_auc_score(y, y_pred_proba[:, 1])]
    return Scores


#-------------Summarize the performances within a table: ----------------


def Model_Comparison_Table(Table_Data, Title):
    """
    Compares the scores of the best estimators from all the models we tried out.
    :param Table_Data: Table of the scores from the classifiers we want to compare.
    :param Title: Title of the table.
    :return: Comparison Table.
    """

    a = Table_Data.shape[1]
    #plt.figure(figsize=(10, 14))
    fig, ax = plt.subplots(1, 1)
    table = ax.table(cellText=Table_Data,
                     rowLabels=['LogReg', 'Linear SVC', 'Non-Linear SVC','RFC'],
                     colLabels=['Se','Sp','PPV','NPV','Accuracy', 'F1', 'AUC'],
                     colColours=["salmon"] * 7, rowColours=["skyblue"] * a,
                     cellLoc='center', loc='upper center')
    ax.set_title(Title, fontweight="bold")
    ax.axis('tight')
    ax.axis('off')
    plt.show()


#-----------------RFC---------------------

def RFC_Feature_Selection(feat, RF):
    """
    Using a sorting algorithm, shows a Bar Plot of the features sorted by importance.
    :param feat: Names of the features in the data.
    :param RF: RFC fitted to data.
    :return: Bar Plot.
    """
    imp_feat = sorted(zip(map(lambda a: round(a, 4), RF.feature_importances_), feat))
    imp_feat.reverse()
    sorted_names = [a[1] for a in imp_feat]
    imp_vals = [a[0] for a in imp_feat]

    fig, ax = plt.subplots(figsize=(5,4))
    feat_plot = sns.barplot(x=sorted_names, y=imp_vals, ax=ax)
    for item in feat_plot.get_xticklabels():
        item.set_rotation(70)
    feat_plot.set_xticklabels(feat_plot.get_xticklabels(), size=5)
    plt.rcParams['xtick.labelsize'] = 13
    plt.title('Features Importance - RFC')
    plt.show()


#-----------------PLOT 2D PCA-----------------------------------


def plot_2d_pca(X_pca,y):
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


