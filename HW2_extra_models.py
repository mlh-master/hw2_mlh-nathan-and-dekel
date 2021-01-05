from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def model_LDA(x_train, X_Test, Y_train, y_test, flag=False):
    """
    Fits the data with the best LDA model in terms of AUC (using GridSearchCV) and returns other scores.
    :param x_train: x_train.
    :param X_Test: X_Test.
    :param Y_train: Y_train.
    :param y_test: y_test.
    :param flag: If true, plot confusion matrix for the best estimator.
    :return LDA: Best estimator achieved by GridSearchCV
    :return LDA_Scores: Scores of the best estimator (Se, Sp, PPV, NPV, Accuracy, F1, AUC)
    """
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    param_grid = {'LDA__solver': ['svd', 'lsqr', 'eigen']}
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('LDA', LinearDiscriminantAnalysis(store_covariance='True'))])
    sh = GridSearchCV(estimator=pipe, param_grid=param_grid,
                      scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                      cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
    sh.fit(x_train, Y_train)
    LDA = sh.best_estimator_

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    y_pred_test = LDA.predict(X_Test)
    TP = calc_TP(y_test, y_pred_test)
    TN = calc_TN(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)

    LDA_Se = TN / (TN + FP)
    LDA_Sp = TP / (TP + FN)
    LDA_PPV = TP / (TP + FP)
    LDA_NPV = TN / (TN + FN)
    LDA_Accuracy = (TP + TN) / (TP + FP + TN + FN)
    LDA_F1 = 2 * ((LDA_PPV * LDA_Se) / (LDA_PPV + LDA_Se))
    LDA_AUC = roc_auc_score(y_test, LDA.predict_proba(X_Test)[:, 1])


    LDA_Scores = (LDA_Se, LDA_Sp, LDA_PPV, LDA_NPV, LDA_Accuracy, LDA_F1, LDA_AUC)
    if flag:
        plot_confusion_matrix(LDA, X_Test, y_test)
        plt.grid(False)
        plt.title('LDA-Test Set')
        plt.show()
    return LDA, LDA_Scores





def model_KNN(x_train, X_Test, Y_train, y_test, flag=False):
    """
    Fits the data with the best KNN model in terms of AUC (using GridSearchCV) and returns other scores.
    :param x_train: x_train.
    :param X_Test: X_Test.
    :param Y_train: Y_train.
    :param y_test: y_test.
    :param flag: If true, plot confusion matrix for the best estimator.
    :return KNN: Best estimator achieved by GridSearchCV
    :return KNN_Scores: Scores of the best estimator (Se, Sp, PPV, NPV, Accuracy, F1, AUC)
    """
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    param_grid = {'KNN__n_neighbors': [1, 2, 3, 4, 5],
                  'KNN__weights': ['uniform', 'distance'],
                  'KNN__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'KNN__metric': ['euclidian', 'manhattan', 'chebyshev']}
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('KNN', KNeighborsClassifier())])
    sh = GridSearchCV(estimator=pipe, param_grid=param_grid,
                      scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                      cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
    sh.fit(x_train, Y_train)
    KNN = sh.best_estimator_

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    y_pred_test = KNN.predict(X_Test)
    TP = calc_TP(y_test, y_pred_test)
    TN = calc_TN(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)

    KNN_Se = TN / (TN + FP)
    KNN_Sp = TP / (TP + FN)
    KNN_PPV = TP / (TP + FP)
    KNN_NPV = TN / (TN + FN)
    KNN_Accuracy = (TP + TN) / (TP + FP + TN + FN)
    KNN_F1 = 2 * ((KNN_PPV * KNN_Se) / (KNN_PPV + KNN_Se))
    KNN_AUC = roc_auc_score(y_test, KNN.predict_proba(X_Test)[:, 1])

    KNN_Scores = (KNN_Se, KNN_Sp, KNN_PPV, KNN_NPV, KNN_Accuracy, KNN_F1, KNN_AUC)
    if flag:
        plot_confusion_matrix(KNN, X_Test, y_test)
        plt.grid(False)
        plt.title('KNN-Test Set')
        plt.show()

    return KNN, KNN_Scores

def model_DT(x_train, X_Test, Y_train, y_test, flag=False):
    """
    Fits the data with the best Decision Tree model in terms of AUC (using GridSearchCV) and returns other scores.
    :param x_train: x_train.
    :param X_Test: X_Test.
    :param Y_train: Y_train.
    :param y_test: y_test.
    :param flag: If true, plot confusion matrix for the best estimator.
    :return DT: Best estimator achieved by GridSearchCV
    :return DT_Scores: Scores of the best estimator (Se, Sp, PPV, NPV, Accuracy, F1, AUC)
    """
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    param_grid = {'DT__max_features': ['auto', 'sqrt', 'log2'],
                  'DT__criterion': ['gini', 'entropy'],
                  'DT__splitter': ['best', 'random']}
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('DT', DecisionTreeClassifier(random_state=42))])
    sh = GridSearchCV(estimator=pipe, param_grid=param_grid,
                      scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                      cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
    sh.fit(x_train, Y_train)
    DT = sh.best_estimator_

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    y_pred_test = DT.predict(X_Test)
    TP = calc_TP(y_test, y_pred_test)
    TN = calc_TN(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)

    DT_Se = TN / (TN + FP)
    DT_Sp = TP / (TP + FN)
    DT_PPV = TP / (TP + FP)
    DT_NPV = TN / (TN + FN)
    DT_Accuracy = (TP + TN) / (TP + FP + TN + FN)
    DT_F1 = 2 * ((DT_PPV * DT_Se) / (DT_PPV + DT_Se))
    DT_AUC = roc_auc_score(y_test, DT.predict_proba(X_Test)[:, 1])

    DT_Scores = (DT_Se, DT_Sp, DT_PPV, DT_NPV, DT_Accuracy, DT_F1, DT_AUC)
    if flag:
        plot_confusion_matrix(DT, X_Test, y_test)
        plt.grid(False)
        plt.title('DT-Test Set')
        plt.show()

    return DT, DT_Scores