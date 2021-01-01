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

def model_logreg(x_train, X_Test, Y_train, y_test, flag=False):
    """
    Fits the data with the best Logistic Regression model in terms of AUC (using GridSearchCV) and returns other scores.
    :param x_train: x_train.
    :param X_Test: X_Test.
    :param Y_train: Y_train.
    :param y_test: y_test.
    :param flag: If true, plot confusion matrix for the best estimator.
    :return logreg: Best estimator achieved by GridSearchCV
    :return LogReg_Scores: Scores of the best estimator (Se, Sp, PPV, NPV, Accuracy, F1, AUC)
    """
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    param_grid = {'logreg__C': [0.01, 0.1, 1, 10, 100],
                'logreg__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'logreg__penalty': ['l1', 'l2']}
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('logreg', LogisticRegression(max_iter=1000,
                                                         random_state=42))])
    sh = GridSearchCV(estimator=pipe, param_grid=param_grid,
                       scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                       cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
    sh.fit(x_train, Y_train)
    logreg = sh.best_estimator_

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    y_pred_test = logreg.predict(X_Test)
    TP = calc_TP(y_test,y_pred_test)
    TN = calc_TN(y_test,y_pred_test)
    FN = calc_FN(y_test,y_pred_test)
    FP = calc_FP(y_test,y_pred_test)

    logreg_Se = TN/(TN+FP)
    logreg_Sp = TP/(TP+FN)
    logreg_PPV = TP/(TP+FP)
    logreg_NPV = TN/(TN+FN)
    logreg_Accuracy = (TP+TN)/(TP+FP+TN+FN)
    logreg_F1 = 2*((logreg_PPV*logreg_Se)/(logreg_PPV+logreg_Se))
    logreg_AUC = roc_auc_score(y_test, logreg.predict_proba(X_Test)[:, 1])
    logreg_log_loss = log_loss(y_true=y_test, y_pred=y_pred_test)
    LogReg_Scores = (logreg_Se, logreg_Sp, logreg_PPV, logreg_NPV, logreg_Accuracy, logreg_F1, logreg_AUC, logreg_log_loss)
    if flag:
        plot_confusion_matrix(logreg, X_Test, y_test)
        plt.grid(False)
        plt.title('LogReg')
        plt.show()
    return logreg, LogReg_Scores

def model_lin_svm(x_train, X_Test, Y_train, y_test, flag=False):
    """
    Fits the data with the best linear SVM model in terms of AUC (using GridSearchCV)
    and returns other scores.
    :param x_train: x_train.
    :param X_Test: X_Test.
    :param Y_train: Y_train.
    :param y_test: y_test.
    :param flag: If true, plot confusion matrix for the best estimator.
    :return SVM: Best estimator achieved by GridSearchCV
    :return SVM_Scores: Scores of the best estimator (Se, Sp, PPV, NPV, Accuracy, F1, AUC)
    """
    skf = StratifiedKFold(n_splits=5, random_state=74, shuffle=True)
    param_grid = {'svc__C': [0.01, 0.1, 1, 10, 100],
                  'svc__kernel': ['linear']}
    pipe = Pipeline(steps=[('scale', StandardScaler()), ('svc', SVC(gamma='auto',
                                                                    probability=True,
                                                                    random_state=42))])
    sh = GridSearchCV(estimator=pipe, param_grid=param_grid,
                      scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                      cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
    sh.fit(x_train, Y_train)
    lin_SVM = sh.best_estimator_

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    y_pred_test = lin_SVM.predict(X_Test)
    TP = calc_TP(y_test, y_pred_test)
    TN = calc_TN(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)

    SVM_Se = TN / (TN + FP)
    SVM_Sp = TP / (TP + FN)
    SVM_PPV = TP / (TP + FP)
    SVM_NPV = TN / (TN + FN)
    SVM_Accuracy = (TP + TN) / (TP + FP + TN + FN)
    SVM_F1 = 2 * ((SVM_PPV * SVM_Se) / (SVM_PPV + SVM_Se))
    SVM_AUC = roc_auc_score(y_test, lin_SVM.predict_proba(X_Test)[:, 1])
    SVM_log_loss = log_loss(y_true=y_test, y_pred=y_pred_test)

    lin_SVM_scores = (SVM_Se, SVM_Sp, SVM_PPV, SVM_NPV, SVM_Accuracy, SVM_F1, SVM_AUC, SVM_log_loss)
    if flag:
        plot_confusion_matrix(lin_SVM, X_Test, y_test)
        plt.grid(False)
        plt.title('Linear SVC')
        plt.show()
    return lin_SVM, lin_SVM_scores
def model_nonlin_svm(x_train, X_Test, Y_train, y_test, flag=False):
    """
    Fits the data with the best non-linear SVM model in terms of AUC(using GridSearchCV)
    and returns other scores.
    :param x_train: x_train.
    :param X_Test: X_Test.
    :param Y_train: Y_train.
    :param y_test: y_test.
    :param flag: If true, plot confusion matrix for the best estimator.
    :return SVM: Best estimator achieved by GridSearchCV
    :return SVM_Scores: Scores of the best estimator (Se, Sp, PPV, NPV, Accuracy, F1, AUC)
    """
    skf = StratifiedKFold(n_splits=5, random_state=74, shuffle=True)
    param_grid = {'svc__C': [0.01, 0.1, 1, 10, 100],
                  'svc__kernel': ['poly', 'rbf', 'sigmoid']}
    pipe = Pipeline(steps=[('scale', StandardScaler()), ('svc', SVC(gamma='auto',
                                                                    probability=True,
                                                                    random_state=42))])
    sh = GridSearchCV(estimator=pipe, param_grid=param_grid,
                      scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                      cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
    sh.fit(x_train, Y_train)
    nonlin_SVM = sh.best_estimator_

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    y_pred_test = nonlin_SVM.predict(X_Test)
    TP = calc_TP(y_test, y_pred_test)
    TN = calc_TN(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)

    SVM_Se = TN / (TN + FP)
    SVM_Sp = TP / (TP + FN)
    SVM_PPV = TP / (TP + FP)
    SVM_NPV = TN / (TN + FN)
    SVM_Accuracy = (TP + TN) / (TP + FP + TN + FN)
    SVM_F1 = 2 * ((SVM_PPV * SVM_Se) / (SVM_PPV + SVM_Se))
    SVM_AUC = roc_auc_score(y_test, nonlin_SVM.predict_proba(X_Test)[:, 1])
    SVM_log_loss = log_loss(y_true=y_test, y_pred=y_pred_test)
    nonlin_SVM_scores = (SVM_Se, SVM_Sp, SVM_PPV, SVM_NPV, SVM_Accuracy, SVM_F1, SVM_AUC, SVM_log_loss)
    if flag:
        plot_confusion_matrix(nonlin_SVM, X_Test, y_test)
        plt.grid(False)
        plt.title('Non-Linear SVC')
        plt.show()
    return nonlin_SVM, nonlin_SVM_scores


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
    LDA_log_loss = log_loss(y_true=y_test, y_pred=y_pred_test)

    LDA_Scores = (LDA_Se, LDA_Sp, LDA_PPV, LDA_NPV, LDA_Accuracy, LDA_F1, LDA_AUC, LDA_log_loss)
    if flag:
        plot_confusion_matrix(LDA, X_Test, y_test)
        plt.grid(False)
        plt.title('LDA')
        plt.show()
    return LDA, LDA_Scores

def model_RFC(x_train, X_Test, Y_train, y_test, flag=False):
    """
    Fits the data with the best Random Forest Classifier model in terms of AUC (using GridSearchCV) and returns other scores.
    :param x_train: x_train.
    :param X_Test: X_Test.
    :param Y_train: Y_train.
    :param y_test: y_test.
    :param flag: If true, plot confusion matrix for the best estimator.
    :return RFC: Best estimator achieved by GridSearchCV
    :return RFC_Scores: Scores of the best estimator (Se, Sp, PPV, NPV, Accuracy, F1, AUC)
    """
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    param_grid = {'RF__max_features': ['auto', 'sqrt', 'log2'],
                  'RF__criterion': ['gini', 'entropy'],
                  'RF__class_weight': ['balanced', 'balanced_subsample']}
    pipe = Pipeline(steps=[('scale', StandardScaler()),
                           ('RF', RandomForestClassifier(random_state=42))])
    sh = GridSearchCV(estimator=pipe, param_grid=param_grid,
                      scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                      cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
    sh.fit(x_train, Y_train)
    RF = sh.best_estimator_

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    y_pred_test = RF.predict(X_Test)
    TP = calc_TP(y_test, y_pred_test)
    TN = calc_TN(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)

    RF_Se = TN / (TN + FP)
    RF_Sp = TP / (TP + FN)
    RF_PPV = TP / (TP + FP)
    RF_NPV = TN / (TN + FN)
    RF_Accuracy = (TP + TN) / (TP + FP + TN + FN)
    RF_F1 = 2 * ((RF_PPV * RF_Se) / (RF_PPV + RF_Se))
    RF_AUC = roc_auc_score(y_test, RF.predict_proba(X_Test)[:, 1])
    RF_log_loss = log_loss(y_true=y_test, y_pred=y_pred_test)

    RF_Scores = (RF_Se, RF_Sp, RF_PPV, RF_NPV, RF_Accuracy, RF_F1, RF_AUC, RF_log_loss)
    if flag:
        plot_confusion_matrix(RF, X_Test, y_test)
        plt.grid(False)
        plt.title('RFC')
        plt.show()
    return RF, RF_Scores

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
    KNN_log_loss = log_loss(y_true=y_test, y_pred=y_pred_test)

    KNN_Scores = (KNN_Se, KNN_Sp, KNN_PPV, KNN_NPV, KNN_Accuracy, KNN_F1, KNN_AUC, KNN_log_loss)
    if flag:
        plot_confusion_matrix(KNN, X_Test, y_test)
        plt.grid(False)
        plt.title('KNN')
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
    DT_log_loss = log_loss(y_true=y_test, y_pred=y_pred_test)

    DT_Scores = (DT_Se, DT_Sp, DT_PPV, DT_NPV, DT_Accuracy, DT_F1, DT_AUC, DT_log_loss)
    if flag:
        plot_confusion_matrix(DT, X_Test, y_test)
        plt.grid(False)
        plt.title('DT')
        plt.show()

    return DT, DT_Scores

def Model_Comparison_Table(Table_Data, Title):
    """
    Compares the scores of the best estimators from all the models we tried out.
    :param Table_Data: Table of the scores from the classifiers we want to compare.
    :param Title: Title of the table.
    :return: Comparison Table.
    """
    Table_Data = np.array(Table_Data)
    Table_Data = Table_Data.round(decimals=5)
    a = Table_Data.shape[1]
    fig, ax = plt.subplots(1, 1)
    table = ax.table(cellText=Table_Data,
                     rowLabels=['LogReg', 'Linear SVC', 'Non-Linear SVC',
                                'LDA', 'DT', 'RFC', 'KNN'],
                     colLabels=['Se', 'Sp', 'PPV', 'NPV', 'Accuracy', 'F1', 'AUC', 'Log Loss'],
                     colColours=["salmon"] * 8, rowColours=["skyblue"] * a,
                     cellLoc='center', loc='upper center')
    ax.set_title(Title,
                 fontweight="bold")
    ax.axis('tight')
    ax.axis('off')
    plt.show()

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

    fig, ax = plt.subplots(figsize=(5, 4))
    feat_plot = sns.barplot(x=sorted_names, y=imp_vals, ax=ax)
    for item in feat_plot.get_xticklabels():
        item.set_rotation(70)
    feat_plot.set_xticklabels(feat_plot.get_xticklabels(), size=5)
    plt.rcParams['xtick.labelsize'] = 9
    plt.title('Features Importance - RFC')
    plt.show()