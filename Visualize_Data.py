import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def corr_mat(df):
    """
    Shows the correlation matrix of the data.
    :param df: The Data.
    :return: Correlation Matrix.
    """
    corr = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True)
    plt.show()

def countplots(data, feat1, feat2):
    """
    Creates a count plot between 2 features (or 1 feature and the diagnosis).
    :param data: The Data.
    :param feat1: Feature #1.
    :param feat2: Feature #2.
    :return: Count Plot.
    """
    sns.countplot(x=feat1, hue=feat2, data=data)
    plt.show()

def test_train_dist_check(X_Test, x_train, feat1, feat2, feat3):
    """
    Creates a table that shows 3 different feature distributions in the Train & Test sets
    to see if the distributions of the features in the Train/Test sets are similar.
    :param X_Test: X_Test.
    :param x_train: x_train.
    :param feat1: Feature #1.
    :param feat2: Feature #2.
    :param feat3: Feature #3.
    :return: Distribution check table.
    """
    N = len(X_Test)
    M = len(x_train)

    Pct_test_1 = 100 * len(X_Test[X_Test[feat1] == 1])/N
    Pct_test_2 = 100 * len(X_Test[X_Test[feat2] == 1])/N
    Pct_test_3 = 100 * len(X_Test[X_Test[feat3] == 1])/N
    Pct_train_1 = 100 * len(x_train[x_train[feat1] == 1])/M
    Pct_train_2 = 100 * len(x_train[x_train[feat2] == 1])/M
    Pct_train_3 = 100 * len(x_train[x_train[feat3] == 1])/M

    Table_Data = [[Pct_test_1, Pct_train_1, abs(Pct_test_1 - Pct_train_1)],
                  [Pct_test_2, Pct_train_2, abs(Pct_test_2 - Pct_train_2)],
                  [Pct_test_3, Pct_train_3, abs(Pct_test_3 - Pct_train_3)]]
    Table_Data = np.array(Table_Data)
    Table_Data = Table_Data.round(decimals=2)

    fig, ax = plt.subplots(1, 1)
    table = ax.table(cellText=Table_Data,rowLabels=[feat1, feat2, feat3],
                     colLabels=['Train%', 'Test%', 'Delta%'],
                     colColours=["salmon"]*3, rowColours=["skyblue"]*3,
                     cellLoc='center', loc='upper center')
    ax.set_title('Test/Train Distribution Check',
                 fontweight ="bold")
    ax.axis('tight')
    ax.axis('off')
    plt.show()

def scplot(df, feat1, feat2):
    """
    Creates a scatter plot between 2 features (In our case, 'Age' and something else).
    :param df: The Data.
    :param feat1: Feature #1.
    :param feat2: Feature #2.
    :return: Scatter Plot.
    """
    plt.scatter(df.loc[:, feat1], df.loc[:, feat2])
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.show()