import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
def impute(df):
    """
    Replaces binary classification strings with 1 and 0.
    Imputes missing values (NANs) in a dataframe using an iterative imputer
    that takes the 2 nearest features into consideration.
    :param df : Dataframe with missing values (NANs).
    :return: Dataframe with missing values (NANs) replaced by the imputer.
    """
    df = df.replace(['Yes', 'Positive', 'No', 'Negative', 'Male', 'Female'], [1, 1, 0, 0, 1, 0]).copy()

    df_train = df.iloc[0:500, :]

    imp_most_freq = IterativeImputer(max_iter=100, n_nearest_features=2, random_state=42,
                                     initial_strategy='most_frequent')
    imp_most_freq.fit(df_train)
    df_new = imp_most_freq.transform(df).copy()
    df_new = df_new.round()
    df_new = pd.DataFrame(df_new)
    df_new.columns = df.columns
    return df_new
