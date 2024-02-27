"""
Module for feature selection. It contain funcs for determination
which features from the data should be used to generate recommendations.
"""
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import LassoCV


def find_correlation(pandas_df, config):
    """
    Func for calc correlation between columns (parameters)
    :param pandas_df: pandas DataFrame with user data
    :param config: config dict
    :return: a
    """

    # # computing correaltion between all columns by 3 methods (spearman, kendall and pearcing) and add by abs
    correlated_pandas_df = (
                abs(pandas_df.corr()) + abs(pandas_df.corr(method='kendall')) + abs(pandas_df.corr(method='spearman')))

    # find, delete duplicates and return parameters with correlation
    correlating_parameters = []
    corr_coef = config["correlation_coef"]
    for i in range(len(correlated_pandas_df.index) - 1):
        for j in range(i + 1, len(correlated_pandas_df.columns) - 1):
            value = correlated_pandas_df.iloc[i, j]
            if corr_coef < value < 3:
                correlating_parameters.append((correlated_pandas_df.index[i],
                                               correlated_pandas_df.columns[j],
                                               correlated_pandas_df.iloc[i, j]))

    corr_pandas_df = pd.DataFrame(correlating_parameters, columns=["MainFeature", "CorrFeature", "Value"])

    return corr_pandas_df


def ols_find_features(pandas_df):
    """
    Func for OLS feature selection
    :param pandas_df: pandas DataFrame from OLS selection
    :return: pandas Series with selected features
    """
    feature_df = pandas_df.drop("GameTitle", axis=1)  # Feature Matrix

    # Backward Elimination
    target_columns = list(feature_df.columns)
    while len(target_columns) > 0:
        # add target column
        x_1 = sm.add_constant(feature_df[target_columns])

        # calc ols
        model = sm.OLS(pandas_df["GameTitle"], x_1).fit()

        # drop value greater than 0.05
        pvalue = pd.Series(model.pvalues.values[1:], index=target_columns)
        pmax = max(pvalue)
        feature_with_p_max = pvalue.idxmax()
        if pmax > 0.05:
            target_columns.remove(feature_with_p_max)
        else:
            break

    return pd.Series(target_columns, name="CorrFeature")


def lasso_find(pandas_df):
    """
    Func for Lasso feature selection
    :param pandas_df: pandas DataFrame from Lasso selection
    :return: pandas DataFrame with selected features
    """
    # from main featire column
    feature_df = pandas_df.drop("GameTitle", axis=1)

    # init LassoCV var
    reg = LassoCV()

    # calc Lasso coef
    reg.fit(feature_df, pandas_df["GameTitle"])

    # return Features as pandas DataFrame
    coef = pd.DataFrame([feature_df.columns, reg.coef_], index=["CorrFeature", "Value"]).transpose()
    coef = coef[coef['Value'] != 0.0]
    return coef
