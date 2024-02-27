"""
Main file for launch main func for work with all tasks
Tasks - research and develop a recomendation system. Current system should oriented on users activities in game.
As source a dataset ml_test_rec_sys.parquet, which contain all needable data.

Main aims:
Data exploration: Analyze the data provided, formulate hypotheses about user behavior, and visualize your findings.
Data prioritization: Determine which aspects of the data are most relevant to your recommendations and justify your
choices.
System development: Create a recommendation system using machine learning methods. Test different models, evaluate
their quality and choose the best option.
Personalization: Develop a mechanism that suggests games to the user based on their interests and behavior.
"""
from os import getcwd
from sklearn.model_selection import train_test_split

import src._1_exploration.base_statistic as base
import src._1_exploration.upload_sourse as src
import src._2_prioritization.finding_params as fnd
import src._3_sys_dev.rec_models as model
import src.helpers.helpers as help_

config_path = f"{getcwd()}\\config.yaml"


def main():
    """
    Main func with all aims realization
    """

    # uploading config and source file
    config_dict = help_.upload_config(config_path)
    if type(config_dict) is not dict:
        print(f"Wrong config type!!! Check config.yaml!!! It must be by path: {config_path}")
        exit()
    path_to_source = config_dict["path_to_source"]
    user_data_df = src.upload_parquet(path_to_source)

    #
    base_statistics_df = base.data_analysis(user_data_df, config_dict)
    # conole view
    print(base_statistics_df.to_string())

    # standartise valuse (chage str to int)
    user_data_df_ = help_.change_str_to_int_values(user_data_df)
    user_data_df = user_data_df_.drop("UserID", axis=1)

    # find corr params by (piercing, spearman, kendall)
    correlate_params = fnd.find_correlation(user_data_df, config_dict)
    corr_params_by_gametitle = correlate_params.loc[correlate_params["MainFeature"] == "GameTitle"]
    print(f"\nCorrelating params by (piercing, spearman, kendall) coefs:\n{corr_params_by_gametitle['CorrFeature']}")

    # find corr params by OLS
    ols_params = fnd.ols_find_features(user_data_df)
    print(f"\nCorrelating params by OLS:\n{ols_params}")

    # find corr params by Lasso
    lasso_params = fnd.lasso_find(user_data_df)
    print(f"\nCorrelating params by Lasso method:\n{lasso_params['CorrFeature']}")

    # collect all features
    all_corr_features = list(set(corr_params_by_gametitle["CorrFeature"]) |
                             set(ols_params) |
                             set(lasso_params["CorrFeature"]))
    all_corr_features.sort()

    # prepare dataset for rec models
    full_dataset_x = user_data_df[all_corr_features]
    full_dataset_y = user_data_df["GameTitle"]

    train_dataset_x, test_dataset_x, train_dataset_y, test_dataset_y = train_test_split(full_dataset_x, full_dataset_y,
                                                                                        test_size=0.2, random_state=45)

    # k-neigbours algorhythm
    k_neighbours_model = model.kneighbours_model(train_dataset_x, test_dataset_x, train_dataset_y, test_dataset_y)

    # gradient boosting algorhythm
    grad_boost_model = model.grad_boost(train_dataset_x, test_dataset_x, train_dataset_y, test_dataset_y)

    # MNN algorhythm
    mnn = model.mnn_model(train_dataset_x, test_dataset_x, train_dataset_y, test_dataset_y)

    return {"k_neighbours_model": k_neighbours_model.predict,
            "grad_boost_model": grad_boost_model.predict, "mnn": mnn.predict}, all_corr_features


if __name__ == "__main__":
    main()
