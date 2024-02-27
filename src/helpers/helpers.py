"""
Helpers file whcih comtain some useful functions for common use
"""
import yaml as y


def upload_config(path):
    """
    Function for upload config as dict{key: dict, .....}
    :param path: str -> contain path for config
    :return: config_dict: dict{key: dict, .....} -> dictionary which contain config data
    """
    config_dict = {}
    try:
        with open(path, 'r', encoding='UTF-8') as f:
            config_dict = y.safe_load(f)
        return config_dict
    except Exception as ex:
        return ex


def change_str_to_int_values(pandas_df):
    """
    Func for replace str values to int values
    :param pandas_df: pandas Dataframe with str values
    :return: pandas_df: without str values
    """

    # prepare dicts fore replacing string values by int values
    dict_for_replacing = {
        "UserID": {str(f"user_{i}"): i for i in range(1, 1001)},
        "GameTitle": {str(f"game_{i}"): i for i in range(1, 51)},
        "Rating": {},
        "Age": {},
        "Gender": {"Male": 1, "Other": 2, "Female": 1},
        "Location": {"Africa": 5, "Europe": 3, "Asia": 4, "North America": 1, "Australia": 6, "South America": 2},
        "Device": {"Mobile": 1, "Console": 2, "PC": 3},
        "PlayTimeOfDay": {"Evening": 3, "Afternoon": 2, "Morning": 1, "Night": 4},
        "TotalPlaytimeInHours": {},
        "PurchaseHistory": {"Yes": 1, "No": 0},
        "InvolvementLevel": {"Intermediate": 2, "Master": 3, "Beginner": 1, "Expert": 4},
        "UserReview": {"Negative": 1, "Neutral": 2, "Positive": 3},
        "GameGenre": {"Strategy": 1, "Puzzle": 2, "RPG": 3, "Simulation": 4, "Action": 5, "Sports": 6, "Adventure": 7},
        "GameUpdateFrequency": {"Frequent": 1, "Occasional": 2, "Rare": 3},
        "SocialActivity": {"Moderate": 2, "Low": 1, "Active": 3},
        "LoadingTimeInSeconds": {},
        "GameSettingsPreference": {"Balanced": 2, "High Performance": 1, "High Graphics": 3}}

    # change str to int values
    for column in pandas_df.columns:
        pandas_df[column] = pandas_df[column].replace(dict_for_replacing[column])

    return pandas_df
