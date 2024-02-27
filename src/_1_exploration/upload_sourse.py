"""
File contain functions for uploading data from *.parquet
    and
represent in pandas dataframe type
"""
from pandas import read_parquet


class NonePathException(Exception):
    pass


def upload_parquet(path=None):
    """
    Function for upload data from parquet
    :param path: str -> path for a parquet file
    :return: pandas DataFrame
    """

    if path is None or path == "":
        return NonePathException(f"File by path'{path}' are not exist. Try to correct your path.")

    try:
        game_data_df = read_parquet(path)
        return game_data_df
    except Exception as ex:
        return ex


if __name__ == "__ipload_parquet__":
    print(upload_parquet())
