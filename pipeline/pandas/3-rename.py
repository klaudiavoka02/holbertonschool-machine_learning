import pandas as pd
import datetime


def from_file(filename, delimiter):

    df = pd.read.csv(filename, delimiter=delimiter)
    return df

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
print(df1.head())