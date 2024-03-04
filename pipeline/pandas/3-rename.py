import pandas as pd
import datetime


def from_file(filename, delimiter):

    data = pd.read_csv(filename, delimiter=delimiter, engine='python')
    return data

    
file = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
file.rename(columns={"Timestamp": "Datetime"}, inplace=True)
file['Datetime'] = pd.to_datetime(file['Datetime'], unit='s')
print(file[['Datetime', 'Close']].tail())