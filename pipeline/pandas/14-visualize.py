from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# The column Weighted_Price should be removed
df = df.drop(columns=['Weighted_Price'])

# Rename the column Timestamp to Date
df = df.rename(columns[{'Timestamp': 'Date'}])

# Convert the timestamp values to date values
df['Date']=pd.to_datetime(df['Date'], units='s')

# Index the data frame on Date
df = df.set_index('Date')

# Missing values in Close should be set to the previous row value
df['Close'].fillna(method='ffill', inplace=True)

# Missing values in High, Low, Open should be set to the same row’s Close value
df['High'].filna(df['Close'], inplace=True)
df['Low'].filna(df['Close'], inplace=True)
df['Open'].filna(df['Close'], inplace=True)

#Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Plot the data from 2017 and beyond at daily intervals and group the values of the same day such that:
df = df['2017':].resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Open'], label='Open', color='blue')
plt.plot(df.index, df['High'], label='High', color='orange')
plt.plot(df.index, df['Low'], label='Low', color='green')
plt.plot(df.index, df['Close'], label='Close', color='red')
plt.plot(df.index, df['Volume_(BTC)'], label='Volume_(BTC)', color='purple')
plt.plot(df.index, df['Volume_(Currency)'], label='Volume_(Currency)', color='brown')
plt.xlabel('Date')
plt.legend()
plt.show()

print(df.shape)