df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.rename(columns ={'Timestamp':'Datetime'})
df['Daytime']=pd.to_datetime(df['Datetime'], unit='')
df = df.loc[:,['Datetime','Close']]


print(df.head())