import pandas as pd
import matplotlib.pyplot as plt

#Import CSV to df, change the path to the csv location
df = pd.read_csv('/home/maverick/Documents/movingaverage/OrderTable.csv')

#Turn order_date into datetime, change the names to match column names
df['Order_Date'] = pd.to_datetime(df["Order_Date"])

#calculate the 30-day moving average
groupedData = df.groupby('Order_Date')['Price'].sum().reset_index()

#calculate the 7-day moving average
groupedData['MovingAve7'] = groupedData['Price'].rolling(window=7).mean()

#calculate the 30-day moving average
groupedData['MovingAve30'] = groupedData['Price'].rolling(window=30).mean()

#print the groupedData table
print(groupedData)

#plot the 7 day moving average
plt.plot(groupedData['Order_Date'], groupedData['MovingAve7'])
plt.xlabel('Order Date')
plt.ylabel('Moving Average')
plt.title('7-day Moving Average')
plt.show()

#plot the 30 day moving average
plt.plot(groupedData['Order_Date'], groupedData['MovingAve30'])
plt.xlabel('Order Date')
plt.ylabel('Moving Average')
plt.title('30-day Moving Average')
plt.show()