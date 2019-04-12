# -*- coding=UTF-8 -*-
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

# load dataset
def parser(x):
    return datetime.strptime('190'+x,'%Y-%m')

series = read_csv('C:/kyy/test/shampoo-sales.csv',header=0,parse_dates=[0],index_col=0,squeeze=True,date_parser=parser)
# summarize first few rows
print(series.head())
#line plot
series.plot()
pyplot.show()
# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]
print(train)
print("test",test)

history = [x for x in train]
predictions = []
for i in range(len(test)):
    predictions.append(history[-1])
    history.append(test[i])

# report performance
rmse = sqrt