import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

##filename = 'EURUSD_GMT+0_NO-DST.csv'
##filename = 'EURUSD_GMT+0_NO-DST1.csv'
filename = 'EURUSD_GMT_ts_bid_ask_5-4-02--9-9-19.csv' # this is like 11 GB so be careful

##data = pd.read_csv(filename, delimiter=',',skiprows=0)

# to deal with me having add a header once...
##try: ts = data['Date'].str.split(" ",n=1, expand=True)
##except KeyError:
##    data = pd.read_csv(filename, delimiter=',',skiprows=0,names=['Date','Bid','Ask'])
##    ts = data['Date'].str.split(" ",n=1, expand=True)
    
##data['Day']=ts[0]
##data['Time']=ts[1]
##data.drop(columns=['Date'],inplace=True)
##data['Day'] = pd.to_datetime(data['Day'],yearfirst=True) # parse days into date time frmt
##
##ts_day = data['Day']
##bid = data['Bid']
##ask = data['Ask']

##rng = pd.date_range(datetime.date(2015,1,1),datetime.date(2015,7,31))

### hacky
##freq = []
##for yy in range(2003,2020):
##    print('counting data in year ',yy,' ...')
##    for mm in range(1,13): # i think one too big
##        for dd in range(1,33): # hacky
##            try:
##                freq.append(data.loc[data['Day'] == pd.Timestamp(datetime.date(yy,mm,dd))].shape[0])
##            except ValueError: pass

##plt.plot(range(len(bid.values)),bid.values)
##plt.show()

##ticks_per_day = data['Day'].value_counts()

##fig, ax = plt.subplots()

##ax.set_xticks(data['Day'])
##ax.xaxis_date()

##ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
##ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m-%d"))
##plt.xticks(rotation=90)
##ax = data['Day'].value_counts().plot(kind='bar')
##ax.bar(data['Day'].values,data['Day'].value_counts().values)
##plt.show()


##chunksize = 10 ** 8
chunksize = 1000
c = 0
timer = dt.datetime.now()
# each chunk really is a new dataframe
for chunk in pd.read_csv(filename, chunksize=chunksize, names=['Date','Bid','Ask']):
    ems = (dt.datetime.now() - timer).microseconds
    timer = dt.datetime.now()
    print('micros:',ems)
    print(' ')
##    print('chunk ',c)
##    print(chunk)
##    process(chunk)
    if c > 2: break
    c += 1


##df.to_csv('my_csv.csv', mode='a', header=False)
