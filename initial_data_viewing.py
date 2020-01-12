import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

##filename = 'EURUSD_GMT+0_NO-DST.csv'
##filename = 'EURUSD_GMT+0_NO-DST1.csv'
filename = 'EURUSD_GMT_ts_bid_ask_5-4-02--9-9-19.csv' # this is like 11 GB so be careful

##data = pd.read_csv(filename, delimiter=',',skiprows=0)

Num_ticks = 100
Time_forcast = 5*60 # seconds
Pip_thresh = 20

def time_parse(time_string):
    nofrag = time_string.split('.')[0:-1]
    nofrag='.'.join(nofrag)
    frag = time_string.split('.')[-1]
    
    nofrag_dt = dt.datetime.strptime(nofrag,'%Y.%m.%d %H:%M:%S')
    dt_time = nofrag_dt.replace(microsecond=int(frag)*1000)
    return dt_time
    
    
year_indices = {}

def process_data(dat, Nticks, Tforcast, Pthresh):
    global year_indices
    
    start_time = time_parse(dat.loc[dat.index[0],'Date'])
    end_time = time_parse(dat.loc[dat.index[-1],'Date'])
    elapsed_time = end_time - start_time
    
    if str(start_time.year) not in year_indices.keys():
        year_indices[str(start_time.year)] = dat.index[0]
        print(year_indices)
    # ~ dt_list = []
    # ~ for row in dat.index:
        # ~ dt_list.append(time_parse(dat.loc[row,'Date']))
    
    # ~ dat['dt_obj'] = dt_list
        
    # ~ print('start year: ',start_time.year,' month: ',start_time.month,' day: ',start_time.day)
    # ~ print('elapsed:',elapsed_time)
    
    # ~ print('Start Bid',dat.loc[dat.index[0],'Bid'])
    # ~ print('Start Date',dat.loc[dat.index[0],'Date'])
    # ~ print('End Date',dat.loc[dat.index[-1],'Date'])
    
    pip_range = (max(dat['Bid'].values)-min(dat['Bid'].values))*10000
    # ~ print('Range of Bid variation:',pip_range)
    
    
    
    return pip_range > 20 # nothing for now

try:
    ##chunksize = 10 ** 8
    chunksize = 1000
    c = 0
    timer = dt.datetime.now()
    init_timer = dt.datetime.now()
    # each chunk really is a new dataframe
    # skiprows = 100000000 gets to around 2010
    # skiprows=200000000 around 2016
    for chunk in pd.read_csv(filename, skiprows=0, chunksize=chunksize, names=['Date','Bid','Ask']):
        ems = (dt.datetime.now() - timer).microseconds # ems = elapsed microseconds
        timer = dt.datetime.now()
        print('micros:',ems,' total: ',(dt.datetime.now()-init_timer))
        print(' ')
        
    ##    print('chunk ',c)
        # ~ print(chunk)
        
        flag = process_data(chunk, Num_ticks, Time_forcast, Pip_thresh)
        
        # ~ if flag: 
        # ~ bids = chunk['Bid'].values
        # ~ plt.clf()
        # ~ plt.plot(bids)
        # ~ plt.show()
        # ~ plt.pause(0.75)
        #if c > 300000: break
        c += 1
except KeyboardInterrupt: pass

print(year_indices)

# ~ import pickle
# ~ f = open("year_inds.pkl","wb")
# ~ pickle.dump(year_indices,f)
# ~ f.close()

# ~ pkl_file = open('year_inds.pkl', 'rb')
# ~ file = pickle.load(pkl_file)

##df.to_csv('my_csv.csv', mode='a', header=False)
