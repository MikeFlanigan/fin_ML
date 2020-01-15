import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
##filename = 'EURUSD_GMT+0_NO-DST.csv'
##filename = 'EURUSD_GMT+0_NO-DST1.csv'
filename = 'EURUSD_GMT_ts_bid_ask_5-4-02--9-9-19.csv' # this is like 11 GB so be careful

##data = pd.read_csv(filename, delimiter=',',skiprows=0)



pkl_file = open('year_inds.pkl', 'rb')
year_indices = pickle.load(pkl_file)
pkl_file.close()
print(year_indices)

yr = 2003
for i in range(16):
    print('Ticks in year ',str(yr),year_indices[str(yr+1)]-year_indices[str(yr)])
    yr += 1 
    
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
    
    
# ~ year_indices = {}

def process_data(dat, Nticks, Tforcast, Pthresh):
    global year_indices
    
    start_time = time_parse(dat.loc[dat.index[0],'Date'])
    end_time = time_parse(dat.loc[dat.index[-1],'Date'])
    elapsed_time = end_time - start_time
    
    
    # ~ Was used for finding the start tick index of each year
    # ~ if str(start_time.year) not in year_indices.keys():
        # ~ year_indices[str(start_time.year)] = dat.index[0]
        # ~ print(year_indices)
        
    dt_list = []
    for row in dat.index:
        dt_list.append(time_parse(dat.loc[row,'Date']))
    
    dat['dt_obj'] = dt_list
        
    print('start year: ',start_time.year,' month: ',start_time.month,' day: ',start_time.day)
    print('elapsed time:',elapsed_time.seconds/60,' mins ',elapsed_time.seconds,' secs')
    
    # ~ print('Start Bid',dat.loc[dat.index[0],'Bid'])
    # ~ print('Start Date',dat.loc[dat.index[0],'Date'])
    # ~ print('End Date',dat.loc[dat.index[-1],'Date'])
    
    pip_range = (max(dat['Bid'].values)-min(dat['Bid'].values))*10000
    print('Range of Bid variation:',pip_range)
    
    esecs = np.linspace(0,elapsed_time.seconds,chunksize)
    
    bids = dat['Bid'].values
    input('press enter')
    plt.clf()
    plt.plot(esecs, bids)
    plt.ylim(min(bids) - 0.0010,min(bids)+0.0050)
    plt.pause(0.001)
    
    
    return pip_range > 20 # nothing for now

try:
    DF_2018 = pd.DataFrame()
    ##chunksize = 10 ** 8
    chunksize = 1
    c = 0
    timer = dt.datetime.now()
    init_timer = dt.datetime.now()
    # each chunk really is a new dataframe
    # skiprows = 100000000 gets to around 2010
    # skiprows=200000000 around 2016
    for chunk in pd.read_csv(filename, skiprows=year_indices['2018']+674365, chunksize=chunksize, names=['Date','Bid','Ask']):
        
        ems = (dt.datetime.now() - timer).microseconds # ems = elapsed microseconds
        timer = dt.datetime.now()
        
        if c % 100 == 0:
            print('micros:',ems,' total: ',(dt.datetime.now()-init_timer))
            print(' ')
        
        if c == 0: 
            DF_2018 = chunk[['Date','Bid']]
        else: 
            if time_parse(chunk.loc[chunk.index[0],'Date']).year == 2018:
                DF_2018 = DF_2018.append(chunk[['Date','Bid']])
            else:
                print('out of 2018')
                break
        
        
        # ~ print('chunk ')
        # ~ print(chunk)
        
        # ~ flag = process_data(chunk, Num_ticks, Time_forcast, Pip_thresh)


        # ~ if c > 15: break
        c += 1
except KeyboardInterrupt: pass

DF_2018.to_csv('DF_2018.csv',index=False)

# ~ print(DF_2018)
# ~ print(year_indices)

##df.to_csv('my_csv.csv', mode='a', header=False)
