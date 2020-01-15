from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time
# ~ import matplotlib.dates as mdates
import pickle
from keras.models import load_model

filename = 'EURUSD_GMT_ts_bid_ask_5-4-02--9-9-19.csv' # this is like 11 GB so be careful

pkl_file = open('year_inds.pkl', 'rb')
year_indices = pickle.load(pkl_file)
pkl_file.close()


def time_parse(time_string):
    nofrag = time_string.split('.')[0:-1]
    nofrag='.'.join(nofrag)
    frag = time_string.split('.')[-1]
    
    nofrag_dt = dt.datetime.strptime(nofrag,'%Y.%m.%d %H:%M:%S')
    dt_time = nofrag_dt.replace(microsecond=int(frag)*1000)
    return dt_time
    

user_load = input('Load model from disk? [yes] [no]')
if  user_load =='yes':
    # save model and architecture to single file
    model = load_model("fin_ml_model.h5")
    model.summary()
elif user_load =='no':
    print('compiling a new model')
    model = Sequential()
    model.add(LSTM(14, return_sequences=False, input_shape=(None, 1))) # variable input length x 1 feature
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
else:
    print('unknown command')


try:
    chunksize = 20000 # how many rows of data to load at once, can load thousands at a time
    
    look_back = 30 # minutes 
    label_look_ahead = 1 # minutes ahead to predict
    pip_thresh = 0.0002 # pip delta threshold # ------ this feels a little small, but it helps even the class distributions
    
    
    trained_ups = 0
    trained_downs = 0
    trained_stays = 0
    
    finished_2018 = False
    c = 0
    timer = dt.datetime.now()
    init_timer = dt.datetime.now()

    loss_hist = []
    plt.figure(1)
    print('this can take a while to load if skipping many years of data...')
    for chunk in pd.read_csv(filename, skiprows=year_indices['2018'], chunksize=chunksize, names=['Date','Bid','Ask']):
        
        # loop time tracking
        ems = (dt.datetime.now() - timer).microseconds # ems = elapsed microseconds
        timer = dt.datetime.now()
        
        if c % 1 == 0:
            print(time_parse(chunk.loc[chunk.index[0],'Date']))
            print('chunks: ',c,' micros:',ems,' total run time so far: ',(dt.datetime.now()-init_timer))
            print(' ')
            
        
        # chunk statistics
        chunk_ups = 0
        chunk_downs = 0
        chunk_stays = 0
        
        seq_found_flag = False
        seq_start_ind = chunk.index[0]
        seq_init_time = time_parse(chunk.loc[seq_start_ind,'Date'])
        loop_count = 0
        predicted = 0
        predict_next_N = False
        for r in range(chunk.index[0]+1,chunk.index[-1]-300):
            # ~ print('r:',r,' seq_start_ind:',seq_start_ind)
            # ~ print((time_parse(chunk.loc[r,'Date']) - seq_init_time).seconds)
            # ~ if loop_count > 3: break
            if (time_parse(chunk.loc[r,'Date']) - seq_init_time).seconds >= 60*45: # 45 minutes is too much, really looking for sunday big changes
                seq_start_ind += 1 # move the tail of window up
                # ~ print('too long')
                continue # don't evaluate anything else
            elif (time_parse(chunk.loc[r,'Date']) - seq_init_time).seconds >= 60*look_back: # less than 45, and greater than 30 minutes of data
                seq_found_flag = True
                pass
            elif (time_parse(chunk.loc[r,'Date']) - seq_init_time).seconds < 60*look_back:
                # ~ print('too short')
                continue # don't evaluate anything else, let the head of the window move further forward
            # only evaluating all of this if the sequence is of the right length
            
            x_train = chunk.loc[seq_start_ind:r,'Bid'].values
            
            ii = 1
            while (time_parse(chunk.loc[r+ii,'Date']) - time_parse(chunk.loc[r,'Date'])).seconds < 60*label_look_ahead:
                ii += 1
                if ii > 2000: 
                    print('BADDDD')
                    break
            y_val = chunk.loc[r+ii-1,'Bid']
            if y_val - chunk.loc[r,'Bid'] >= pip_thresh:
                            # the second element of the array is to fix the categorical number of classes
                y_train = [0,2] # pip increase by >= thresh
                chunk_ups += 1
            elif y_val - chunk.loc[r,'Bid'] <= -pip_thresh:
                y_train = [1,2] # pip decrease by >= thresh
                chunk_downs += 1
            else: 
                y_train = [2,2] # pip stayed the same within tolerance
                chunk_stays += 1
            
            if seq_found_flag and x_train.shape[0] < 1000:
                continue # assuming this is a periodic sunday problem
            
            class_imbalance_tolerance = 100
            if y_train[0]==0:
                if trained_ups > trained_downs + class_imbalance_tolerance or trained_ups > trained_stays + class_imbalance_tolerance:
                    continue # don't train or will create a class distribution imbalance
                else: trained_ups += 1
            elif y_train[0]==1:
                if trained_downs > trained_ups + class_imbalance_tolerance or trained_downs > trained_stays + class_imbalance_tolerance:
                    continue # don't train or will create a class distribution imbalance
                else: trained_downs += 1
            elif y_train[0]==2:
                if trained_stays > trained_ups + class_imbalance_tolerance or trained_stays > trained_downs + class_imbalance_tolerance:
                    continue # don't train or will create a class distribution imbalance
                else: trained_stays += 1
            else: print('unexpected error')
            
            y_train = np.reshape(to_categorical(y_train)[0,:],(1,3))
            x_train = np.reshape(x_train,(1,len(x_train),1))
            
            # ~ print('y')
            # ~ print(y_train.shape)
            # ~ print(y_train)
            if predict_next_N:
                print('Actual: ',y_train)
                print('Predicted: ',model.predict(x_train))
                predicted += 1
                if predicted > 2: 
                    predict_next_N = False
                    predicted = 0
            else: 
                # train model on sequence
                e_hist = model.fit(x_train, y_train, steps_per_epoch=1, epochs=1, verbose=0) # can vary steps_per_epoch
                loss_hist.append(e_hist.history['loss'][0])
            
            if loop_count % 10 == 0:
                plt.figure(1)
                plt.clf()
                plt.plot(loss_hist,'bo')
                plt.xlabel('epoch')
                plt.ylabel('train loss')
                plt.pause(0.001)
                
                print(' ')
                print('Chunk ',c,' current data date:',time_parse(chunk.loc[r,'Date']))
                
                if time_parse(chunk.loc[r,'Date']).year == 2019: 
                    finished_2018 = True
                    break
                
                predict_next_N = True

                
            # ~ print('found an x sequence and y label')
            # ~ print(x_train.shape)
            # ~ print(y_val)
            
            seq_start_ind += 1 
            # ~ print('r:',r,' seq_start_ind:',seq_start_ind)
            
            # cool moving plot of the tick data
            # ~ plt.figure(2)
            # ~ plt.clf()
            # ~ plt.plot(x_train)
            # ~ plt.pause(0.001)
            
            loop_count += 1
            
        print('chunk statistics')
        print('pip increases:',chunk_ups)
        print('pip decreases:',chunk_downs)
        print('pip stays:',chunk_stays)
        print('total trained up:',trained_ups,' downs:',trained_downs,' stays:',trained_stays)
        # ~ time.sleep(3)
        
        if finished_2018:
            print('stopping since all of 2018 data has been processed')
            break
        c += 1
except KeyboardInterrupt: pass

if input('Save model? [yes] [no]') =='yes' or finished_2018:
    # save model and architecture to single file
    model.save("fin_ml_model.h5")
    print("Saved model to disk")





