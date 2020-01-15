from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


X_train = np.zeros((1, 200))
X_train[0, range(0,200,5)] = 1 # every 5th value is a one

model = Sequential()
model.add(LSTM(4, return_sequences=False, input_shape=(None, 1))) # variable input length x 1 feature
##model.add(LSTM(2, return_sequences=True))
##model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))


##model.add(Dense(4, activation='relu'))

print(model.summary()) # had a 90 inside, not sure what that does

model.compile(loss='categorical_crossentropy',
              optimizer='adam')

sequence_length = 13
##def train_generator():
##    index = 0
##    while True:
####        sequence_length = np.random.randint(10, 100) # variable sequence length
##        
####        x_train = np.random.random((1000, sequence_length, 5)) # 1000 examples, various length, 5 features...?
####        # y_train will depend on past 5 timesteps of x
####        y_train = x_train[:, :, 0]
####        for i in range(1, 5):
####            y_train[:, i:] += x_train[:, :-i, i]
####        y_train = to_categorical(y_train > 2.5)
##
##        
##        x_train = np.reshape(X_train[0,index:index+sequence_length],(1,sequence_length,1))
##        y_train = np.reshape(X_train[0,index+sequence_length+3],(1,)) # trying to predict the ones, 3 time steps into the future
##
##        
##        print('x')
##        print(x_train.shape)
##        print(x_train)
##        print('y')
##        print(y_train.shape)
##        print(y_train)
##        print('i will need this to be stateful')
##        print('time step: ',index)
##        yield x_train, y_train

##model.fit_generator(train_generator(), steps_per_epoch=5, epochs=2, verbose=1)

### more hacky way to do this
trainX = np.asarray([])
trainY = np.asarray([])
t = 0
predict_forward = 3
while True:
    if trainX.shape[0] == 0: trainX = np.reshape(X_train[0,t:t+sequence_length],(1,sequence_length,1))
    else:
        trainX = np.concatenate((trainX,np.reshape(X_train[0,t:t+sequence_length],(1,sequence_length,1))),0)
    if trainY.shape[0] == 0: trainY = np.reshape(X_train[0,t+sequence_length+predict_forward],(1,))
    else:
        trainY = np.concatenate((trainY,np.reshape(X_train[0,t+sequence_length+predict_forward],(1,))),0)
    t += 1
    if t > 200- sequence_length - predict_forward - 5: break 

trainY = to_categorical(trainY)

train_size = int(trainX.shape[0] * 0.95)
test_size = trainX.shape[0] - train_size

testX = trainX[train_size:,:,:]
trainX = trainX[0:train_size,:,:]

testY = trainY[train_size:]
trainY = trainY[0:train_size]

loss_hist = []
epochs = 50
for e in range(epochs):
    print('epoch: ',e,'/',str(epochs))
    var_trainX = trainX[e,np.random.randint(3):sequence_length-np.random.randint(3),:]
##    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
    e_hist = model.fit(trainX, trainY, steps_per_epoch=30, epochs=1, verbose=2)
    loss_hist.append(e_hist.history['loss'][0])

##model.fit(trainX, trainY, epochs=100, batch_size=8, verbose=2)
##model.fit(trainX, trainY, steps_per_epoch=30, epochs=100, verbose=2)

preds = model.predict(testX)
outs = np.zeros((preds.shape[0],1))
outs[preds[:,0]>0.5] = 1
# plot some results
##plt.plot(X_train[0,train_size-30:train_size],'bo')
##plt.plot(range(X_train[0,train_size-30:train_size].shape[0],X_train[0,train_size-30:train_size].shape[0]+len(outs)),X_train[train_size:],'kx')
##plt.plot(range(X_train[0,train_size-30:train_size].shape[0],X_train[0,train_size-30:train_size].shape[0]+len(outs)),outs,'ro')

plt.figure(1)
plt.plot(loss_hist)
plt.xlabel('epoch')
plt.ylabel('train loss')


plt.figure(2)
plt.plot(testY[:,0],'bo-')
plt.plot(outs,'kx')
plt.show()
