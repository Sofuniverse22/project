#0.사용할 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#랜덤시드 고정시키기
np.random.seed(5)

#1.데이터 준비하기
dataset = pd.read_csv('total_set.csv')
dataset = dataset.values
dataset = dataset.astype('float32')  # float형

#2.데이터셋 생성하기

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(1e-8, 1 - 1e-8))
for i in range (0,16):
    x_train_i = dataset[:, i]
    x_train_i = x_train_i[:, None]
    x_train_i = scaler.fit_transform(x_train_i)
x = np.concatenate([x_train_i,], axis=1)

y = dataset[:,16]
x = dataset[:,:16]

x_train = x[0:520,]
y_train = y[0:520,]
x_test = x[520:,]
y_test = y[520:,]

#3.모델 구성하기
model = Sequential()
model.add(Dense(35, input_dim=16, activation='tanh'))
model.add(Dense(35, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#4.모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#5.모델 학습시키기
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 10)

#6.모델 평가하기
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
print(hist.history['loss'])

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y', label='train loss')
acc_ax.plot(hist.history['acc'],'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 7. 예측하기
y_predict = model.predict(x_test)

df = pd.DataFrame(y_predict)
df.insert(1,'y_test',y_test)
df.to_csv("predict_total_binary.csv")