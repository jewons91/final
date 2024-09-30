import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('C:\\big18\\final\\with\\513data.csv')

data_train = data[:int(len(data) * 0.8)]
data_test = data[int(len(data) * 0.8):]

# 데이터 전처리
# 5시퀀스로 묶기
sequence_length = 5
X, y = [], []

for i in range(len(data_train) - sequence_length):
    X.append(data_train.iloc[i:i + sequence_length].values) # 5개의 시퀀스 데이터
    if data_train.iloc[i + sequence_length, 0] - data_train.iloc[i + sequence_length - 1, 0] > 0:
        target = 2
    elif data_train.iloc[i + sequence_length, 0] - data_train.iloc[i + sequence_length - 1, 0] < 0:
        target = 1
    else:
        target = 0
    y.append(target) # 6번째 행의 답

X_train = np.array(X)
y_train = np.array(y)

# for i in range(len(data_test) - sequence_length):
#     X.append(data_test.iloc[i:i + sequence_length].values) # 5개의 시퀀스 데이터
#     y.append(data_test.iloc[i + sequence_length, 0]) # 6번째 행의 답
for i in range(len(data_test) - sequence_length):
    X.append(data_test.iloc[i:i + sequence_length].values) # 5개의 시퀀스 데이터
    if data_test.iloc[i + sequence_length, 0] - data_test.iloc[i + sequence_length - 1, 0] > 0:
        target = 2
    elif data_test.iloc[i + sequence_length, 0] - data_test.iloc[i + sequence_length - 1, 0] < 0:
        target = 1
    else:
        target = 0
    y.append(target) # 6번째 행의 답

X_test = np.array(X)
y_test = np.array(y)

# 입력 정의
inputs = layers.Input(shape=(sequence_length, 3))

# 1D Convolutional Layer
x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
x = layers.MaxPooling1D(pool_size=1)(x)

# # 추가적인 Conv Layer (선택 사항)
# x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
# x = layers.MaxPooling1D(pool_size=2)(x)

# LSTM Layer
x = layers.LSTM(50)(x)

# Dense Layer
outputs = layers.Dense(1, activation='linear')(x)  # 회귀 문제일 경우

# 모델 생성
model = models.Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split = 0.2)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'R² Score: {r2}')

# 모델 요약
model.summary()