import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv(r'Project 1\gpascore.csv')

print(data.isnull().sum()) # 빈칸 확인
data = data.dropna() # 빈칸제거
# data.fillna() # 빈칸을 채워줌

ydata = data['admit'].values # 답
xdata = []

for i,rows in data.iterrows(): # i 행번호 rows 한행씩
    xdata.append([rows['gre'],rows['gpa'],rows['rank']])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
])

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'] )

model.fit(np.array(xdata),np.array(ydata),epochs=1000)
# x = [[데이터1],[데이터2],[데이터3]]

# 예측
model.predict([[750,3.70,3],[400,2.2,1]])

