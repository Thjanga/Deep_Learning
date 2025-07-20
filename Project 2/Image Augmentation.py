import os
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'Project 2\dataset',
    image_size=(64,64),
    batch_size=128,
    subset='training', # 이름
    validation_split=0.2, # 20%로 쪼갠다
    seed = 1234,
    label_mode='int'
) # 80% > 2만장

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'Project 2\dataset',
    image_size=(64,64),
    batch_size=128,
    subset='validation',
    validation_split=0.2, # 20%로 쪼갠다
    seed = 1234,
    label_mode='int'
) # 20% > 5천장

def f(i,answer):
    i = tf.cast(i/255.0, tf.float32)
    return i, answer
    
train_ds = train_ds.map(f)
val_ds = val_ds.map(f)

model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal',input_shape=(64,64,3)), # 사진 뒤집기
    tf.keras.layers.RandomRotation(0.1), # 사진 돌리기
    tf.keras.layers.RandomZoom(0,1), # 사진 줌

    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'), # color는 3(rgb)
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    tf.keras.layers.Dropout(0.2), # overfitting 현상 완화
    tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation='sigmoid'), 
])

model.summary()

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(train_ds,validation_data=(val_ds), epochs=5)