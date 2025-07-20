import tensorflow as tf 
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28,28,1) )
testX = testX.reshape( (testX.shape[0], 28,28,1) )

log_dir1 = 'logs/{}'.format(datetime.now().strftime("model1_%Y%m%d-%H%M%S"))
log_dir2 = 'logs/{}'.format(datetime.now().strftime("model2_%Y%m%d-%H%M%S"))

# 첫번째 모델
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3,callbacks=[TensorBoard(log_dir=log_dir1)])

# 두번째 모델
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# early stopping
es = EarlyStopping(monitor='val_accuracy',patience=5,mode='max')

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=300,callbacks=[TensorBoard(log_dir=log_dir2), es])

# tensorboard --logdir logs