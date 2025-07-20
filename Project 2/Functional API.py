import tensorflow as tf 
import numpy as np
# from tensorflow.keras.utils import plot_model

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28,28,1) )
testX = testX.reshape( (testX.shape[0], 28,28,1) )

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3)

# 레이어를 그림으로 보기
# from tensorflow.keras.utils import plot_model

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

input1 = tf.keras.layers.Input(shape=[28,28])
flatten1 = tf.keras.layers.Flatten()(input1)
dense1 = tf.keras.layers.Dense(28*28,activation='relu')(flatten1)
reshape1 = tf.keras.layers.Reshape((28,28))(dense1) # 1차원 행렬을 고차원으로 변경 (이전레이어와 총 node수가 같아야 함)

concat1 = tf.keras.layers.Concatenate()([input1,reshape1]) # 합치기
flatten2 = tf.keras.layers.Flatten()(concat1)
output = tf.keras.layers.Dense(10,activation='softmax')(flatten2)

model = tf.keras.Model(input1,output)

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
