import tensorflow as tf 
import numpy as np

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
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3)

# 모델 전체 저장
model.save('model1.keras')
# 모델 불러오기
loaded_model = tf.keras.models.load_model('model1.keras')

loaded_model.summary()
loaded_model.evaluate(testX,testY)

# 가중치만 저장
콜백함수 = tf.keras.callbacks.ModelCheckpoint(
    filepath='model1.weights.h5',
    monitor='val_acc', # val_acc가 최대가 되는 checkpoint만 저장
    mode='max',
    save_weights_only=True,
    save_freq='epoch'
) 