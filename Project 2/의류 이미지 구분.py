import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()



# print(trainX[0])
# print(trainX.shape())

trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape((trainX.shape[0],28,28,1))
testX = testX.reshape((testX.shape[0],28,28,1))



class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# 이미지 출력
# plt.imshow(trainX[0])
# plt.gray()
# plt.colorbar()
# plt.show()

model = tf.keras.Sequential([
    # Convolution layer 32개의 다른 feature 생성
    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Dense(128,input_shape=(28,28),activation='relu'),
    tf.keras.layers.Flatten(), # 행렬을 1차원으로 압축해줌
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax'), # 확률이면 카테고리 갯수만큼
])


# flatten은 응용력이 없어질 수 있음 > convolutional layer로 feature extraction
# sigmoid = binary 예측문제에 사용
# softmax = 카테고리 예측문제에 사용, 확률을 다 더하면 1이 나옴


# 모델 아웃라인 출력 input_shape="" 입력해야함
model.summary()
# exit()


# 카테고리 예측문제에서 사용하는 loss
# epoch 1회 끝날 때마다 평가 validation_data=(testX,testY) 
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(trainX,trainY,validation_data=(testX,testY), epochs=5)

# 모델 평가 학습이 끝난 후
# score = model.evaluate(testX,testY)
# print(score)
# overfitting 현상
