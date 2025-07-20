import os
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

# 데이터 불러오기
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'Project 2\dataset',
    image_size=(150,150),
    batch_size=64,
    subset='training', 
    validation_split=0.2, 
    seed = 1234,
    label_mode='int'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'Project 2\dataset',
    image_size=(150,150),
    batch_size=64,
    subset='validation',
    validation_split=0.2, 
    seed = 1234,
    label_mode='int'
) 

# print(train_ds)

# 전처리
def f(i,answer):
    i = tf.cast(i/255.0, tf.float32)
    return i, answer
    
train_ds = train_ds.map(f)
val_ds = val_ds.map(f)

# InceptionV3 불러오기
inception_model = InceptionV3(input_shape=(150,150,3),include_top=False,weights=None)
inception_model.load_weights(r'Project 2\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

inception_model.summary()

# 레이어 고정
for i in inception_model.layers:
    i.trainable = False # w값 고정

# mixed6 이후 레이어만 fine-tune
unfreeze = False
for i in inception_model.layers:
    if i.name == 'mixed6':
        unfreeze = True
    if unfreeze == True:
        i.trainable = True

# 마지막 레이어 가져오기
lastlayer = inception_model.get_layer('mixed7')

print(lastlayer)
print(lastlayer.output)
print(lastlayer.output.shape)

# 사용자 정의 층
layer1 = tf.keras.layers.Flatten()(lastlayer.output)
layer2 = tf.keras.layers.Dense(1024,activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
layer3 = tf.keras.layers.Dense(1,activation='sigmoid')(drop1)

# 전체 모델 구성
model = tf.keras.Model(inception_model.input,layer3)

# 컴파일 및 학습
model.compile(loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            metrics=['acc'])

model.fit(train_ds,validation_data=val_ds,epochs=2)