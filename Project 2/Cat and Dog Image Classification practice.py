import os
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'Project 2\dataset',
    image_size=(64,64),
    batch_size=64,
    subset='training', # 이름
    validation_split=0.2, # 20%로 쪼갠다
    seed = 1234,
    label_mode='int'
) # 80% > 2만장

# ((xxxxx),(yyyyy)) > train_ds

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'Project 2\dataset',
    image_size=(64,64),
    batch_size=64,
    subset='validation',
    validation_split=0.2, # 20%로 쪼갠다
    seed = 1234,
    label_mode='int'
) # 20% > 5천장

print(train_ds)

# 전처리함수
def f(i,answer):
    i = tf.cast(i/255.0, tf.float32)
    return i, answer
    
train_ds = train_ds.map(f)
val_ds = val_ds.map(f)

# 사진 미리보기
# for i, answer in train_ds.take(1):
#     print(i,answer)
#     plt.imshow(i[0].numpy().astype('uint8'))
#     plt.show()

# 학습
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(64,64,3)), # color는 3(rgb)
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    # tf.keras.layers.Dropout(0.3), # overfitting 현상 완화

    tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D((2,2)),

    # tf.keras.layers.Flatten(),
    tf.keras.layers.GlobalAveragePooling2D(), 

    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1,activation='sigmoid'), 
])

model.summary()

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)
model.fit(train_ds,validation_data=(val_ds), epochs=30,callbacks=[es])