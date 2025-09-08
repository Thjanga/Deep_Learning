"""
Generator 모델 + Discriminator 모델 (TensorFlow / Keras)

학습 전략 개요
1) Discriminator(D) 단독 학습
    - "진짜 이미지"와 "가짜 이미지(G가 만든 샘플)"를 섞어 D가 구분하도록 학습

2) Generator(G) 학습
    - 잠재벡터 z ~ U(-1, 1)에서 샘플한 노이즈를 G에 넣어 가짜 이미지를 생성
    - 이 가짜 이미지를 D에 통과시켜 "진짜(라벨=1)"라고 믿게 만들도록 G의 파라미터를 업데이트
    - 이때 D의 가중치는 동결(Freeze)하여 G만 업데이트

3) 연결
    - GAN = G ∘ D (단, 학습 시 D는 동결)
    - loss(GAN, target=1)을 최소화 → G가 만든 이미지가 D 기준으로 진짜처럼 보이게 만듦

주의) Discriminator 모델을 학습시킬 때는 GAN 모델이 아니라 Discriminator 모델을 사용해야 합니다.
"""

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1) 데이터 로딩 & 전처리
filelist = os.listdir('Project 6/img_align_celeba')

Images = []

for i in filelist[0:50000]:
    # 이미지 로드 → 얼굴 영역 대략 crop → 흑백변환(L) → 64x64 리사이즈
    NumberImage = Image.open('Project 6/img_align_celeba/'+i).crop((20,30,160,180)).convert('L').resize((64,64))
    Images.append(np.array(NumberImage))

Images = np.divide(Images,255)
# (N, 64, 64, 1) 형태로 변형 (Keras Conv2D 입력 규약)
images = Images.reshape(50000,64,64,1)

# 2) 모델 정의: Discriminator
# 입력: (64, 64, 1) 흑백 이미지
# 출력: 스칼라 확률(시그모이드) - "진짜일 확률"
discriminator = tf.keras.models.Sequential([
    # 다운샘플링: 특징 맵 수를 늘리며 해상도 절반씩 축소 (64→32→16)
    tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=[64,64,1]),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    # leakyRelu는 그냥 relu인데 0미만의 값을 다 0으로 바꾸는게 아니라 0.01을 곱해서 작게만 만들어줍니다.
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
]) 

noise_shape = 100

# 3) 모델 정의: Generator
# 입력: 100차원 잠재벡터 z
# 출력: (64,64,1) 이미지 (0~1 범위, sigmoid)
generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4 * 4 * 256, input_shape=(100,) ), 
    tf.keras.layers.Reshape((4, 4, 256)),
    # 업샘플링(Transpose Conv)으로 해상도 4→8→16→32→64
    tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])

# 4) GAN(Generator → Discriminator) 구성
GAN = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

"""
전부 다 트레이닝 안되게 설정하고 싶으면 

for layer in 모델.layers: 
    layer.trainable = False
이렇게 사용하면 됩니다.

주의점은 이 방법은 모델.compile() 전/후 둘다 사용가능합니다. 
"""

GAN.compile(optimizer='adam', loss='binary_crossentropy')


def pltimg():
    randnumber = np.random.uniform(-1, 1, size=(10,100))
    predict = generator.predict(randnumber)
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(predict[i].reshape(64,64), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Project 6/gan_img/img'+str(j)+'.png')

xdata = images

for j in range(300):  # epoch 300번
    print('epoch :', j)
    pltimg()

    for i in range(50000//128):
        if i % 100 == 0:
            print('batch :', i)

        # discriminator 학습
        realimages = xdata[i*128:(i+1)*128]
        marking_1 = np.ones((128,1))  # 진짜이미지 라벨 1
        discriminator.trainable = True
        loss1 = discriminator.train_on_batch(realimages, marking_1)  # 진짜이미지
        
        randnumber = np.random.uniform(-1, 1, size=(128,100))
        fakeimages = generator.predict(randnumber)
        marking_0 = np.zeros((128,1))  # 가짜이미지 라벨 0
        loss2 = discriminator.train_on_batch(fakeimages, marking_0)  # 가짜이미지

        # generator 학습
        noise = np.random.uniform(-1, 1, size=(128,100))
        Y = np.ones((128,1))  
        discriminator.trainable = False # Discriminator 모델 고정
        loss3 =GAN.train_on_batch(noise, Y) 
    print('D_real',loss1, 'D_fake',loss2, 'G',loss3)