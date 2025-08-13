import pandas as pd
import tensorflow as tf
import numpy as np

# 데이터 불러오기
data = pd.read_csv(r'Project 5/train.csv')
print(data.isnull().sum())  # 결측치 확인

# 결측치 처리
data['Age'] = data['Age'].fillna(value=30)  # Age의 결측치를 30으로 대체
data['Embarked'] = data['Embarked'].fillna(value='S')  # Embarked의 결측치를 'S'로 대체

y = data.pop('Survived')  # 타겟값 분리

# 수치형 특성 정규화 레이어 생성 및 적합
Farelayer = tf.keras.layers.Normalization(axis=None)
Farelayer.adapt(np.array(data['Fare']))
print(Farelayer(np.array(data['Fare'])))

Siblayer = tf.keras.layers.Normalization(axis=None)
Siblayer.adapt(np.array(data['SibSp']))

Parchlayer = tf.keras.layers.Normalization(axis=None)
Parchlayer.adapt(np.array(data['Parch']))

Pclasslayer = tf.keras.layers.Normalization(axis=None)
Pclasslayer.adapt(np.array(data['Pclass']))

# Age를 구간화(분할)하는 레이어 생성
Agelayer = tf.keras.layers.Discretization(bin_boundaries=[10,20,30,40,50,60])

# 범주형 특성 원-핫 인코딩 레이어 생성 및 적합
Sexlayer = tf.keras.layers.StringLookup(output_mode='one_hot')
Sexlayer.adapt(np.array(data['Sex']))

Embarkedlayer = tf.keras.layers.StringLookup(output_mode='one_hot')
Embarkedlayer.adapt(np.array(data['Embarked']))

# Ticket 특성 인덱스 변환 레이어 및 임베딩 레이어 생성
Ticketlayer = tf.keras.layers.StringLookup()
Ticketlayer.adapt(np.array(data['Ticket']))

unique = len(data['Ticket'].unique())  # 고유 티켓 개수
TicketEBlayer = tf.keras.layers.Embedding(unique+1,9)  # 임베딩 레이어

# 입력 레이어 정의
Input_fare = tf.keras.Input(shape=(1,),name='Fare')
Input_Parch = tf.keras.Input(shape=(1,),name='Parch')
Input_SibSp = tf.keras.Input(shape=(1,),name='SibSp')
Input_pclass = tf.keras.Input(shape=(1,),name='Pclass')
Input_age = tf.keras.Input(shape=(1,),name='Age')
Input_sex = tf.keras.Input(shape=(1,),name='Sex',dtype=tf.string)
Input_embarked = tf.keras.Input(shape=(1,),name='Embarked',dtype=tf.string)
Input_ticket = tf.keras.Input(shape=(1,),name='Ticket',dtype=tf.string)

# 각 입력에 전처리 레이어 적용
x_fare = Farelayer(Input_fare)
x_Parch = Parchlayer(Input_Parch)
x_SibSp = Siblayer(Input_SibSp)
x_pclass = Pclasslayer(Input_pclass)
x_age = Agelayer(Input_age)
x_sex = Sexlayer(Input_sex)
x_embarked = Embarkedlayer(Input_embarked)

# Ticket 특성 임베딩 처리
x_ticket1 = Ticketlayer(Input_ticket)
x_ticket2 = TicketEBlayer(x_ticket1)
x_ticket2 = tf.keras.layers.Flatten()(x_ticket2)

# 모든 특성 합치기
concat_layer = tf.keras.layers.concatenate([
    x_fare, x_Parch, x_SibSp, x_pclass, x_age, x_sex, x_embarked, x_ticket2
    ])

# 완전연결 신경망 구성
x = tf.keras.layers.Dense(128, activation='relu')(concat_layer)
x = tf.keras.layers.Dense(64, activation='relu')(x)
lastlayer = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # 출력층 (이진 분류)

# 모델 정의
model = tf.keras.Model(inputs=[ 
    Input_fare, Input_Parch, Input_SibSp, Input_pclass, Input_age, Input_sex, Input_embarked, Input_ticket
], outputs=lastlayer)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 입력 데이터 딕셔너리 생성
x = {
    'Fare': np.array(data['Fare']),
    'Age': np.array(data['Age']),
    'Parch': np.array(data['Parch']),
    'SibSp': np.array(data['SibSp']),
    'Pclass': np.array(data['Pclass']),
    'Sex': np.array(data['Sex']),
    'Embarked': np.array(data['Embarked']),
    'Ticket': np.array(data['Ticket'])
}

# 모델 학습
model.fit(x, y, epochs=15, validation_split=0.1)