import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# 3점은 없음
raw = pd.read_table(r'Project 3,4\naver_shopping.txt',names=['rating','review'])

# 3점보다 높으면 1, 아니면 0
raw['label'] = np.where(raw['rating'] > 3, 1, 0)

# 데이터 전처리
raw['review'] = raw['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]','')

# raw.isnull().sum()

raw.drop_duplicates(subset=['review'],inplace=True)

# bag of words
unique = raw['review'].tolist()
unique = ''.join(unique)
unique = list(set(unique))
unique.sort()

tokenizer = Tokenizer(char_level=True,oov_token='<OOV>')
strlist = raw['review'].tolist()
tokenizer.fit_on_texts(strlist)

train_seq = tokenizer.texts_to_sequences(strlist)

Y = raw['label'].values

# 최대 글자수
raw['length'] = raw['review'].str.len()

print(raw.head())
print(raw.describe())

# 100자로 제한
raw['length'][raw['length'] < 100].count()
X = pad_sequences(train_seq, maxlen=100)

trainX, valX, trainY, valY = train_test_split(X,Y,test_size=0.2,random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1,16),
    tf.keras.layers.LSTM(64),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1,activation='sigmoid')

])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=10,validation_data=(valX,valY),batch_size=64)