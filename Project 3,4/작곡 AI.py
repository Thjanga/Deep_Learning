import tensorflow as tf

text = open(r'Project 3,4\pianoabc.txt','r').read() # 악보 데이터를 문자나 숫자로 치환해야함 > abc notation

unique_text = list(set(text)) # 단어주머니
unique_text.sort()

# 문자를 숫자로 변환
text_to_num = {}
num_to_text = {}

for i,data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

number_text = []

for i in text:
    number_text.append(text_to_num[i])

trainX = []
trainY = []

for i in range(len(number_text)-25):
    trainX.append(number_text[0+i:25+i])
    trainY.append(number_text[25+i])

trainX = tf.one_hot(trainX,31)
trainY = tf.one_hot(trainY,31)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100,input_shape=(25,31)),
    tf.keras.layers.Dense(31, activation='softmax')
])

# 카테고리 예측문제
# categorical_crossentropy 와 softmax 세트
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(trainX,trainY,batch_size=64,epochs=60,verbose=2) # 64가 데이터를 학습한 후 w값 업데이트

model.save('Project 3,4/model1.keras')
