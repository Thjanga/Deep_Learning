import tensorflow as tf
import numpy as np

Pmodel = tf.keras.models.load_model(r'Project 3,4\model1.keras')

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

# print(number_text)

first_input = number_text[117:117+25]
first_input = tf.one_hot(first_input,31)
first_input = tf.expand_dims(first_input,axis=0) # 3차원으로 바꿔줌

# print(first_input)

music = []

for i in range(200):
    pre = Pmodel.predict(first_input)
    pre = np.argmax(pre[0]) # 최대값

    # 1. 첫입력값 만들기
    # 2. predict로 다음문자 예측
    # 3. 예측한 다음문자 [] 저장하기
    # 4. 첫입력값 앞에 짜르기
    # 5. 예측한 다음문자를 뒤에 넣기
    # 6. 새로운 입력값을 predict

    music.append(pre)

    next_input = first_input.numpy()[0][1:]

    one_hot_num = tf.one_hot(pre,31)

    first_input = np.vstack([next_input, one_hot_num.numpy()])

    first_input = tf.expand_dims(first_input,axis=0)

# print(music)

music_text = []

for i in music:
    music_text.append(num_to_text[i])

print(''.join(music_text))

