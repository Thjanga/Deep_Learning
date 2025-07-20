import tensorflow as tf

# 기본 텐서 만들기
tensor = tf.constant([3,4,5])
tensor2 = tf.constant([6,7,8])
tensor3 = tf.constant([[1,2],[3,4]])
print(tensor + tensor2)
print(tensor.shape)
w = tf.Variable(1.0)
print(w.numpy())
w.assign(2)

# tf.add()
# tf.subtract()
# tf.divide()
# tf.multiply()
# tf.matmul() # 행렬곱
# tf.zeros([2,2]) # 0만 담긴 텐서
# tf.cast

# 텐서가 필요한 이유
# 행렬로 인풋/w값 저장가능, node값 계산식 쉬워짐

