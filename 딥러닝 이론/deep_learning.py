import tensorflow as tf

키 = [150,160,170,180]
신발 = [152,162,172,182]

a = tf.Variable(0.1)
b = tf.Variable(0.5)

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

for i in range(1000):
    with tf.GradientTape() as tape:
        예측신발 = 키 * a + b
        loss = (예측신발-신발)**2
        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(loss,[a,b])
    # print(gradient)
    opt.apply_gradients([(gradients[0], a), (gradients[1], b)])   
    # a.assign_sub(gradient[0]*0.00001)
    # b.assign_sub(gradient[1]*0.00001)
    print(a.numpy(),b.numpy())

