import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True) 

#  mnist是一个tensorflow内部的变量
sess = tf.InteractiveSession()  # 创建 一个会话

# 权值初始化函数，用截断的正态分布，两倍标准差之外的被截断
def weight_variable(shape)
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
#  偏置初始化函数，偏置初始为0.1
def bias_variable(shape)
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
# 定义卷积方式，步长是1111，padding的SAME是使得特征图与输入图大小一致
def conv2d(x,W)
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding ='SAME')
# 定义池化方式，采用最大池化
def max_pool_2x2(x)    
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],padding='SAME')

# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])    
y_ = tf.placeholder(tf.float32, [None, 10])
# 1D向量（1，784）转2D（28，28）
x_image = tf.reshape(x, [-1,28,28,1])  # -1 表示样本数量不固定

#---------------第14步：定义算法公式-------------------

# 定义 卷积层 conv1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)             
h_pool1 = max_pool_2x2(h_conv1)

# 定义 卷积层 conv2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)             
h_pool2 = max_pool_2x2(h_conv2)

#定义 全连接层 fc1
W_fc1 = weight_variable([7764, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7764])    # 将tensor拉成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 定义Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 定义 Softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#---------------第24步：定义loss和优化器-------------------

# 定义loss 和 参数优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_  tf.log(y_conv), reduction_indices = [1]))  # -sigma y_  log(y)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)                                 # 准确率验证



correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                                        #---------------第34步：训练步骤-------------------# 训练


tf.global_variables_initializer().run()
for i in range(2000)
    batch = mnist.train.next_batch(100)
    if i%100 ==0
        train_accuracy = accuracy.eval(feed_dict= {x batch[0], y_ batch[1], keep_prob1.0})
        print('step %d, training accuracy %g' %(i, train_accuracy))
    train_step.run(feed_dict={x batch[0], y_ batch[1], keep_prob0.5})

#---------------第44步：测试集上评估模型-------------------
# 在验证阶段可能出先一个问题就是GPU内存不够的问题，这里是整个test输入，进行计算
# GPU内存不够大的话，就会出错，分batch的进行


# 这里是输入整个test集的

# print('test accuracy %g ' % accuracy.eval(feed_dict = {
#                                                        x mnist.test.images,
#                                                        y_mnist.test.labels, keep_prob1.0}))


# 这里是分batch验证的

accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
good = 0
total = 0
for i in range(2)
    testSet = mnist.test.next_batch(100)    
    if i ==1  print(testSet[0].shape[0])
    good += accuracy_sum.eval(feed_dict = { x testSet[0], y_ testSet[1], keep_prob 1.0}) 
    total += testSet[0].shape[0]  # testSet[0].shape[0] 是本batch有的样本数量

print(test accuracy %g%(goodtotal))