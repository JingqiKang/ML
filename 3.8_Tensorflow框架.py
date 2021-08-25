"""使用Tensorflow输出一句话"""
# 导入tensorflow工具包并重命名为tf
import tensorflow as tf
# 导入numpy并重命名为np
import numpy as np

tf.compat.v1.disable_eager_execution() # 保证sess.run()能够正常运行
# 初始化一个Tensorflow的常量：Hello Google Tensorflow！
greeting = tf.constant('Hello Google Tensorflow!')
# 启动一个会话
# sess = tf.Session()
sess = tf.compat.v1.Session()
# 使用会话执行greeting计算模块
result = sess.run(greeting)
# 输出会话执行的结果
print(result)
# 显式关闭会话
sess.close()

"""使用Tensorflow完成一次线性函数的计算"""
# 声明matrix1为Tensorflow的一个1*2的行向量。
matrix1 = tf.constant([[3., 3.]])
# 声明matrix2为Tensorflow的一个2*1的列向量。
matrix2 = tf.constant([[2.], [2.]])

# product将上述两个算子相乘，作为新算例。
product = tf.matmul(matrix1, matrix2)

# 继续将product与一个标量2.0求和拼接，作为最终的linear算例。
linear = tf.add(product, tf.constant(2.0))

# 直接在会话中执行linear算例，相当于将上面所有的单独算例拼接成流程图来执行。
with tf.compat.v1.Session() as sess:
    result = sess.run(linear)
    print(result)