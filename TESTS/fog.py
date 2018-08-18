import tensorflow as tf
from tensorflow.python.framework import function

@function.Defun(tf.float32)
def G(x):
	return x * x


@function.Defun(tf.float32)
def F(x):
	return x + x


a = tf.constant([4.0], name="a")
b = tf.placeholder(tf.float32, name="MyPlaceHolder")

add = tf.add(a, b, name="add")

ret = F(G(add), name='mycall')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:
	print(sess.run([ret], feed_dict={b:1}))

writer.close()
