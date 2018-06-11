import tensorflow as tf
from tensorflow.python.framework import function

@function.Defun(tf.float32, tf.float32)
def MyFunc(x, y):
	return x + y, x - y


# Building the graph.

a = tf.constant([4.0], name="a")
b = tf.placeholder(tf.float32, name="MyPlaceHolder")

add = tf.add(a, b, name="add")
sub = tf.subtract(a, b, name="sub")

c, d = MyFunc(add, sub, name='mycall')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:			# no need to manually close the session
	print(sess.run([add, sub], feed_dict={b:1}))
	print(sess.run([c,d], feed_dict={b:1}))

writer.close()
