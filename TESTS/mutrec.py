import tensorflow as tf
from tensorflow.python.framework import function

f = function.Declare("F", [("n", tf.int32)], [("ret", tf.int32)])
g = function.Declare("G", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="F", out_names=["ret"])
def FImpl(n):
	return tf.cond(tf.less_equal(n, 1),
		lambda: tf.constant(1),
		lambda: g(n - 1))

@function.Defun(tf.int32, func_name="G", out_names=["ret"])
def GImpl(n):
	return f(n)

# Building the graph.

FImpl.add_to_graph(tf.get_default_graph())
GImpl.add_to_graph(tf.get_default_graph())


n = tf.placeholder(tf.int32, name="MyPlaceHolder")
x = f(n)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session() as sess:			# no need to manually close the session
	print(sess.run([x], feed_dict={n:4}))

writer.close()
