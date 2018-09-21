import tensorflow as tf
from tensorflow.python.framework import function

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

f = function.Declare("F", [("n", tf.int32)], [("ret", tf.int32)])
g = function.Declare("G", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="F", out_names=["ret"])
def FImpl(n):

	def f1(): 
		with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
			ret = tf.constant(1)
		return ret
	def f2(): 
		with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
			x = n - 1
			ret = g(x)
		return ret

#	with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
	pred = tf.less_equal(n, 1)

	return tf.cond(pred, f1, f2)


@function.Defun(tf.int32, func_name="G", out_names=["ret"])
def GImpl(n):

	with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
		x = n - 1
		ret = f(x)
	return ret


FImpl.add_to_graph(tf.get_default_graph())
GImpl.add_to_graph(tf.get_default_graph())


n = tf.placeholder(tf.int32, name="MyPlaceHolder")
x = f(n)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session("grpc://localhost:2222") as sess:
	print(sess.run([x], feed_dict={n:4}))

writer.close()
