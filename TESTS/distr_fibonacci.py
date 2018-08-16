import tensorflow as tf
from tensorflow.python.framework import function

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

fib = function.Declare("Fib", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="Fib", out_names=["ret"])
def FibImpl(n):

	def f1(): 
		with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
			ret = tf.constant(1)
		return ret
	def f2(): 
		with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
			ret = fib(n-1) + fib(n-2)
		return ret

	return tf.cond(tf.less_equal(n, 1), f1, f2)

FibImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
x = fib(n)

res = tf.add(x, 1)

#print(tf.get_default_graph().as_graph_def())

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session("grpc://localhost:2222") as sess:
	print(sess.run(res, feed_dict={n: 20}))

writer.close()
