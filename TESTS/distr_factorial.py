import tensorflow as tf
from tensorflow.python.framework import function

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

fac = function.Declare("Fac", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="Fac", out_names=["ret"])
def FacImpl(n):

	def f1(): 
		with tf.device("/job:local/task:0"):
			ret = tf.constant(1)
		return ret
	def f2(): 
		with tf.device("/job:local/task:1"):
			ret = n * fac(n - 1)
		return ret

	return tf.cond(tf.less_equal(n, 1), f1, f2)

FacImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
x = tf.add(n, 1)
result = fac(x)
y = tf.add(result, 1)

print(tf.get_default_graph().as_graph_def())

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session("grpc://localhost:2222") as sess:
	print(sess.run(y, feed_dict={n: 0}))

writer.close()

