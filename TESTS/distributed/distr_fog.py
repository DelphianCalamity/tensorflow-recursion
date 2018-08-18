import tensorflow as tf
from tensorflow.python.framework import function

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

@function.Defun(tf.float32)
def G(x):
	with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
		add = x + 1
	with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
		ret = x * add 
	return ret

@function.Defun(tf.float32)
def F(x):
	with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
		add = x + 1 
	with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
		ret = x * add
	return ret


a = tf.constant([4.0], name="a")
b = tf.placeholder(tf.float32, name="MyPlaceHolder")

add = tf.add(a, b, name="add")

ret = F(G(add), name='mycall')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session("grpc://localhost:2222") as sess:
	print(sess.run([ret], feed_dict={b:1}))

writer.close()

