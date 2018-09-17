import tensorflow as tf
from tensorflow.python.framework import function

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

@function.Defun(tf.int32, tf.int32)
def MyFunc(x, y):
	
	with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
		add1 = x + y

	return [add1, x - y]


# Building the graph.

a = tf.constant([4], name="x")
b = tf.placeholder(tf.int32, name="MyPlaceHolder")

with tf.device("/job:local/replica:0/task:0/device:CPU:0"):
	add = tf.add(a, b, name="add")

with tf.device("/job:local/replica:0/task:1/device:CPU:0"):
	sub = tf.subtract(a, b, name="sub")

[c,d] = MyFunc(add, sub, name='mycall')

x = tf.add(c, d)

#print(tf.get_default_graph().as_graph_def())

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

with tf.Session("grpc://localhost:2222") as sess:
	print(sess.run([x], feed_dict={b:1}))
writer.close()
