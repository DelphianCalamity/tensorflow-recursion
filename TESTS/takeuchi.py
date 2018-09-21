import tensorflow as tf
from tensorflow.python.framework import function

ack = function.Declare("tak", [("x", tf.int32), ("y", tf.int32), ("z", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, tf.int32, tf.int32, func_name="Tak", out_names=["ret"])
def TakImpl(x,y,z):
	return tf.cond(tf.less(y, x),
        lambda: tak(tak(x-1,y,z), tak(y-1,z,x), tak(z-1,x,y))
		lambda: z)

TakImpl.add_to_graph(tf.get_default_graph())

x = tf.placeholder(tf.int32, shape=[])
y = tf.placeholder(tf.int32, shape=[])
z = tf.placeholder(tf.int32, shape=[])
res = tak(x,y,z)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()

#print(tf.get_default_graph().as_graph_def())

writer.close()
print(sess.run(res, feed_dict={x:24, y:16, z:8}))

sess.close()
