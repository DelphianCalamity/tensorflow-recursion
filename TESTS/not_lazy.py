import tensorflow as tf
from tensorflow.python.framework import function

fac = function.Declare("Fac", [("x", tf.int32), ("y", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, tf.int32, func_name="Fac", out_names=["ret"])
def FacImpl(x, y):
	return tf.cond(tf.less_equal(x, 1),
		lambda: tf.constant(1),
		lambda: fac(x-1, fac(x,y)))

FacImpl.add_to_graph(tf.get_default_graph())

x = tf.placeholder(tf.int32, shape=[])
result = fac(x, 2)


y = tf.add(result, 1)

#print(tf.get_default_graph().as_graph_def())

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()
print(sess.run(y, feed_dict={x:2}))

writer.close()

sess.close()
