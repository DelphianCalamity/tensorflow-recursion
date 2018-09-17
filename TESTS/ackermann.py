import tensorflow as tf
from tensorflow.python.framework import function

ack = function.Declare("Ack", [("m", tf.int32), ("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, tf.int32, func_name="Ack", out_names=["ret"])
def AckImpl(m,n):

	def f1(): 
		ret = n + 1
		return ret

	def f2():
		def ff1():
			r = ack(m-1,1)
			return r

		def ff2():
			r = ack(m-1, ack(m, n-1))
			return r

		ret = tf.cond(tf.equal(n, 0), ff1, ff2)
		return ret

	return tf.cond(tf.equal(m, 0), f1, f2)


AckImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
m = tf.placeholder(tf.int32, shape=[])
res = ack(m,n)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()

#print(tf.get_default_graph().as_graph_def())

print(sess.run(res, feed_dict={m:2, n:3}))

sess.close()

writer.close()
