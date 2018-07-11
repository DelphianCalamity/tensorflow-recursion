import tensorflow as tf
from tensorflow.python.framework import function

ack = function.Declare("ack", [("n", tf.int32), ("m", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, tf.int32, func_name="Ack", out_names=["ret"])
def AckImpl(n,m):
	return tf.cond(tf.equal(m, 0),
        lambda: n + 1,
        tf.cond(tf.equals(n, 0),
            lambda: ack(m-1,1),
            lambda: ack(m-1,ack(m,n-1))))

AckImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
m = tf.placeholder(tf.int32, shape=[])
res = ack(n,m)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()

#print(tf.get_default_graph().as_graph_def())

writer.close()
print(sess.run(res, feed_dict={n:2, m:3}))

sess.close()
