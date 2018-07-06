import tensorflow as tf
from tensorflow.python.framework import function

fib = function.Declare("Fib", [("n", tf.int32)], [("ret", tf.int32)])

@function.Defun(tf.int32, func_name="Fib", out_names=["ret"])
def FibImpl(n):
	return tf.cond(tf.less_equal(n, 1),
		lambda: tf.constant(1),
		lambda: fib(n-1) + fib(n-2))

FibImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
x = fib(n)

res = tf.add(x, 1)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()

#print(tf.get_default_graph().as_graph_def())


writer.close()
print(sess.run(res, feed_dict={n: 5}))

sess.close()
