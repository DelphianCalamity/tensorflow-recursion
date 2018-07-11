import tensorflow as tf
from tensorflow.python.framework import function

primes = function.Declare("primes", [("x", tf.int32)], [("ret", tf.int32)])
findPrimePlus  = function.Declare("findPrimePlus",  [("n", tf.int32),("i", tf.int32)], [("ret", tf.int32)])
findPrimeMinus = function.Declare("findPrimeMinus", [("n", tf.int32),("i", tf.int32)], [("ret", tf.int32)])
testPrime      = function.Declare("testPrime",      [("n", tf.int32),("i", tf.int32)], [("ret", tf.bool)])


@function.Defun(tf.int32, func_name="primes", out_names=["ret"])
def PrimesImpl(n):
	return tf.cond(tf.less_equal(n, 0),
        lambda: 2,
		lambda: tf.cond(tf.equal(n, 1),
					lambda: 3,
					lambda: findPrimeMinus(n-2,1)
					))
PrimesImpl.add_to_graph(tf.get_default_graph())

@function.Defun(tf.int32, tf.int32, func_name="findPrimeMinus", out_names=["ret"])
def FindPrimeMinusImpl(n,i):
	return tf.cond(testPrime(6*i-1, 1),
        lambda: tf.cond(tf.equal(n, 0),
		             lambda: 6*i-1,
					 lambda: findPrimePlus(n-1,i)),
		lambda: findPrimePlus(n,i))
FindPrimeMinusImpl.add_to_graph(tf.get_default_graph())

@function.Defun(tf.int32, tf.int32, func_name="findPrimePlus", out_names=["ret"])
def FindPrimePlusImpl(n,i):
	return tf.cond(testPrime(6*i-1, 1),
        lambda: tf.cond(tf.equal(n, 0),
		             lambda: 6*i-1,
					 lambda: findPrimMinus(n-1,i+1)),
		lambda: findPrimeMinus(n,i+1))
FindPrimePlusImpl.add_to_graph(tf.get_default_graph())


@function.Defun(tf.int32, tf.int32, func_name="testPrime", out_names=["ret"])
def TestPrimeImpl(n,i):
	return tf.cond(tf.greater((6*i-1)*(6*i-1), n),
				lambda: True,
				lambda: tf.cond(tf.equal(tf.mod(n, (6*i-1)), 0),
							lambda: False,
							lambda: tf.cond(tf.equal(tf.mod(n, (6*i-1)), 0),
										lambda: False,
										lambda: testPrime(n, i+1))))
TestPrimeImpl.add_to_graph(tf.get_default_graph())

n = tf.placeholder(tf.int32, shape=[])
res = primes(n)

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()

#print(tf.get_default_graph().as_graph_def())

writer.close()
print(sess.run(res, feed_dict={n:7500}))

sess.close()
