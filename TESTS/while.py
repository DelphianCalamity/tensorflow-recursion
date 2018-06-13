import tensorflow as tf

n = tf.constant(4)
res  = tf.while_loop(lambda i, n: i > 0, lambda i, n: (i-1, n*2), [4, 1])


writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

sess = tf.Session()
result = sess.run([res])
print(result)

writer.close()
sess.close()
