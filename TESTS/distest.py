import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

with tf.device("/job:local/task:1"):
	x = tf.constant(4)
	y2 = x - 2

with tf.device("/job:local/task:0"):
	y1 = x + 2
	y = y1 + y2

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session("grpc://localhost:2222") as sess:
    result = sess.run(y)
    print(result)
writer.close()
