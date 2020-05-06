import tensorflow as tf


def conv2d(inputs, filters, kernel_size=5, strides=1, padding='SAME'):
	channels = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[kernel_size, kernel_size, channels, filters],
		mean=0,
		stddev=0.1
	))
	biases = tf.Variable(tf.zeros([filters]))
	layer = tf.nn.conv2d(
		inputs,
		filter=weights,
		strides=[1, strides, strides, 1],
		padding=padding
	) + biases
	
	return tf.nn.relu(layer)


def maxpool2d(inputs, k=2, padding='SAME'):
	return tf.nn.max_pool2d(
		inputs,
		ksize=[1, k, k, 1],
		strides=[1, k, k, 1],
		padding=padding
	)


def flatten(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	
	return tf.reshape(layer, [-1, num_features])


def linear(inputs, num_outputs, relu=True):
	num_inputs = inputs.get_shape()[-1]
	weights = tf.Variable(tf.random.truncated_normal(
		shape=[num_inputs, num_outputs],
		mean=0,
		stddev=0.1
	))
	biases = tf.Variable(tf.zeros([num_outputs]))
	layer = tf.matmul(inputs, weights) + biases
	
	if relu:
		return tf.nn.relu(layer)
	else:
		return layer


def feedforward(x):
	# Layer 1 = Linear: 784 -> 512
	linear1 = linear(x, num_outputs=512)
	# Layer 2 = Linear: 512 -> 256
	linear2 = linear(linear1, num_outputs=256)
	# Layer 3 = Linear: 256 -> 64
	linear3 = linear(linear2, num_outputs=64)
	# Layer 4 = Linear: 64 -> 10
	logits = linear(linear3, num_outputs=10)
	
	return logits


def lenet(x):
	# Layer 0 = Reshape: 784 -> 28x28@1
	x_img = tf.reshape(x, shape=[-1, 28, 28, 1])
	# Layer 1 = Convolution: 28x28@1 -> 28x28@32 + ReLU
	conv1 = conv2d(x_img, filters=32, kernel_size=5, padding='SAME')
	# Layer 2 = Pooling: 28x28@32 -> 14x14@32
	pool1 = maxpool2d(conv1, padding='SAME')
	# Layer 3 = Convolution: 14x14@32 -> 14x14@64 + ReLU
	conv2 = conv2d(pool1, filters=64, kernel_size=5, padding='SAME')
	# Layer 4 = Pooling: 14x14@64 -> 7x7@64
	pool2 = maxpool2d(conv2, padding='SAME')
	# Layer 5 = Flatten: 7x7@64 -> 3136
	flat = flatten(pool2)
	# Layer 6 = Fully Connected: 3136 -> 512
	fc1 = linear(flat, num_outputs=512)
	# Layer 7 = Fully Connected: 512 -> 128
	fc2 = linear(fc1, num_outputs=128)
	# Layer 8 = Logits: 128 -> 10
	logits = linear(fc2, num_outputs=10, relu=False)
	
	return logits


def alexnet(x):
	# Layer 0 = Reshape: 784 -> 28x28@1
	x_img = tf.reshape(x, shape=[-1, 28, 28, 1])
	# Layer 1 = Convolution: 28x28@1 -> 28x28@32 + ReLU
	conv1 = conv2d(x_img, filters=32, kernel_size=3, padding='SAME')
	# Layer 2 = Pooling: 28x28@32 -> 14x14@32
	pool1 = maxpool2d(conv1, padding='SAME')
	# Layer 3 = Convolution: 14x14@32 -> 14x14@64 + ReLU
	conv2 = conv2d(pool1, filters=64, kernel_size=3, padding='SAME')
	# Layer 4 = Pooling: 14x14@64 -> 7x7@64
	pool2 = maxpool2d(conv2, padding='SAME')
	# Layer 5 = Convolution: 7x7@64 -> 7x7@128 + ReLU
	conv3 = conv2d(pool2, filters=128, kernel_size=3, padding='SAME')
	# Layer 6 = Pooling: 7x7@128 -> 4x4@128
	pool3 = maxpool2d(conv3, padding='SAME')
	# Layer 7 = Flatten: 4x4@128 -> 2048
	flat = flatten(pool3)
	# Layer 8 = Fully Connected: 2048 -> 512
	fc1 = linear(flat, num_outputs=512)
	# Layer 9 = Fully Connected: 1024 -> 128
	fc2 = linear(fc1, num_outputs=128)
	# Layer 10 = Logits: 128 -> 10
	logits = linear(fc2, num_outputs=10, relu=False)
	
	return logits


def vggnet(x):
	# Layer 0 = Reshape: 784 -> 28x28@1
	x_img = tf.reshape(x, shape=[-1, 28, 28, 1])
	# Layer 1 = Convolution: 28x28@1 -> 28x28@32 + ReLU (x2)
	conv1_1 = conv2d(x_img, filters=32, kernel_size=3, padding='SAME')
	conv1_2 = conv2d(conv1_1, filters=32, kernel_size=3, padding='SAME')
	# Layer 2 = Pooling: 28x28@32 -> 14x14@32
	pool1 = maxpool2d(conv1_2, padding='SAME')
	# Layer 3 = Convolution: 14x14@32 -> 14x14@64 + ReLU (x2)
	conv2_1 = conv2d(pool1, filters=64, kernel_size=3, padding='SAME')
	conv2_2 = conv2d(conv2_1, filters=64, kernel_size=3, padding='SAME')
	# Layer 4 = Pooling: 14x14@64 -> 7x7@64
	pool2 = maxpool2d(conv2_2, padding='SAME')
	# Layer 5 = Flatten: 7x7@64 -> 3136
	flat = flatten(pool2)
	# Layer 6 = Fully Connected: 3136 -> 512
	fc1 = linear(flat, num_outputs=512)
	# Layer 7 = Fully Connected: 512 -> 128
	fc2 = linear(fc1, num_outputs=128)
	# Layer 8 = Logits: 128 -> 10
	logits = linear(fc2, num_outputs=10, relu=False)
	
	return logits


def loss(logits, y):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	loss = tf.reduce_mean(cross_entropy)
	
	return loss


def accuracy(logits, y):
	prediction = tf.argmax(logits, axis=1)
	correct_prediction = tf.equal(prediction, tf.argmax(y, axis=1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	return accuracy
