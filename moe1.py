"""
Mixture of Experts
Feedforward Neural Networks
MNIST Classifier
Densely Gated
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_networks import LeNet, FeedForward


class MOE:
	def __init__(self, num_experts=5):
		self.sess = tf.Session()
		
		self.learning_rate = 0.001
		self.batch_size = 64
		
		self.num_inputs = 784
		self.num_classes = 10
		self.num_experts = num_experts
		
		self.load_data()
		self.build()
	
	def load_data(self):
		mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
		self.training_data = mnist.train
		self.X_valid = mnist.validation.images
		self.y_valid = mnist.validation.labels
		self.X_test = mnist.test.images
		self.y_test = mnist.test.labels
	
	def build(self):
		with tf.variable_scope('ensemble', reuse=tf.AUTO_REUSE) as scope:
			self.x = tf.placeholder(tf.float32, [None, self.num_inputs], name='x')
			self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')
		
		self.networks = [LeNet() for _ in range(self.num_experts)]
		experts = tf.stack([net.logits for net in self.networks], axis=-1)
		
		w_g = tf.Variable(tf.zeros([self.num_inputs, self.num_experts]))
		gates = tf.nn.softmax(tf.matmul(self.x, w_g))
		gates = tf.expand_dims(gates, axis=1)
		
		self.logits = tf.reduce_sum(tf.multiply(experts, gates), axis=-1)
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
		
		self.prediction = tf.argmax(self.logits, axis=1)
		correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.predict = tf.nn.softmax(self.logits)
		
	def train(self, epochs=500):
		self.sess.run(tf.global_variables_initializer())
		
		for e in range(epochs):
			x_batch, y_batch = self.training_data.next_batch(self.batch_size)
			feed_dict = {self.x: x_batch, self.y: y_batch}
			self.sess.run(self.optimizer, feed_dict=feed_dict)
			train_loss, train_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			
			feed_dict = {self.x: self.X_valid, self.y: self.y_valid}
			valid_loss, valid_acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
			
			print(f'epoch {e + 1}:',
				f'train loss = {train_loss:.4f},',
				f'train acc = {train_acc:.4f},',
				f'valid loss = {valid_loss:.4f},',
				f'valid acc = {valid_acc:.4f}'
			)
		print('training complete')
		
		feed_dict = {self.x: self.X_test, self.y: self.y_test}
		loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
		print(f'test loss = {loss:.4f}, test acc = {acc:.4f}')


if __name__ == '__main__':
	model = MOE()
	model.train(20)
