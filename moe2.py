"""
Mixture of Experts
Convolutional Neural Networks
MNIST Classifier
Densely Gated
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_networks import LeNet, FeedForward, linear_layer


class MoE:
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
		
		gate_activations = linear_layer(self.x, self.num_classes * (self.num_experts + 1), relu=False)
		gating_distribution = tf.nn.softmax(tf.reshape(gate_activations, [-1, self.num_experts + 1]))
		
		expert_activations = linear_layer(self.x, self.num_classes * self.num_experts, relu=False)
		expert_distribution = tf.nn.sigmoid(tf.reshape(expert_activations, [-1, self.num_experts]))
		
		final_probabilities = tf.reduce_sum(gating_distribution[:, :self.num_experts] * expert_distribution, 1)
		self.logits = tf.reshape(final_probabilities, [-1, self.num_classes])
		
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
		self.loss = tf.reduce_mean(cross_entropy)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
		
		self.prediction = tf.argmax(self.logits, axis=1)
		correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.predict = tf.nn.softmax(self.logits)
		
	def train(self, epochs):
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
	model = MoE()
	model.train(1000)
