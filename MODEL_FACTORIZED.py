import tensorflow as tf
import numpy as np

def model_factorized(input_tensor):
	with tf.device("/gpu:0"):
		weights = []
		tensor = None

		conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/32)))
		conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

		depth = 50
		for i in range(depth-1):
			depthwise_filter = tf.get_variable("depth_conv_%02d_w" % (i+1), [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/32)))
			pointwise_filter = tf.get_variable("point_conv_%02d_w" % (i+1), [1,1,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/1/128)))
			conv_b = tf.get_variable("conv_%02d_b" % (i+1), [64], initializer=tf.constant_initializer(0))
			weights.append(depthwise_filter)
			weights.append(pointwise_filter)
			weights.append(conv_b)
			conv_tensor = tf.nn.bias_add(tf.nn.separable_conv2d(tensor, depthwise_filter, pointwise_filter, [1,1,1,1], padding='SAME'), conv_b)
			"""
			conv_tensor = tf.nn.relu(tf.nn.depthwise_conv2d(tensor, depthwise_filter, [1,1,1,1], padding='SAME'))
			conv_tensor = tf.nn.bias_add(tf.nn.conv2d(conv_tensor, pointwise_filter, [1,1,1,1], padding='VALID'), conv_b)
			"""
			tensor = tf.nn.relu(tf.add(tensor, conv_tensor))
			
		
		conv_w = tf.get_variable("conv_%02d_w"%depth, [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
		conv_b = tf.get_variable("conv_%02d_b"%depth, [1], initializer=tf.constant_initializer(0))
		weights.append(conv_w)
		weights.append(conv_b)
		tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)

		tensor = tf.add(tensor, input_tensor)
		return tensor, weights
