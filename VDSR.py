import tensorflow as tf
from PIL import Image
import numpy as np
import os, glob,re,signal
from random import shuffle
import scipy.io
from MODEL import model
from MODEL_FACTORIZED import model_factorized
DATA_PATH = "/home/jongchan/Projects/SRCNN/VDSR_SNU/aug_more/"
IMG_SIZE = (41, 41)
BATCH_SIZE = 64
BASE_LR = 0.1
LR_RATE = 0.1
LR_STEP_SIZE = 20 #epoch
MAX_EPOCH = 120

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path
from PSNR import psnr

TEST_DATA_PATH = "/home/jongchan/Projects/SRCNN/VDSR_SNU/gt_y/"
from TEST import test_VDSR

def get_train_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	print len(l)
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	print len(l)
	train_list = []
	for f in l:
		if os.path.exists(f):
			if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
			if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
			if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])
	return train_list
"""
print len(get_train_list())
print get_train_list()[10]
import sys
sys.exit(1)
"""
def get_image_batch(train_list,offset,batch_size):
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	cbcr_list = []
	for pair in target_list:
		"""
		input_img 	= np.ndarray((IMG_SIZE[1],IMG_SIZE[0],3), 'u1', Image.open(pair[1]).convert("YCbCr").tostring()).astype('float')
		gt_img 		= np.ndarray((IMG_SIZE[1],IMG_SIZE[0],3), 'u1', Image.open(pair[0]).convert("YCbCr").tostring()).astype('float')
		"""
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
		#cbcr_list.append([input_img[:,:,1:], gt_img[:,:,1:]])
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	return input_list, gt_list, np.array(cbcr_list)
import sys

def get_test_image(test_list, offset, batch_size):
	target_list = test_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	for pair in target_list:
		mat_dict = scipy.io.loadmat(pair[1])
		input_img = None
		if mat_dict.has_key("img_2"): 	input_img = mat_dict["img_2"]
		elif mat_dict.has_key("img_3"): input_img = mat_dict["img_3"]
		elif mat_dict.has_key("img_4"): input_img = mat_dict["img_4"]
		else: continue
		gt_img = scipy.io.loadmat(pair[0])['img_raw']
		input_list.append(input_img[:,:,0])
		gt_list.append(gt_img[:,:,0])
	return input_list, gt_list
if __name__ == '__main__':
	train_list = get_train_list(DATA_PATH)

	train_input  	= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
	train_gt  		= tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
	#train_output, weights 	= model_factorized(train_input)
	train_output, weights 	= model(train_input)
	tf.get_variable_scope().reuse_variables()
	#loss = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(train_output, train_gt), 2)))
	loss = tf.reduce_sum(tf.nn.l2_loss(tf.sub(train_output, train_gt)))
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4

	global_step 	= tf.Variable(0, trainable=False)
	learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)

	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
	"""
	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_value(grad, tf.div(-0.1*BASE_LR, learning_rate), tf.div(0.1*BASE_LR,learning_rate)), var) for grad, var in gvs]
	#capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
	opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)
	"""
	"""
	gvs = optimizer.compute_gradients(loss)
	capped_gvs = [(tf.clip_by_norm(grad, 0.1), var) for grad, var in gvs]
	opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)
	"""

	tvars = tf.trainable_variables()
	gvs = zip(tf.gradients(loss,tvars), tvars)
	norm = 0.1*BASE_LR
	capped_gvs = [(tf.clip_by_norm(grad, 0.01), var) for grad, var in gvs]
	opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)
	"""
	tvars = tf.trainable_variables()
	grads,_ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 0.1)
	opt = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
	"""

	saver = tf.train.Saver(weights, max_to_keep=0)

	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		def signal_handler(signum,frame):
			print "stop training, save checkpoint..."
			saver.save(sess, "./checkpoints/VDSR_const_clip_epoch_%03d.ckpt" % epoch ,global_step=global_step)
			print "Done"
			sys.exit(1)
		original_sigint = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGINT, signal_handler)

		if model_path:
			print "restore model..."
			saver.restore(sess, model_path)
			print "Done"


		for epoch in xrange(0, MAX_EPOCH):
			shuffle(train_list)

			for step in range(len(train_list)//BATCH_SIZE):
				offset = step*BATCH_SIZE
				input_data, gt_data, cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE)
				feed_dict = {train_input: input_data, train_gt: gt_data}
				_,l,output,lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step], feed_dict=feed_dict)
				norm = 0.1*BASE_LR / lr
				print "[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr)
				del input_data, gt_data, cbcr_data

			saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)

