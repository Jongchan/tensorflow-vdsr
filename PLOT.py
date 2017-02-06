import matplotlib.pyplot as plt
import pickle, glob
import numpy as np
import sys
psnr_prefix = './psnr/*'
psnr_paths = sorted(glob.glob(psnr_prefix))

psnr_means = {}

def filter_by_scale(row, scale):
	return row[-1]==scale

for i, psnr_path in enumerate(psnr_paths):
	print ""
	print psnr_path
	psnr_dict = None
	epoch = str(i)#psnr_path.split("_")[-1]
	with open(psnr_path, 'rb') as f:
		psnr_dict = pickle.load(f)
	dataset_keys = psnr_dict.keys()
	for j, key in enumerate(dataset_keys):
		print 'dataset', key
		psnr_list = psnr_dict[key]
		psnr_np = np.array(psnr_list)

		psnr_np_2 = psnr_np[np.array([filter_by_scale(row,2) for row in psnr_np])]
		psnr_np_3 = psnr_np[np.array([filter_by_scale(row,3) for row in psnr_np])]
		psnr_np_4 = psnr_np[np.array([filter_by_scale(row,4) for row in psnr_np])]
		print "x2:",np.mean(psnr_np_2, axis=0).tolist()
		print "x3:",np.mean(psnr_np_3, axis=0).tolist()
		print "x4:",np.mean(psnr_np_4, axis=0).tolist()

		mean_2 = np.mean(psnr_np_2, axis=0).tolist()
		mean_3 = np.mean(psnr_np_3, axis=0).tolist()
		mean_4 = np.mean(psnr_np_4, axis=0).tolist()
		psnr_mean = [mean_2, mean_3, mean_4]
		#print 'psnr mean', psnr_mean
		if psnr_means.has_key(key):
			psnr_means[key][epoch] = psnr_mean
		else:
			psnr_means[key] = {epoch: psnr_mean}

#sys.exit(1)

keys = psnr_means.keys()
for i, key in enumerate(keys):
	psnr_dict = psnr_means[key]
	epochs = sorted(psnr_dict.keys())
	x_axis = []
	bicub_mean = []
	vdsr_mean_2 = []
	vdsr_mean_3 = []
	vdsr_mean_4 = []

	for epoch in epochs:
		print epoch
		print psnr_dict[epoch]
		x_axis.append(int(epoch))
		bicub_mean.append(psnr_dict[epoch][0][0])
		vdsr_mean_2.append(psnr_dict[epoch][0][1])
		vdsr_mean_3.append(psnr_dict[epoch][1][1])
		vdsr_mean_4.append(psnr_dict[epoch][2][1])
	plt.figure(i)
	print key
	print len(x_axis), len(bicub_mean), len(vdsr_mean_2)
	print vdsr_mean_2
	print "x2", np.argmax(vdsr_mean_2), np.max(vdsr_mean_2)
	print "x3", np.argmax(vdsr_mean_3), np.max(vdsr_mean_3)
	print "x4", np.argmax(vdsr_mean_4), np.max(vdsr_mean_4)
	lines_bicub = plt.plot(vdsr_mean_2, 'g')
	lines_bicub = plt.plot(vdsr_mean_4, 'b', vdsr_mean_3, 'y')
	plt.setp(lines_bicub, linewidth=3.0)
	#plt.show()

"""
psnr_means :
	{
		'DATASET_NAME' :
			{
				'EPOCH' : [bicubic psnr, vdsr psnr]
			}
		'DATASET_NAME_2':
			{
				'EPOCH' : [bicubic psnr, vdsr psnr]
			}
		...
	}
"""
for i, psnr_path in enumerate(psnr_paths):
	print i, psnr_path
