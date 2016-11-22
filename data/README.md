# data generation
0. Download train/test data from [original author's project page](http://cv.snu.ac.kr/research/VDSR/)
1. Download and unzip 291 dataset, and set the proper directory in 'aug_train.m'.
2. Download and unzip other test dataset (Set5, Set14, B100, Urban100), and set the proper directory in 'aug_test.m'.
3. run 'aug_train.m' and 'aug_test.m' matlab code for patch/data generation


- Please note that data are generated/manipulated in Matlab, for a good bicubic interpolation and reproducibility. (OpenCV2 interpolation is strange)
- Too much data will make the network diverge. I'm currently using patches with original/rotate90/original flipped/rotate90 flipped, and you can find the data [here](https://drive.google.com/file/d/0B4KsMpU0Beosc1FNQVlFZWlMOG8/view?usp=sharing)
