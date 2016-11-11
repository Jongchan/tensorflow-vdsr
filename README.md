# tensorflow-vdsr

## Overview
This is a Tensorflow implementation for ["Accurate Image Super-Resolution Using Very Deep Convolutional Networks", CVPR 16'](http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf).
- [The author's project page](http://cv.snu.ac.kr/research/VDSR/)
To download the required data for training/testing, please refer to the README.md at data directory.\

## Files
- VDSR.py : main training file.
- MODEL.py : model definition.
- MODEL_FACTORIZED.py : model definition for Factorized CNN. (not recommended to use. for record purpose only)
- PSNR.py : define how to calculate PSNR in python
- TEST.py : test all the saved checkpoints
- PLOT.py : plot the test result from TEST.py

## How To Use
### Training
```python
# if start from scratch
python VDSR.py
# if start with a checkpoint
python VDSR.py --model_path ./checkpoints/CHECKPOINT_NAME.ckpt
```
### Testing
```python
# this will test all the checkpoint in ./checkpoint directory.
# and save the results in ./psnr directory
python TEST.py
```
### Plot Result
```python
# plot the psnr result stored in ./psnr directory
python PLOT.py
```

## Result

## Remarks
- The training is further accelerated with asynchronous data fetch.
- Tried to accelerate the network with the idea from [Factorized CNN](https://128.84.21.199/pdf/1608.04337v1.pdf). It is possible to implement with `tf.nn.depthwise_conv2d` and 1x1 convolution, but not so effective.
