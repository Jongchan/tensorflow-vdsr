# tensorflow-vdsr

## Overview
This is a Tensorflow implementation for ["Accurate Image Super-Resolution Using Very Deep Convolutional Networks", CVPR 16'](http://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf).
- [The author's project page](http://cv.snu.ac.kr/research/VDSR/)
- To download the required data for training/testing, please refer to the README.md at data directory.

## Files
- VDSR.py : main training file.
- MODEL.py : model definition.
- MODEL_FACTORIZED.py : model definition for Factorized CNN. (not recommended to use. for record purpose only)
- PSNR.py : define how to calculate PSNR in python
- TEST.py : test all the saved checkpoints
- PLOT.py : plot the test result from TEST.py

## How To Use
### Training
```shell
# if start from scratch
python VDSR.py
# if start with a checkpoint
python VDSR.py --model_path ./checkpoints/CHECKPOINT_NAME.ckpt
```
### Testing
```shell
# this will test all the checkpoint in ./checkpoint directory.
# and save the results in ./psnr directory
python TEST.py
```
### Plot Result
```shell
# plot the psnr result stored in ./psnr directory
python PLOT.py
```

## Result
The checkpoint is file is [here](https://drive.google.com/file/d/0B4KsMpU0BeosbDB2NllZZkdvY1U/view?usp=sharing)
##### Results on Set 5

|  Scale    | Bicubic | VDSR | tf_VDSR |
|:---------:|:-------:|:----:|:-------:|
| **2x** - PSNR/SSIM|   33.66/0.9929	|   37.53/0.9587	|  37.24 |
| **3x** - PSNR/SSIM|   30.39/0.8682	|   33.66/0.9213	|  33.37 |
| **4x** - PSNR/SSIM|   28.42/0.8104	|   31.35/0.8838	|  31.09 |

##### Results on Set 14

|  Scale    | Bicubic | VDSR | tf_VDSR |
|:---------:|:-------:|:----:|:-------:|
| **2x** - PSNR/SSIM|   30.24/0.8688	|   33.03/0.9124	| 32.80 |
| **3x** - PSNR/SSIM|   27.55/0.7742	|   29.77/0.8314	| 29.67 |
| **4x** - PSNR/SSIM|   26.00/0.7027	|   28.01/0.7674	| 27.87 |

## Remarks
- The training is further accelerated with asynchronous data fetch.
- Tried to accelerate the network with the idea from [Factorized CNN](https://128.84.21.199/pdf/1608.04337v1.pdf). It is possible to implement with `tf.nn.depthwise_conv2d` and 1x1 convolution, but not so effective.
- Thanks to @harungunaydin 's comment, **AdamOptimizer** gives a much more stable training. There's an option added.
