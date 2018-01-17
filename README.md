# Code for the paper "Accurate Optical Flow via Direct Cost Volume Processing. Jia Xu, René Ranftl, and Vladlen Koltun. CVPR 2017"

If you use this code or the provided models in your research, please cite the following paper:

	@inproceedings{XRK2017,
		author    = {Jia Xu and Ren\'e Ranftl and Vladlen Koltun},
		title     = {{Accurate Optical Flow via Direct Cost Volume Processing}},
		booktitle = {CVPR},
		year      = {2017},
	}

## Dependencies

- CMake 3.2
- Caffe + MatCaffe (needs to be in Matlab path, see ``matlab/demo.m``)
	- We have seen issues with the latest version of caffe. We recommend to use this brunch (training and testing): https://github.com/Wangyida/caffe/tree/cnn_triplet
- OpenCL

## Setup

- Set path to OpenCL SDK:
    - For Intel OpenCL  set `export INTELOCLSDKROOT=<path to intel ocl sdk>`, e.g., `export INTELOCLSDKROOT=/usr/local/intel/opencl`
    - For NVIDIA OpenCL set `export CUDA_PATH=<path to cuda home>`, e.g., `export CUDA_PATH=/usr/local/cuda`
    - For AMD OpenCL set `export AMDAPPSDKROOD=<path to amd ocl sdk>`, e.g., `export AMDAPPSDKROOD=/usr/local/amd/opencl`
- Set ``MATLAB_ROOT`` environment variable, e.g., `export MATLAB_ROOT=/usr/local/MATLAB/R2017a`
- ``mkdir build``
- ``cd build``
- ``cmake ..``
- ``make``
- ``make install``

## Running the code:
See ``matlab/demo.m``

## Log
- Version 0.1, 2017-07-20

 	Includes feature embedding code/models, 4-D cost volume construction and processing, and forward-backward consistency checking. Part of poster-processing (EpicFlow inpainting, homography fitting) can not be included due to license issues. We expect to release them in future versions. You may download the EpicFlow code (or other inpainting code), and replace the match file with our output to obtain a dense optical flow filed.
