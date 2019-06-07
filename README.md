# Digit-Recognition-MNIST-SHM

## Implementing CNN for Digit Recognition (MNIST and SVHN dataset) using PyTorch C++ API 

**Branch - Version 1**

	1. Using MNIST dataset currently.
	2. Installing PyTorch C++ API.
	3. Training using PyTorch C++ API.

**Note**: There may be C10:errors either because of:

1. Incorrect data path. Check your data directory of MNIST dataset.
2. Make sure while using `cmake`: `cmake -DCMAKE_PREFIX_PATH=/absolute_path_to_libtorch/` - it should be absolute path to `libtorch.`

## Process

Note: Prefer using stable version of libtorch. This code doesn't use GPU, as I don't have the resources to test it right now.
1. `mkdir build`
2. `cd build`
3. cmake: `cmake -DCMAKE_PREFIX_PATH=/home/user/path/to/libtorch/ ..`
4. `make`
5. `./example`
