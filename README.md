# GPR - Motion Modelling 
This repository is a fork of Christoph Jud's *Basic Gaussian process regression library* implementation in C++.

The code was amended specifically for ultrasound-based respiratory motion modelling using:
* Principal Component Analysis (PCA) for dimensionality reduction, and
* a linear autoregressive (AR) model for prediction the respiratory state ahead of time.

**Usage:**
1. Use *GaussianProcessLearn.cpp* for training
2. Use *GaussianProcessPredict.cpp* for prediction

## Requirements:
* General reqiuirements are listed below
* Python requirements are listed in scripts/requirements.txt

## References
This code has been used for publication in:
```
@article{10.1088/1361-6560/abaa26,
	author={Alina Tamara Giger and Miriam Krieger and Christoph Jud and Alisha Duetschler and Rares Salomir and Oliver Bieri and Grzegorz Bauman and Damien Nguyen and Damien Charles Weber and Antony John Lomax and Ye Zhang and Philippe C Cattin},
	title={Liver-ultrasound based motion modelling to estimate 4D dose distributions for lung tumours in scanned proton therapy},
	journal={Physics in Medicine & Biology},
	url={http://iopscience.iop.org/article/10.1088/1361-6560/abaa26},
	year={2020}
}
```
# GPR - Basic Gaussian Process Library

Basic Gaussian process regression library. (Eigen3 required)

## Features
	* Multivariate Gaussian process regression
	* Calculation of the derivative at a point
	* Calculation of the uncertainty at a point
	* Save and Load the Gaussian process to/from files
	* Kernels: White, Gaussian, Periodic, RationalQuadratic, Sum and Product
	* Derivative of the kernels
	* Likelihood functions: Gaussian Log Likelihood (incl. derivative wrt. kernel parameter)
	* Prior distributions: Gaussian, Inverse Gaussian, Gamma (incl. sampling, cdf and inverse cdf)
	* Prior distributions can be built by providing their mode and variance



## Getting Started
	To setup the library clone the git repository first
	```
	git clone https://github.com/ChristophJud/GPR.git
	```

	The building of GPR is based on [cmake](http://www.cmake.org/). So navigate to the main directory GPR and create a build directory.
	```
	mkdir build	# create a build directory
	cd build
	ccmake ..	# ccmake is an easy tool to set config parameters
	```
	Set the build type and the installation directory and
	```
	CMAKE_CXX_FLAGS		-std=c++11
	```

	Since GPR depends on the matrix library [Eigen](http://eigen.tuxfamily.org) provide its include directory
	```
	EIGEN3_INCLUDE_DIR	/path/to/eigen/eigen-3.2.4/install/include/eigen3
	```

	If not all required [Boost](http://www.boost.org) libraries are found on the system provide a custom installation
	```
	Boost_INCLUDE_DIR 	/path/to/boost/boost_1_57_0/
	Boost_LIBRARY_DIR	/path/to/boost/boost_1_57_0/stage/lib/

	```
	Make sure that boost has been built with C++11 by adding ```cxxflags="-std=c++11"``` to the ```b2``` command.

	Finally, type
	```
	make install -j8
	```
	and the library including all test programs will be built.

### Include the library in your own cmake project
	If you want to include the library into your own project the straight forward way is the following:
	Add 
	```
FIND_PACKAGE(GPR REQUIRED)
	``` 
	to your CMakeLists.txt file and provide
	```
	GPR_DIR			/path/to/main/gpr/project/dir/cmake 
	```
	In the CMakeLists.txt you can link your program with ```${GPR_LIBRARIES}```.

## Examples
	The tests can be seen as good examples to how to use the library. 

## TODOs
	* Matrix valued kernels
	* Store/load into/from hdf5 files

## Issues

## References
	A thorough introduction can be found in the open book of C.E. Rasmussen: Rasmussen, Carl Edward. [Gaussian processes for machine learning.](http://www.gaussianprocess.org/gpml/) (2006).

## License
	GPR itself is licensed under the Apache 2.0 license. It depends, however, on other open source projects, which are distributed under different licenses. These are [Eigen](http://eigen.tuxfamily.org), [LAPACK](http://www.netlib.org/lapack/) and [Boost](http://www.boost.org).
