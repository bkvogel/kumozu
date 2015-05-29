# Kumozu
Kumozu is research software for implementing deep convolutional networks and matrix factorization algorithms.
Brian Vogel, 2015

##Features

Includes a simple multi-dimensional matrix class, various utilitiy functions for implementing SGD, convolutional layers, and classes that make it straightforward to implement deep convolutional neural networks.

Written in a modern C++ style from scratch. The only dependencies are Boost (optional) and OpenBLAS (only the sgemm function is used).

Uses OpenBLAS sgemm to perform matrix multiplications. The convolutional layers also use BLAS sgemm to compute the convolution using the method described in [this paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.482&rep=rep1&type=pdf). This method can use a bit more memory than the naive method but is often much faster.

Uses OpenMP where possible.

Currently runs only on the CPU, but is still fast enough to experiement with relatively small datasets such as MNIST and CIFAR. 

##Requirements

This software was developed under Ubuntu 14.04 but should also run under Mac OS X if scientific python packages, gcc, OpenBLAS, and Boost are installed with Homebrew.

Install g++-4.9 or newer.

Python is used for loading datasets and for data visualization. Either install a scientific python distribution such as [Anaconda](https://store.continuum.io/cshop/anaconda/) or manually install Python 2.7.x, Scipy, Numpy, Matplotlib, etc.

Install Boost (optional, only needed if running the collaborative filtering example) and OpenBLAS. It is important to build OpenBLAS from source so that it can take advantage of your specific CPU.

##Instalation

Go to src_cpp folder and open **makefile** in an editor. Edit the library and include paths to point to the locations of OpenBLAS and Boost libary and include folders.

Open main.cpp and set ```<number of threads>``` in

```C++
omp_set_num_threads(<number of threads>);
```
to the desired number of threads, typically the same as the number of available hardware threads of your CPU. Build the **main** executable:

```bash
$ make
```

##Setup paths

Open includePaths.py and set executablePath to point to location of the **main** executable.

Open ```mnist_example_1_run.py``` and set ```PATH_TO_MNIST_DATA``` to point to the location of the MNIST data set.

Open ```cifar10_example_1_run.py``` and set ```PATH_TO_CIFAR10_DATA``` to point to the location of the CIFAR10 data set.

Open ```ExamplesRecSystem.cpp``` and set ```training_set``` and ```probe``` to the locations of the Netflix Prize training set fold and probe.txt file. Note: This data set is no longer publicly available.

##Usage

The code currently supports deep feedforward convolutional neural networks (convnets). By default, Rmsprop is used. Various activations functions are supported. Dropout is supported. Softmax output layers are not yet supported.

There are currently two examples that illustrate the usage:

Run MNIST example:

```bash
cd src_python/mnist
python mnist_example_1_run.py
```

The training and test error will be displayed after each epoch. The default parameter settings should result in around 0.48% error on the test set.

To plot some stats and debugging plots:

```bash
python mnist_example_1_make_plots.py
```

Run CIFAR-10 example:

```bash
cd src_python/cifar10
python cifar10_example_1_run.py
```

The training and test error will be displayed after each epoch. The default parameter settings should result in around 22% error on the test set.

To plot some stats and debugging plots:

```bash
python cifar10_example_1_make_plots.py
```

The matrix factoriation classes and examples are not yet available, except for the collaborative filtering example in ```ExamplesRecSystem.h/cpp```. This example requires the Netflix Prize dataset (which is not included because it is no longer publicly available). If you happen to already have this dataset, the example can be run by typing:

```bash
cd src_cpp
./main netflix_prize_example_1
```

The error on the probe dataset should reach a minimum RMSE of around 0.91. 

##License

FreeBSD license.

##Todo

Clean up documentation. Perhaps this software might be useful to others in its present state, but be advised that some of the code documentation is currently outdated and inaccurate.

Adaptive learning rates are currently hard coded to use RMSProp.

Softmax output layer is not yet available (hope to have it soon).

Add PFN/NMF examples and classes.