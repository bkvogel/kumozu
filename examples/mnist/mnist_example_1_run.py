"""
MNIST example.

Train a deep convolutional network on the MNIST data set.
"""
import sys
sys.path =  ['../../src_python/'] + sys.path
import arrayUtils
import os
import scipy
import scipy.signal
from pylab import *
import numpy as np
from os.path import join
import mnist_utils

###############################################################
# Name of the C++ function to run:
RUN_CPP_FUNCTION_NAME = 'mnist_example_1' 
# Full path to the folder that contains the MNIST training and test data.
PATH_TO_MNIST_DATA = 'data'
###############################################################

def load_mnist():
    """
    Load the training and test images and labels.

    Notes:
    Images are 28x28 pixels.
    There are 10 classes.
    
    train_images and test_images each have size: N x image_height x image_width 3D array where N is the number of images,
    image_height is the height in pixels and image_width is the width in pixels. 

    C-style ordering is used for the backing array of the 3D matrix containing the images.
    """


    train_images = mnist_utils.load_images_3d(join(PATH_TO_MNIST_DATA, 'train-images-idx3-ubyte')) 
    train_labels = mnist_utils.load_labels(join(PATH_TO_MNIST_DATA, 'train-labels-idx1-ubyte'))
    test_images = mnist_utils.load_images_3d(join(PATH_TO_MNIST_DATA, 't10k-images-idx3-ubyte')) 
    test_labels = mnist_utils.load_labels(join(PATH_TO_MNIST_DATA, 't10k-labels-idx1-ubyte'))
    return (train_images, train_labels, test_images, test_labels)

    
def run_mnist_test1():
    """
    Load MNIST data set, save to file, run C++ code to learn
    a deep convnet.
    """
    print 'loading training images...'
    (training_images, training_labels, testing_images, testing_labels) = load_mnist()
    

    # Set to true to run quick tests on less data.
    USE_SMALL_DATA = True
    if USE_SMALL_DATA:
        #keep = 60000
        #keep = 59904 # batch size up to 512
        #keep = 59968 # batch size 64
        
        #keep = 10000
        #keep = 9984 # batch size up to 256 quick tests.  
        #keep = 9728 # batch size up to 512 quick tests.  
        keep = 1000
        #keep = 1024 # quick tests
        #keep = 1000
        #keep = 128 # quick tests
        training_images = training_images[0:keep,:,:]
        training_labels = training_labels[0:keep]
        
    
    if USE_SMALL_DATA:
        #keep = 10000
        #keep = 9984 # batch size up to 256
        #keep = 9728 # batch size up to 512 quick tests.  
        keep = 1000
        #keep = 1024 # quick tests
        #keep = 128 # quick tests
        testing_images = testing_images[0:keep,:,:]
        testing_labels = testing_labels[0:keep]

    training_images = np.reshape(training_images, (training_images.shape[0], 1, 28, 28))
    print 'Shape is ', training_images.shape
    print 'Number of traiing images = ', training_images.shape[0]
    print 'Row pixels of each image = ', training_images.shape[2]
    print 'Column pixels of each image = ', training_images.shape[3]

    testing_images = np.reshape(testing_images, (testing_images.shape[0], 1, 28, 28))

    DEBUG = False
    if DEBUG:
        # Plot 1 image to check the format:
        image_ind = 2
        the_image = training_images[image_ind, 0, :, :] # One of the images, which is a 2D array.
        print 'Image value = ', training_labels[image_ind]
        figure(7)
        imshow(the_image, cmap=cm.hot, aspect='auto',
             interpolation='nearest')
        title('Sample image')
        colorbar()   
        show()
        sys.exit()
    
    arrayUtils.writeArray(training_images, "training_images.dat")
    arrayUtils.writeArray(testing_images, "testing_images.dat")
    arrayUtils.writeArray(testing_labels, "array_testing_labels.dat")
    arrayUtils.writeArray(training_labels, "array_training_labels.dat")
    print "Finished making data."
    runExe = '../../src_cpp/main ' + RUN_CPP_FUNCTION_NAME
    print(runExe)
    os.system(runExe)
    
if __name__ == '__main__':
    run_mnist_test1()
