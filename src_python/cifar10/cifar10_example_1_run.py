# -*- coding: utf-8 -*-
"""
CIFAR10 example.

Train a deep convolutional network on the CIFAR10 data set.
"""
import sys
sys.path =  ['..'] + sys.path
import arrayUtils
import includePaths
import os
import scipy
import scipy.signal
from pylab import *
import numpy as np
#import mnist_utils
from os.path import join
import cPickle as pickle

###############################################################
# Name of the C++ function to run:
RUN_CPP_FUNCTION_NAME = 'cifar10_example_1'
# Full path to the folder that contains the CIFAR10 training and test data.
PATH_TO_CIFAR10_DATA = '/home/brian/data/prototyping/deep_learning_prototyping/'
###############################################################


def load_cifar():
    """
    Load the training and test images and labels.
    
    """
    cifar_path = '/home/brian/data/prototyping/cifar_data'
    # Load training data batches.
    with open(join(cifar_path,'cifar-10-batches-py','data_batch_1'),'rb') as f:
        train_data_b1 = pickle.load(f)
    with open(join(cifar_path,'cifar-10-batches-py','data_batch_2'),'rb') as f:
        train_data_b2 = pickle.load(f)
    with open(join(cifar_path,'cifar-10-batches-py','data_batch_3'),'rb') as f:
        train_data_b3 = pickle.load(f)
    with open(join(cifar_path,'cifar-10-batches-py','data_batch_4'),'rb') as f:
        train_data_b4 = pickle.load(f)
    with open(join(cifar_path,'cifar-10-batches-py','data_batch_5'),'rb') as f:
        train_data_b5 = pickle.load(f)
    # Load test data
    with open(join(cifar_path,'cifar-10-batches-py','test_batch'),'rb') as f:
        test_data = pickle.load(f)


    labels_b1 = train_data_b1['labels']
    labels_b2 = train_data_b2['labels']
    labels_b3 = train_data_b3['labels']
    labels_b4 = train_data_b4['labels']
    labels_b5 = train_data_b5['labels']
    # Complete training labels as a 1 x 50000 array.
    labels_train = hstack((labels_b1, labels_b2, labels_b3, labels_b4, labels_b5))

    labels_test = np.array(test_data['labels'])

    # Shape of images_b1 is 
    #images_b1 = train_data_b1['data'].reshape((10000,3,32,32)).astype('float32')/255

    # Each training batch has size 10000 x 3072
    train_images_b1 = train_data_b1['data'].astype('float32')/255
    train_images_b2 = train_data_b2['data'].astype('float32')/255
    train_images_b3 = train_data_b3['data'].astype('float32')/255
    train_images_b4 = train_data_b4['data'].astype('float32')/255
    train_images_b5 = train_data_b5['data'].astype('float32')/255

    # Testing images. Size is 10000 x 3072
    test_images_2d_format = test_data['data'].astype('float32')/255
    
    # Complete training images as a 2D matrix with shape = 50000 x 3072 (1 image per row).
    train_images_2d_format = vstack((train_images_b1, train_images_b2, train_images_b3, train_images_b4, train_images_b5))

    # Reshape training images into shape (50000, 3, 32, 32)
    train_images_net_format = train_images_2d_format.reshape((50000,3,32,32))
    # Reshape test images into shape (10000, 3, 32, 32)
    test_images_net_format = test_images_2d_format.reshape((10000,3,32,32))


    # Keep only a subset of the data. 

    train_keep = 50000 # Keep all data
    #train_keep = 49920 # batch size up to 256.
    #train_keep = 9984 # batch size up to 256 quick tests.   
    #train_keep = 9728 # batch size up to 512 quick tests.  
    #train_keep = 1024
    #train_keep = 1000

    test_keep = 10000 # Keep all data
    #test_keep = 9984 # batch size up to 256 quick tests.   
    #test_keep = 9728 # batch size up to 512 quick tests.  
    #test_keep = 1024
    #test_keep = 1000

    train_images_net_format = train_images_net_format[0:train_keep,:,:,:]
    labels_train = labels_train[0:train_keep]

    test_images_net_format = test_images_net_format[0:test_keep,:,:,:]
    labels_test = labels_test[0:test_keep]

    with open(join(cifar_path,'cifar-10-batches-py','batches.meta'),'rb') as f:
        batch_metadata = pickle.load(f)

    label_names = batch_metadata['label_names']
    #print 'label_names = ', label_names
    #  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    DEBUG_PLOTS = False
    if (DEBUG_PLOTS):
        figure(1)
        subplot(121)
        # Change dimensions to (1000, 32, 32, 3) for displaying images.
        images_plot_format = np.rollaxis(train_images_net_format, 1, 4)
        #images_plot_format = np.rollaxis(test_images_net_format, 1, 4)
        image_number = 123
        imshow(images_plot_format[image_number,:,:,:], interpolation='nearest')
        title('Training image: ' + label_names[labels_train[image_number]])

        subplot(122)
        # Change dimensions to (1000, 32, 32, 3) for displaying images.
        images_plot_format = np.rollaxis(test_images_net_format, 1, 4)
        image_number = 456
        imshow(images_plot_format[image_number,:,:,:], interpolation='nearest')
        title('Test image: ' + label_names[labels_test[image_number]])
        show()

    return (train_images_net_format, labels_train, test_images_net_format, labels_test)
    
def computeGroundTruth(training_labels):
        """
        Compute the ground truth matix M such that M(r,c) is 1 if y^(c) = r and 0 otherwise.
        """
        training_data_count = len(training_labels)
        numClasses = np.max(training_labels) + 1
        M = np.zeros((numClasses, training_data_count))
        for c in range(training_data_count):
            classLabel = training_labels[c] # value is in range [0, numClasses-1]
            for r in range(numClasses):
                if r == classLabel:
                    M[r,c] = 1.0
        return M
    
    
def run_cifar():
    """
    Load CIFAR-10 training and test data, save to file, and run the script to learn the model.

    """
    print 'Loading data...'
    (train_images_net_format, labels_train, test_images_net_format, labels_test) = load_cifar()

    M_ground_truth_train = computeGroundTruth(labels_train)    
    M_ground_truth_test = computeGroundTruth(labels_test)    
    arrayUtils.writeArray(train_images_net_format, "training_images.dat")
    arrayUtils.writeArray(M_ground_truth_train, "training_class_labels.dat")
    arrayUtils.writeArray(test_images_net_format, "testing_images.dat")
    arrayUtils.writeArray(M_ground_truth_test, "testing_class_labels.dat")
    
    arrayUtils.writeArray(labels_test, "array_testing_labels.dat")
    arrayUtils.writeArray(labels_train, "array_training_labels.dat")
    print "Finished making data."
    runExe = includePaths.executablePath + ' ' + RUN_CPP_FUNCTION_NAME
    print(runExe)
    os.system(runExe)


if __name__ == '__main__':
    run_cifar()
