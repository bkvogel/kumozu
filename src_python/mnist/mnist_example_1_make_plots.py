# -*- coding: utf-8 -*-
"""
Plot results for the MNIST example.

Be sure to run mnist_example_1_run.py before runing this file.
"""
import sys
sys.path =  ['..'] + sys.path
import arrayUtils
import includePaths
import scipy
import scipy.signal
from pylab import *
import numpy as np
import mnist_utils


def display_patches(patches, image_height, image_width):
    """
    Display the images in the supplied 'patches' matrix as an image.
    
    patches: An M x N matrix where each Mx1 column vector contain 1 image.
    Since each image is image_height x image_width pixels, this means that
    M must be equal to image_height x image_width. Thus, there are N images
    in total.
    
    The N images in 'patches' will be displayed in an image plot of tiled
    images of approximately sqrt(N) x sqrt(N) images.
    
    """
    (M, N) = patches.shape
    #print 'M = ', M
    assert M == (image_height*image_width), "patches is not consistent with supplied image dimensions."
    tile_rows_cols = int(np.ceil(np.sqrt(N))) #number of images to tile along each row and column.
    display_mat = np.zeros((tile_rows_cols*image_height, tile_rows_cols*image_width))
    index_cur_image = 0
    for cur_tile_row in range(tile_rows_cols):
        for cur_tile_col in range(tile_rows_cols):
            if index_cur_image < N:
                cur_imag_vect = patches[:,index_cur_image]
                cur_image = np.reshape(cur_imag_vect, (image_height, image_width))
                upper_left_row = cur_tile_row*image_height
                upper_left_col = cur_tile_col*image_width
                display_mat[upper_left_row:(upper_left_row+image_height), upper_left_col:(upper_left_col+image_width)] = cur_image.copy()
            index_cur_image += 1
    return display_mat.T # Mor MNIST, take transpose.

def display_3d_array_as_patches(mat3d):
    """
    Return an image containing each of the images in the supplied 3d array.
    
    mat3d: A 3d array of size R x M x N. It is assume that this corresponds to R images
    of size M x N.
    
    The R images in 'mat3d' will be arranged into a 2d array of
    images.
    
    """
    (R, M, N) = mat3d.shape
    image_height = M
    image_width = N
    tile_rows_cols = int(np.ceil(np.sqrt(R))) #number of images to tile along each row and column.
    display_mat = np.zeros((tile_rows_cols*image_height, tile_rows_cols*image_width))
    index_cur_image = 0
    for cur_tile_row in range(tile_rows_cols):
        for cur_tile_col in range(tile_rows_cols):
            if index_cur_image < R:
                cur_image = mat3d[index_cur_image,:,:]
                #max_val = np.max(cur_image)
                #print 'W max val = ', max_val
                #cur_image = cur_image/max_val
                upper_left_row = cur_tile_row*image_height
                upper_left_col = cur_tile_col*image_width
                display_mat[upper_left_row:(upper_left_row+image_height), upper_left_col:(upper_left_col+image_width)] = cur_image.copy()
            index_cur_image += 1
    return display_mat


def compute_test_error(network_output, true_output):
    """
    Test the performance of the NN on the test set. 
  
    true_output: array of class labels.    
    
    Return the test score as the fraction correct, assuming binary-valued output data.
    """
    
    class_label_estimates = np.argmax(network_output, axis=0)
    test_data_count = len(true_output)
    number_correct = 0.0
    for n in range(test_data_count):
        if class_label_estimates[n] == true_output[n]:
            number_correct += 1
    fraction_correct = float(number_correct)/float(test_data_count)
    print 'Test set fraction correct = ', fraction_correct  
    return fraction_correct
    
def plot_results():
    """
    Plot results.
    """
    W_conv_filters = arrayUtils.readArray('mnist_example_Conv_layer1_W.dat')
    W1_learned = arrayUtils.readArray('mnist_example_Conv_layer1_W.dat')
    print 'W1_learned shape = ', W1_learned.shape
    W2_learned = arrayUtils.readArray('mnist_example_Conv_layer2_W.dat')
    print 'W2_learned shape = ', W2_learned.shape


    ##################################
    # Fully-connected layer 1:
    #
    weight_updates_full_layer1 = arrayUtils.readArray('mnist_example_FC_layer1_weight_updates.dat')
    weight_max_val_full_layer1 = arrayUtils.readArray('mnist_example_FC_layer1_weight_max_val.dat')
    weight_min_val_full_layer1 = arrayUtils.readArray('mnist_example_FC_layer1_weight_min_val.dat')
    bias_updates_full_layer1 = arrayUtils.readArray('mnist_example_FC_layer1_bias_updates.dat')
    bias_max_val_full_layer1 = arrayUtils.readArray('mnist_example_FC_layer1_bias_max_val.dat')
    bias_min_val_full_layer1 = arrayUtils.readArray('mnist_example_FC_layer1_bias_min_val.dat')

    figure(765)
    subplot(321)
    plot(weight_updates_full_layer1)
    title('Weight update size vs mini-batch index for full layer 1')

    subplot(323)
    plot(weight_max_val_full_layer1, 'r--')
    plot(weight_min_val_full_layer1, 'b--')
    title('Weight max and min values vs mini-batch index for full layer 1')

    subplot(325)
    update_to_weight_ratio1 = weight_updates_full_layer1/weight_max_val_full_layer1
    plot(update_to_weight_ratio1)
    title('Weight to update ratio vs mini-batch index for full layer 1')

    subplot(322)
    plot(bias_updates_full_layer1)
    title('Bias update size vs mini-batch index for full layer 1')

    subplot(324)
    plot(bias_max_val_full_layer1, 'r--')
    plot(bias_min_val_full_layer1, 'b--')
    title('Bias max and min values vs mini-batch index for full layer 1')

    subplot(326)
    update_to_bias_ratio1 = bias_updates_full_layer1/bias_max_val_full_layer1
    plot(update_to_bias_ratio1)
    title('Bias to update ratio vs mini-batch index for full layer 1')
    ##################################


    ##################################
    # Fully-connected layer 2:
    #
    weight_updates_full_layer2 = arrayUtils.readArray('mnist_example_FC_layer2_weight_updates.dat')
    weight_max_val_full_layer2 = arrayUtils.readArray('mnist_example_FC_layer2_weight_max_val.dat')
    weight_min_val_full_layer2 = arrayUtils.readArray('mnist_example_FC_layer2_weight_min_val.dat')
    bias_updates_full_layer2 = arrayUtils.readArray('mnist_example_FC_layer2_bias_updates.dat')
    bias_max_val_full_layer2 = arrayUtils.readArray('mnist_example_FC_layer2_bias_max_val.dat')
    bias_min_val_full_layer2 = arrayUtils.readArray('mnist_example_FC_layer2_bias_min_val.dat')

    figure(766)
    subplot(321)
    plot(weight_updates_full_layer2)
    title('Weight update size vs mini-batch index for full layer 2')

    subplot(323)
    plot(weight_max_val_full_layer2, 'r--')
    plot(weight_min_val_full_layer2, 'b--')
    title('Weight max and min values vs mini-batch index for full layer 2')

    subplot(325)
    update_to_weight_ratio2 = weight_updates_full_layer2/weight_max_val_full_layer2
    plot(update_to_weight_ratio2)
    title('Weight to update ratio vs mini-batch index for full layer 2')

    subplot(322)
    plot(bias_updates_full_layer2)
    title('Bias update size vs mini-batch index for full layer 2')

    subplot(324)
    plot(bias_max_val_full_layer2, 'r--')
    plot(bias_min_val_full_layer2, 'b--')
    title('Bias max and min values vs mini-batch index for full layer 2')

    subplot(326)
    update_to_bias_ratio2 = bias_updates_full_layer2/bias_max_val_full_layer2
    plot(update_to_bias_ratio2)
    title('Bias to update ratio vs mini-batch index for full layer 2')
    ##################################

    ##################################
    # Conv layer 1:
    #
    weight_updates_conv_layer = arrayUtils.readArray('mnist_example_Conv_layer1_weight_updates.dat')
    weight_max_val_conv_layer = arrayUtils.readArray('mnist_example_Conv_layer1_weight_max_val.dat')
    weight_min_val_conv_layer = arrayUtils.readArray('mnist_example_Conv_layer1_weight_min_val.dat')
    bias_updates_conv_layer = arrayUtils.readArray('mnist_example_Conv_layer1_bias_updates.dat')
    bias_max_val_conv_layer = arrayUtils.readArray('mnist_example_Conv_layer1_bias_max_val.dat')
    bias_min_val_conv_layer = arrayUtils.readArray('mnist_example_Conv_layer1_bias_min_val.dat')

    figure(767)
    subplot(321)
    plot(weight_updates_conv_layer)
    title('Weight update size vs mini-batch index for conv layer')

    subplot(323)
    plot(weight_max_val_conv_layer, 'r--')
    plot(weight_min_val_conv_layer, 'b--')
    title('Weight max and min values vs mini-batch index for conv layer')

    subplot(325)
    update_to_weight_ratio = weight_updates_conv_layer/weight_max_val_conv_layer
    plot(update_to_weight_ratio)
    title('Update to weight ratio vs mini-batch index')

    subplot(322)
    plot(bias_updates_conv_layer)
    title('Bias update size vs mini-batch index for conv layer')

    subplot(324)
    plot(bias_max_val_conv_layer, 'r--')
    plot(bias_min_val_conv_layer, 'b--')
    title('Bias max and min values vs mini-batch index for conv layer')

    subplot(326)
    update_to_bias_ratio = bias_updates_conv_layer/bias_max_val_conv_layer
    plot(update_to_bias_ratio)
    title('Update to bias ratio vs mini-batch index')
    ##################################


    ##################################
    # Conv layer 2:
    #
    weight_updates_conv_layer2 = arrayUtils.readArray('mnist_example_Conv_layer2_weight_updates.dat')
    weight_max_val_conv_layer2 = arrayUtils.readArray('mnist_example_Conv_layer2_weight_max_val.dat')
    weight_min_val_conv_layer2 = arrayUtils.readArray('mnist_example_Conv_layer2_weight_min_val.dat')
    bias_updates_conv_layer2 = arrayUtils.readArray('mnist_example_Conv_layer2_bias_updates.dat')
    bias_max_val_conv_layer2 = arrayUtils.readArray('mnist_example_Conv_layer2_bias_max_val.dat')
    bias_min_val_conv_layer2 = arrayUtils.readArray('mnist_example_Conv_layer2_bias_min_val.dat')

    figure(867)
    subplot(321)
    plot(weight_updates_conv_layer2)
    title('Weight update size vs mini-batch index for conv layer 2')

    subplot(323)
    plot(weight_max_val_conv_layer2, 'r--')
    plot(weight_min_val_conv_layer2, 'b--')
    title('Weight max and min values vs mini-batch index for conv layer 2')

    subplot(325)
    update_to_weight_ratio2 = weight_updates_conv_layer2/weight_max_val_conv_layer2
    plot(update_to_weight_ratio2)
    title('Update to weight ratio vs mini-batch index')

    subplot(322)
    plot(bias_updates_conv_layer2)
    title('Bias update size vs mini-batch index for conv layer 2')

    subplot(324)
    plot(bias_max_val_conv_layer2, 'r--')
    plot(bias_min_val_conv_layer2, 'b--')
    title('Bias max and min values vs mini-batch index for conv layer 2')

    subplot(326)
    update_to_bias_ratio2 = bias_updates_conv_layer2/bias_max_val_conv_layer2
    plot(update_to_bias_ratio2)
    title('Update to bias ratio vs mini-batch index')
    ##################################



    training_erros = arrayUtils.readArray('mnist_example_train_errors.dat')
    test_erros = arrayUtils.readArray('mnist_example_test_errors.dat')

    figure(768)
    plot(training_erros, 'b--')
    plot(test_erros, 'r--')
    title('Train error (blue), Test error (red)')


    figure(33)
    W_conv_as_patches = display_3d_array_as_patches(W_conv_filters)
    imshow(np.abs(W_conv_as_patches), cmap=cm.hot, aspect='auto',
             interpolation='nearest')
    title('Learned W_conv as patches')
    colorbar()
    
   
    show()
    
    
if __name__ == '__main__':
    plot_results()
