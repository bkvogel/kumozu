/*
 * Copyright (c) 2005-2015, Brian K. Vogel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 *
 */

#include "ExamplesConvNet.h"
#include <cstdlib>
#include <string.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <ctime>
#include <omp.h>
#include <algorithm>

#include "Matrix.h"
#include "MatrixIO.h"
#include "UnitTests.h"
#include "PlotUtilities.h"

#include "SequentialLayer.h"
#include "BoxActivationFunction.h"
#include "ColumnActivationFunction.h"
#include "ConvLayer2D.h"
#include "BatchNormalization3D.h"
#include "Dropout3D.h"
#include "PoolingLayer.h"
#include "ImageToColumnLayer.h"
#include "Dropout1D.h"
#include "LinearLayer.h"
#include "BatchNormalization1D.h"
#include "CrossEntropyCostFunction.h"
#include "MinibatchTrainer.h"
#include "Accumulator.h"


// Uncomment following line to disable assertion checking.
//#define NDEBUG
// Uncomment to enable assertion checking.
//#undef NDEBUG

using namespace std;

namespace kumozu {


// This is just a smaller version of the CIFAR 10 example network that I quickly put together to
// try on the MNIST data set.
// This network reaches around 0.35% test error.
void mnist_example_1() {
    cout << "MNIST Example 1" << endl << endl;
    /////////////////////////////////////////////////////////////////////////////////////////
    // Load training and test data
    //

    MatrixF full_train_images = load_matrix("training_images.dat");
    MatrixF full_test_images = load_matrix("testing_images.dat");
    MatrixF target_training_labels_float = load_matrix("array_training_labels.dat"); // fixme: float -> int
    MatrixF target_testing_labels_float = load_matrix("array_testing_labels.dat");

    // Hack to put training labels in integer-valued matrix:
    Matrix<int> target_training_labels(target_training_labels_float.get_extents());
    for (int i = 0; i < target_training_labels.size(); i++) {
        target_training_labels[i] = static_cast<int>(target_training_labels_float[i]);
    }
    Matrix<int> target_testing_labels(target_testing_labels_float.get_extents());
    for (int i = 0; i < target_testing_labels.size(); i++) {
        target_testing_labels[i] = static_cast<int>(target_testing_labels_float[i]);
    }
    cout << "Test examples = " << target_testing_labels.extent(0) << endl;

    /////////////////////////////////////////////////////////////////////////////////////////
    // Set parameters for each layer of the network.
    //

    const int minibatch_size = 50; // 50 // try 10-200


    const int dim_fully_connected_hidden = 512; // 512

    const float batch_norm_momentum = 0.05f;

    const int class_label_count = 1 + static_cast<int>(max_value(target_testing_labels));
    cout << "Class label count = " << class_label_count << endl;

    // Notes:
    // image depth is full_train_images.extent(1), which is 1 for greyscale images, 3 for color
    // image height is full_train_images.extent(2)
    // image width is full_train_images.extent(3)

    const bool conv_fixed_rand = false; //
    const bool linear_fixed_rand = false; //

    // Try starting with large momentum and reduce as training progresses.
    //float batch_norm_momentum = 0.05f; // 0.1
    const bool enable_gamma_beta = true; // ok, this does seem to help if set true.

    const bool is_enable_bias_linear_layers = false;

    SequentialLayer network("sequential network 1");

    ImageActivationFunction::ACTIVATION_TYPE box_activation_type = ImageActivationFunction::ACTIVATION_TYPE::leakyReLU;
    ColumnActivationFunction::ACTIVATION_TYPE col_activation_type = ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU;

    // Notes:
    // Pooling region extents are (depth, height, width) where depth is the number of convolution filter channels. Therefore,
    // pooling is often performed along the height and width only, but you can try "3D max pooling" to include the depth also.

    const int filter_height1 = 3; //
    const int filter_width1 = 3; //
    const int filter_count1 = 64; // Number of convolutional filters.
    const vector<int> pooling_region_extents1 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes1 = {1, 1, 1};
    ConvLayer2D conv_layer1(filter_count1, filter_height1, filter_width1, is_enable_bias_linear_layers, "Conv Layer 1");
    conv_layer1.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer1);
    BatchNormalization3D batch_norm3d_1(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 1");
    network.schedule_layer(batch_norm3d_1);
    ImageActivationFunction box_activation_layer1(box_activation_type, "Box Activation Function 1");
    network.schedule_layer(box_activation_layer1);
    PoolingLayer pooling_layer1(pooling_region_extents1, pooling_region_step_sizes1, "Pooling Layer 1");
    //network.add_layer(pooling_layer1);
    Dropout3D dropout3d_1(0.8f, "Dropout3D 1");
    network.schedule_layer(dropout3d_1);

    const int filter_height2 = 3; // 5
    const int filter_width2 = 3; // 5
    const int filter_count2 = 64; //
    ConvLayer2D conv_layer2(filter_count2, filter_height2, filter_width2, is_enable_bias_linear_layers, "Conv Layer 2");
    conv_layer2.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer2);
    BatchNormalization3D batch_norm3d_2(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 2");
    network.schedule_layer(batch_norm3d_2);
    ImageActivationFunction box_activation_layer2(box_activation_type, "Box Activation Function 2");
    network.schedule_layer(box_activation_layer2);
    const vector<int> pooling_region_extents2 = {1, 3, 3};
    const vector<int> pooling_region_step_sizes2 = {1, 2, 2};
    PoolingLayer pooling_layer2(pooling_region_extents2, pooling_region_step_sizes2, "Pooling Layer 2");
    network.schedule_layer(pooling_layer2);
    Dropout3D dropout3d_2(0.8f, "Dropout3D 2");
    network.schedule_layer(dropout3d_2);

    const int filter_height3 = 3; // 5
    const int filter_width3 = 3; // 5
    const int filter_count3 = 128; // 128
    ConvLayer2D conv_layer3(filter_count3, filter_height3, filter_width3, is_enable_bias_linear_layers, "Conv Layer 3");
    conv_layer3.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer3);
    BatchNormalization3D batch_norm3d_3(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 3");
    network.schedule_layer(batch_norm3d_3);
    ImageActivationFunction box_activation_layer3(box_activation_type, "Box Activation Function 3");
    network.schedule_layer(box_activation_layer3);
    const vector<int> pooling_region_extents3 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes3 = {1, 1, 1};
    PoolingLayer pooling_layer3(pooling_region_extents3, pooling_region_step_sizes3, "Pooling Layer 3");
    //network.add_layer(pooling_layer3);
    Dropout3D dropout3d_3(0.8f, "Dropout3D 3");
    //network.add_layer(dropout3d_3);

    const int filter_height4 = 3; // 5
    const int filter_width4 = 3; // 5
    const int filter_count4 = 128; // 128
    ConvLayer2D conv_layer4(filter_count4, filter_height4, filter_width4, is_enable_bias_linear_layers, "Conv Layer 4");
    conv_layer4.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer4);
    BatchNormalization3D batch_norm3d_4(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 4");
    network.schedule_layer(batch_norm3d_4);
    ImageActivationFunction box_activation_layer4(box_activation_type, "Box Activation Function 4");
    network.schedule_layer(box_activation_layer4);
    const vector<int> pooling_region_extents4 = {1, 3, 3};
    const vector<int> pooling_region_step_sizes4 = {1, 2, 2};
    PoolingLayer pooling_layer4(pooling_region_extents4, pooling_region_step_sizes4, "Pooling Layer 4");
    network.schedule_layer(pooling_layer4);
    Dropout3D dropout3d_4(0.8f, "Dropout3D 4");
    network.schedule_layer(dropout3d_4);

    const int filter_height5 = 3; // 5
    const int filter_width5 = 3; // 5
    const int filter_count5 = 256; // 128
    ConvLayer2D conv_layer5(filter_count5, filter_height5, filter_width5, is_enable_bias_linear_layers, "Conv Layer 5");
    conv_layer5.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer5);
    BatchNormalization3D batch_norm3d_5(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 5");
    network.schedule_layer(batch_norm3d_5);
    ImageActivationFunction box_activation_layer5(box_activation_type, "Box Activation Function 5");
    network.schedule_layer(box_activation_layer5);
    const vector<int> pooling_region_extents5 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes5 = {1, 1, 1};
    PoolingLayer pooling_layer5(pooling_region_extents5, pooling_region_step_sizes5, "Pooling Layer 5");
    //network.add_layer(pooling_layer5);
    Dropout3D dropout3d_5(0.9f, "Dropout3D 5");
    //network.add_layer(dropout3d_5);

    const int filter_height6 = 3; //
    const int filter_width6 = 3; //
    const int filter_count6 = 256; //
    ConvLayer2D conv_layer6(filter_count6, filter_height6, filter_width6, is_enable_bias_linear_layers, "Conv Layer 6");
    conv_layer6.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer6);
    BatchNormalization3D batch_norm3d_6(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 6");
    network.schedule_layer(batch_norm3d_6);
    ImageActivationFunction box_activation_layer6(box_activation_type, "Box Activation Function 6");
    network.schedule_layer(box_activation_layer6);
    const vector<int> pooling_region_extents6 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes6 = {1, 1, 1};
    PoolingLayer pooling_layer6(pooling_region_extents6, pooling_region_step_sizes6, "Pooling Layer 6");
    //network.add_layer(pooling_layer6);
    Dropout3D dropout3d_6(0.9f, "Dropout3D 6");
    //network.add_layer(dropout3d_6);

    const int filter_height7 = 3; //
    const int filter_width7 = 3; //
    const int filter_count7 = 256; //
    ConvLayer2D conv_layer7(filter_count7, filter_height7, filter_width7, is_enable_bias_linear_layers, "Conv Layer 7");
    conv_layer7.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer7);
    BatchNormalization3D batch_norm3d_7(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 7");
    network.schedule_layer(batch_norm3d_7);
    ImageActivationFunction box_activation_layer7(box_activation_type, "Box Activation Function 7");
    network.schedule_layer(box_activation_layer7);
    const vector<int> pooling_region_extents7 = {1, 3, 3};
    const vector<int> pooling_region_step_sizes7 = {1, 2, 2};
    PoolingLayer pooling_layer7(pooling_region_extents7, pooling_region_step_sizes7, "Pooling Layer 7");
    network.schedule_layer(pooling_layer7);
    Dropout3D dropout3d_7(0.8f, "Dropout3D 7");
    network.schedule_layer(dropout3d_7);

    ImageToColumnLayer image_to_col_layer("Image To Column Layer 1");
    network.schedule_layer(image_to_col_layer);

    const float prob_keep1d_1 = 0.5f;
    Dropout1D dropout1d_1(prob_keep1d_1, "Dropout1D 1");
    network.schedule_layer(dropout1d_1);

    LinearLayer linear_laye1(dim_fully_connected_hidden, is_enable_bias_linear_layers, "Linear Layer 1");
    linear_laye1.enable_fixed_random_back_prop(linear_fixed_rand);
    //linear_laye1.enable_bias(is_enable_bias_linear_layers);
    network.schedule_layer(linear_laye1);



    BatchNormalization1D batch_norm1d_1(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 1d 1");
    network.schedule_layer(batch_norm1d_1);
    ColumnActivationFunction column_activation_layer1(col_activation_type, "Column Activation Function 1");
    network.schedule_layer(column_activation_layer1);

    const float prob_keep1d_2 = 0.7f;
    Dropout1D dropout1d_2(prob_keep1d_2, "Dropout1D 2");
    //network.add_layer(dropout1d_2);

    LinearLayer linear_laye2(class_label_count, "Linear Layer 2");
    linear_laye2.enable_fixed_random_back_prop(linear_fixed_rand);
    network.schedule_layer(linear_laye2);

    CrossEntropyCostFunction cost_func("Cross Entropy Cost Function");
    network.schedule_layer(cost_func);
    // Set up learning rates.
    float learning_rate_weights = 1e-3f; // 1e-3
    float learning_rate_bias = 1e-3f; //
    float weight_decay = 5e-4f; // doesn't do much
    const bool enable_weight_decay = true;

    /////////////////////////////////////////////////////////////////////////////////////////
    // Train the network
    //
    MinibatchTrainer<float, int> trainer(full_train_images, 0, target_training_labels, 0, minibatch_size);
    trainer.enable_shuffling(true);

    VariableF &train_input_mini = trainer.get_input_batch();
    const MatrixI &train_output_mini = trainer.get_output_batch_mat();
    cost_func.set_target_activations(train_output_mini);
    // Connect the input activations and deltas to the network.
    network.create_input_port(train_input_mini);
    // Initialize the network:
    network.forward();

    // Optionally load saved parameters:
    if (false) {
        network.load_parameters("mnist_model_1");
    }
    Updater updater(network.get_params());
    updater.set_mode_constant_learning_rate(learning_rate_weights); //
    updater.set_flag_weight_decay(weight_decay, enable_weight_decay);

    // For testing:
    MinibatchTrainer<float, int> tester(full_test_images, 0, target_testing_labels, 0, minibatch_size);
    tester.enable_shuffling(true); // Not necessary to shuffle the test set but does not hurt.
    VariableF &test_input_mini = tester.get_input_batch();
    const MatrixI &test_output_mini = tester.get_output_batch_mat();
    int train_epochs = 0;
    network.set_train_mode(true);

    Accumulator train_accumulator(minibatch_size);
    Accumulator test_accumulator(minibatch_size);
    Accumulator train_loss_accumulator(minibatch_size);
    Accumulator test_loss_accumulator(minibatch_size);
    // Vectors for plotting:
    vector<float> training_errors;
    vector<float> test_errors;
    Gnuplot plot_errors;
    Gnuplot plot_activations;
    while (train_epochs < 125) { // 100-150 epochs is good
        cerr << ".";
        bool end_epoch = trainer.next(); // Get next training mini-batch
        network.forward();
        train_loss_accumulator.accumulate(cost_func.get_output_data()[0]);
        train_accumulator.accumulate(error_count(linear_laye2.get_output_data(), train_output_mini));
        network.back_propagate();
        updater.update();

        if (end_epoch) {
            cout << endl << "---------------" << endl;
            //network.print_paramater_stats(); // enable for debugging info
            cout << "Training epochs: " << train_epochs << endl;
            cout << "Train error rate: " << train_accumulator.get_mean() << endl;
            cout << "Train loss/example: " << train_loss_accumulator.get_mean() << endl;
            cout << "Train examples: " << train_accumulator.get_counter() << endl;
            training_errors.push_back(train_accumulator.get_mean());
            train_accumulator.reset();
            train_loss_accumulator.reset();
            test_accumulator.reset();
            test_loss_accumulator.reset();
            network.set_train_mode(false);
            // Now connect the test input activations and deltas to the network.
            network.create_input_port(test_input_mini);
            cost_func.set_target_activations(test_output_mini);
            bool done = false;
            while (!done) {
                done = tester.next(); // Get next test mini-batch
                network.forward();
                test_loss_accumulator.accumulate(cost_func.get_output_data()[0]);
                test_accumulator.accumulate(error_count(linear_laye2.get_output_data(), test_output_mini));
            }
            cout << "Test error rate: " << test_accumulator.get_mean() << endl;
            cout << "Test loss/example: " << test_loss_accumulator.get_mean() << endl;
            cout << "Test examples: " << test_accumulator.get_counter() << endl;
            test_errors.push_back(test_accumulator.get_mean());
            // Connect the input activations and deltas to the network.
            network.create_input_port(train_input_mini);
            cost_func.set_target_activations(train_output_mini);
            // Display plots?
            if (true) {

                // Update plot:
                plot_errors << "set multiplot layout 2,1 title 'Train/Test errors'" << endl;
                plot(plot_errors, training_errors, "Training error");
                plot(plot_errors, test_errors, "Test error");
                plot_errors << "unset multiplot" << endl;

                // Plot some activations:
                plot_activations << "set multiplot layout 2,5 title 'Network activations'" << endl;

                // Plot an input image to network:
                const int image_index = 0; // arbitrary choice
                MatrixF one_image = select(test_input_mini.data, 0, image_index);
                scale(one_image, one_image, 255.0f);
                plot_images_greyscale_3dim(plot_activations, one_image, "Input test image");



                // Plot conv layer 1 output:
                MatrixF images_conv_layer_1_out = select(conv_layer1.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_1_out, conv_layer1.get_name());

                // Plot conv layer 2 output:
                MatrixF images_conv_layer_2_out = select(conv_layer2.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_2_out, conv_layer2.get_name());

                // Plot conv layer 3 output:
                MatrixF images_conv_layer_3_out = select(conv_layer3.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_3_out, conv_layer3.get_name());

                // Plot conv layer 4 output:
                MatrixF images_conv_layer_4_out = select(conv_layer4.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_4_out, conv_layer4.get_name());

                // Plot conv layer 5 output:
                MatrixF images_conv_layer_5_out = select(conv_layer5.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_5_out, conv_layer5.get_name());

                // Plot conv layer 6 output:
                MatrixF images_conv_layer_6_out = select(conv_layer6.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_6_out, conv_layer6.get_name());

                // Plot conv layer 7 output:
                MatrixF images_conv_layer_7_out = select(conv_layer7.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_7_out, conv_layer7.get_name());


                // Plot linear layer 1:
                MatrixF linear_laye1_out = select(linear_laye1.get_output_data(), 1, image_index);
                plot_image_greyscale(plot_activations, linear_laye1_out, linear_laye1.get_name());

                // Plot linear layer 2:
                MatrixF linear_laye2_out = select(linear_laye2.get_output_data(), 1, image_index);
                plot_image_greyscale(plot_activations, linear_laye2_out, linear_laye2.get_name());

                plot_activations << "unset multiplot" << endl;
            }

            network.set_train_mode(true);
        }


        if (end_epoch) {
            train_epochs++;

            learning_rate_weights *= 0.97f;
            cout << "New learning rate weights: " << learning_rate_weights << endl;
            updater.set_mode_constant_learning_rate(learning_rate_weights);
            learning_rate_bias *= 0.97f;
            cout << "New learning rate bias: " << learning_rate_bias << endl;
            weight_decay *= 0.97f;
            cout << "New weight decay: " << weight_decay << endl;
            updater.set_flag_weight_decay(weight_decay, enable_weight_decay);
        }
    }


    /////////////////////////////////////////////////////////////////////////////////////////
    // Save parameters and stats.
    network.save_parameters("mnist_model_1");

}


// Uses the VGG-like network from:
// http://torch.ch/blog/2015/07/30/cifar.html
// which is based on the network architecture from the paper:
// http://arxiv.org/pdf/1409.1556v6.pdf
//
// The network uses 13 convolutional layers + 2 fully-connected layers with softmax output and NLL cost function.
// Leaky ReLU activations are used. Vanilla SGD + weight decay is used.
// No data augmentation is used.
//
// The main differences from the Torch7 blog example:
// I feed RGB images into the network after scaling to [0,1] range but do not perform any preprocessing.
// I perform max pooling using 3x3 regions with step size 2 so there is a little overlap.
// I use Leaky ReLU instead of ReLU.
// I do not perform any data augmentation, not even horizontal flips.
//
// Using BN + dropout results in 9.3% test error after about 150-170 iterations. Takes a little over 24 hours on 5960x.
void cifar10_example_1() {
    cout << "CIFAR10 Example 1" << endl << endl;
    /////////////////////////////////////////////////////////////////////////////////////////
    // Load training and test data
    //
    MatrixF full_train_images = load_matrix("training_images.dat");
    MatrixF full_test_images = load_matrix("testing_images.dat");
    MatrixF target_training_labels_float = load_matrix("array_training_labels.dat"); // fixme: float -> int
    MatrixF target_testing_labels_float = load_matrix("array_testing_labels.dat"); //

    // Hack to put training labels in integer-valued matrix:
    Matrix<int> target_training_labels(target_training_labels_float.get_extents());
    for (int i = 0; i < target_training_labels.size(); i++) {
        target_training_labels[i] = static_cast<int>(target_training_labels_float[i]);
    }
    Matrix<int> target_testing_labels(target_testing_labels_float.get_extents());
    for (int i = 0; i < target_testing_labels.size(); i++) {
        target_testing_labels[i] = static_cast<int>(target_testing_labels_float[i]);
    }
    cout << "Test examples = " << target_testing_labels.extent(0) << endl;

    /////////////////////////////////////////////////////////////////////////////////////////
    // Set parameters for each layer of the network.
    //

    const int minibatch_size = 100; // try 10-200
    const int dim_fully_connected_hidden = 512; // 512
    const float batch_norm_momentum = 0.01f;
    const int class_label_count = 1 + static_cast<int>(max_value(target_testing_labels));
    cout << "Class label count = " << class_label_count << endl;

    // False is default. If set to true, fixed random weights are used in the convolutional layers.
    const bool conv_fixed_rand = false; //
    // False is default. If set to true, fixed random weights are used in the fully-connected linear layers.
    const bool linear_fixed_rand = false; //

    // Enable/disable the learned gamma and beta parameters in the batch normalization layers. True is default (enabled).
    const bool enable_gamma_beta = true; // ok, this does seem to help if set true.

    // Enable/disable bias parameters in the convolutional and fully-connected layers. Bias should be disabled when
    // batch normalization is used since batch normalization has its own bias.
    const bool is_enable_bias_linear_layers = false;

    SequentialLayer network("sequential network 1");

    ImageActivationFunction::ACTIVATION_TYPE box_activation_type = ImageActivationFunction::ACTIVATION_TYPE::leakyReLU;
    //BoxActivationFunction::ACTIVATION_TYPE box_activation_type = BoxActivationFunction::ACTIVATION_TYPE::ReLU;
    ColumnActivationFunction::ACTIVATION_TYPE col_activation_type = ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU;
    //ColumnActivationFunction::ACTIVATION_TYPE col_activation_type = ColumnActivationFunction::ACTIVATION_TYPE::ReLU;

    // Notes:
    // Pooling region extents are (depth, height, width) where depth is the number of convolution filter channels. Therefore,
    // pooling is often performed along the height and width only, but you can try "3D max pooling" to include the depth also.

    const int filter_height1 = 3; //
    const int filter_width1 = 3; //
    const int filter_count1 = 64; // Number of convolutional filters.
    const vector<int> pooling_region_extents1 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes1 = {1, 1, 1};
    ConvLayer2D conv_layer1(filter_count1, filter_height1, filter_width1, is_enable_bias_linear_layers, "Conv Layer 1");
    conv_layer1.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer1);
    BatchNormalization3D batch_norm3d_1(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 1");
    network.schedule_layer(batch_norm3d_1);
    ImageActivationFunction box_activation_layer1(box_activation_type, "Box Activation Function 1");
    network.schedule_layer(box_activation_layer1);
    PoolingLayer pooling_layer1(pooling_region_extents1, pooling_region_step_sizes1, "Pooling Layer 1");
    //network.add_layer(pooling_layer1);
    Dropout3D dropout3d_1(0.8f, "Dropout3D 1");
    network.schedule_layer(dropout3d_1);

    const int filter_height2 = 3; // 5
    const int filter_width2 = 3; // 5
    const int filter_count2 = 64; //
    ConvLayer2D conv_layer2(filter_count2, filter_height2, filter_width2, is_enable_bias_linear_layers, "Conv Layer 2");
    conv_layer2.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer2);
    BatchNormalization3D batch_norm3d_2(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 2");
    network.schedule_layer(batch_norm3d_2);
    ImageActivationFunction box_activation_layer2(box_activation_type, "Box Activation Function 2");
    network.schedule_layer(box_activation_layer2);
    const vector<int> pooling_region_extents2 = {1, 3, 3};
    const vector<int> pooling_region_step_sizes2 = {1, 2, 2};
    PoolingLayer pooling_layer2(pooling_region_extents2, pooling_region_step_sizes2, "Pooling Layer 2");
    network.schedule_layer(pooling_layer2);
    Dropout3D dropout3d_2(0.8f, "Dropout3D 2");
    network.schedule_layer(dropout3d_2);

    const int filter_height3 = 3; // 5
    const int filter_width3 = 3; // 5
    const int filter_count3 = 128; // 128
    ConvLayer2D conv_layer3(filter_count3, filter_height3, filter_width3, is_enable_bias_linear_layers, "Conv Layer 3");
    conv_layer3.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer3);
    BatchNormalization3D batch_norm3d_3(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 3");
    network.schedule_layer(batch_norm3d_3);
    ImageActivationFunction box_activation_layer3(box_activation_type, "Box Activation Function 3");
    network.schedule_layer(box_activation_layer3);
    const vector<int> pooling_region_extents3 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes3 = {1, 1, 1};
    PoolingLayer pooling_layer3(pooling_region_extents3, pooling_region_step_sizes3, "Pooling Layer 3");
    //network.add_layer(pooling_layer3);
    Dropout3D dropout3d_3(0.8f, "Dropout3D 3");
    //network.add_layer(dropout3d_3);

    const int filter_height4 = 3; // 5
    const int filter_width4 = 3; // 5
    const int filter_count4 = 128; // 128
    ConvLayer2D conv_layer4(filter_count4, filter_height4, filter_width4, is_enable_bias_linear_layers, "Conv Layer 4");
    conv_layer4.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer4);
    BatchNormalization3D batch_norm3d_4(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 4");
    network.schedule_layer(batch_norm3d_4);
    ImageActivationFunction box_activation_layer4(box_activation_type, "Box Activation Function 4");
    network.schedule_layer(box_activation_layer4);
    const vector<int> pooling_region_extents4 = {1, 3, 3};
    const vector<int> pooling_region_step_sizes4 = {1, 2, 2};
    PoolingLayer pooling_layer4(pooling_region_extents4, pooling_region_step_sizes4, "Pooling Layer 4");
    network.schedule_layer(pooling_layer4);
    Dropout3D dropout3d_4(0.8f, "Dropout3D 4");
    network.schedule_layer(dropout3d_4);

    const int filter_height5 = 3; // 5
    const int filter_width5 = 3; // 5
    const int filter_count5 = 256; // 128
    ConvLayer2D conv_layer5(filter_count5, filter_height5, filter_width5, is_enable_bias_linear_layers, "Conv Layer 5");
    conv_layer5.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer5);
    BatchNormalization3D batch_norm3d_5(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 5");
    network.schedule_layer(batch_norm3d_5);
    ImageActivationFunction box_activation_layer5(box_activation_type, "Box Activation Function 5");
    network.schedule_layer(box_activation_layer5);
    const vector<int> pooling_region_extents5 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes5 = {1, 1, 1};
    PoolingLayer pooling_layer5(pooling_region_extents5, pooling_region_step_sizes5, "Pooling Layer 5");
    //network.add_layer(pooling_layer5);
    Dropout3D dropout3d_5(0.9f, "Dropout3D 5");
    //network.add_layer(dropout3d_5);

    const int filter_height6 = 3; //
    const int filter_width6 = 3; //
    const int filter_count6 = 256; //
    ConvLayer2D conv_layer6(filter_count6, filter_height6, filter_width6, is_enable_bias_linear_layers, "Conv Layer 6");
    conv_layer6.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer6);
    BatchNormalization3D batch_norm3d_6(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 6");
    network.schedule_layer(batch_norm3d_6);
    ImageActivationFunction box_activation_layer6(box_activation_type, "Box Activation Function 6");
    network.schedule_layer(box_activation_layer6);
    const vector<int> pooling_region_extents6 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes6 = {1, 1, 1};
    PoolingLayer pooling_layer6(pooling_region_extents6, pooling_region_step_sizes6, "Pooling Layer 6");
    //network.add_layer(pooling_layer6);
    Dropout3D dropout3d_6(0.9f, "Dropout3D 6");
    //network.add_layer(dropout3d_6);

    const int filter_height7 = 3; //
    const int filter_width7 = 3; //
    const int filter_count7 = 256; //
    ConvLayer2D conv_layer7(filter_count7, filter_height7, filter_width7, is_enable_bias_linear_layers, "Conv Layer 7");
    conv_layer7.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer7);
    BatchNormalization3D batch_norm3d_7(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 7");
    network.schedule_layer(batch_norm3d_7);
    ImageActivationFunction box_activation_layer7(box_activation_type, "Box Activation Function 7");
    network.schedule_layer(box_activation_layer7);
    const vector<int> pooling_region_extents7 = {1, 3, 3};
    const vector<int> pooling_region_step_sizes7 = {1, 2, 2};
    PoolingLayer pooling_layer7(pooling_region_extents7, pooling_region_step_sizes7, "Pooling Layer 7");
    network.schedule_layer(pooling_layer7);
    Dropout3D dropout3d_7(0.8f, "Dropout3D 7");
    network.schedule_layer(dropout3d_7);


    const int filter_height8 = 3;
    const int filter_width8 = 3;
    const int filter_count8 = 512;
    ConvLayer2D conv_layer8(filter_count8, filter_height8, filter_width8, is_enable_bias_linear_layers, "Conv Layer 8");
    conv_layer8.enable_fixed_random_back_prop(conv_fixed_rand);
    //conv_layer8.enable_bias(is_enable_bias_linear_layers);
    network.schedule_layer(conv_layer8);
    BatchNormalization3D batch_norm3d_8(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 8");
    network.schedule_layer(batch_norm3d_8);
    ImageActivationFunction box_activation_layer8(box_activation_type, "Box Activation Function 8");
    network.schedule_layer(box_activation_layer8);
    const vector<int> pooling_region_extents8 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes8 = {1, 1, 1};
    PoolingLayer pooling_layer8(pooling_region_extents8, pooling_region_step_sizes8, "Pooling Layer 8");
    //network.add_layer(pooling_layer8);
    Dropout3D dropout3d_8(0.9f, "Dropout3D 8");
    //network.add_layer(dropout3d_8);

    const int filter_height9 = 3;
    const int filter_width9 = 3;
    const int filter_count9 = 512;
    ConvLayer2D conv_layer9(filter_count9, filter_height9, filter_width9, is_enable_bias_linear_layers, "Conv Layer 9");
    conv_layer9.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer9);
    BatchNormalization3D batch_norm3d_9(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 9");
    network.schedule_layer(batch_norm3d_9);
    ImageActivationFunction box_activation_layer9(box_activation_type, "Box Activation Function 9");
    network.schedule_layer(box_activation_layer9);
    const vector<int> pooling_region_extents9 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes9 = {1, 1, 1};
    PoolingLayer pooling_layer9(pooling_region_extents9, pooling_region_step_sizes9, "Pooling Layer 9");
    //network.add_layer(pooling_layer9);
    Dropout3D dropout3d_9(0.9f, "Dropout3D 9");
    //network.add_layer(dropout3d_9);

    const int filter_height10 = 3;
    const int filter_width10 = 3;
    const int filter_count10 = 512;
    ConvLayer2D conv_layer10(filter_count10, filter_height10, filter_width10, is_enable_bias_linear_layers, "Conv Layer 10");
    conv_layer10.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer10);
    BatchNormalization3D batch_norm3d_10(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 10");
    network.schedule_layer(batch_norm3d_10);
    ImageActivationFunction box_activation_layer10(box_activation_type, "Box Activation Function 10");
    network.schedule_layer(box_activation_layer10);
    const vector<int> pooling_region_extents10 = {1, 3, 3};
    const vector<int> pooling_region_step_sizes10 = {1, 2, 2};
    PoolingLayer pooling_layer10(pooling_region_extents10, pooling_region_step_sizes10, "Pooling Layer 10");
    network.schedule_layer(pooling_layer10);
    Dropout3D dropout3d_10(0.8f, "Dropout3D 10");
    network.schedule_layer(dropout3d_10);

    const int filter_height11 = 3;
    const int filter_width11 = 3;
    const int filter_count11 = 512;
    ConvLayer2D conv_layer11(filter_count11, filter_height11, filter_width11, is_enable_bias_linear_layers, "Conv Layer 11");
    conv_layer11.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer11);
    BatchNormalization3D batch_norm3d_11(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 11");
    network.schedule_layer(batch_norm3d_11);
    ImageActivationFunction box_activation_layer11(box_activation_type, "Box Activation Function 11");
    network.schedule_layer(box_activation_layer11);
    const vector<int> pooling_region_extents11 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes11 = {1, 1, 1};
    PoolingLayer pooling_layer11(pooling_region_extents11, pooling_region_step_sizes11, "Pooling Layer 11");
    //network.add_layer(pooling_layer11);
    Dropout3D dropout3d_11(0.9f, "Dropout3D 11");
    //network.add_layer(dropout3d_11);

    const int filter_height12 = 3;
    const int filter_width12 = 3;
    const int filter_count12 = 512;
    ConvLayer2D conv_layer12(filter_count12, filter_height12, filter_width12, is_enable_bias_linear_layers, "Conv Layer 12");
    conv_layer12.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer12);
    BatchNormalization3D batch_norm3d_12(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 12");
    network.schedule_layer(batch_norm3d_12);
    ImageActivationFunction box_activation_layer12(box_activation_type, "Box Activation Function 12");
    network.schedule_layer(box_activation_layer12);
    const vector<int> pooling_region_extents12 = {1, 1, 1};
    const vector<int> pooling_region_step_sizes12 = {1, 1, 1};
    PoolingLayer pooling_layer12(pooling_region_extents12, pooling_region_step_sizes12, "Pooling Layer 12");
    //network.add_layer(pooling_layer12);
    Dropout3D dropout3d_12(0.9f, "Dropout3D 12");
    //network.add_layer(dropout3d_12);

    const int filter_height13 = 3;
    const int filter_width13 = 3;
    const int filter_count13 = 512;
    ConvLayer2D conv_layer13(filter_count13, filter_height13, filter_width13, is_enable_bias_linear_layers, "Conv Layer 13");
    conv_layer13.enable_fixed_random_back_prop(conv_fixed_rand);
    network.schedule_layer(conv_layer13);
    BatchNormalization3D batch_norm3d_13(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 3d 13");
    network.schedule_layer(batch_norm3d_13);
    ImageActivationFunction box_activation_layer13(box_activation_type, "Box Activation Function 13");
    network.schedule_layer(box_activation_layer13);
    const vector<int> pooling_region_extents13 = {1, 2, 2};
    const vector<int> pooling_region_step_sizes13 = {1, 2, 2};
    PoolingLayer pooling_layer13(pooling_region_extents13, pooling_region_step_sizes13, "Pooling Layer 13");
    network.schedule_layer(pooling_layer13);
    Dropout3D dropout3d_13(0.8f, "Dropout3D 13");
    network.schedule_layer(dropout3d_13);


    ImageToColumnLayer image_to_col_layer("Image To Column Layer 1");
    network.schedule_layer(image_to_col_layer);

    const float prob_keep1d_1 = 0.5f;
    Dropout1D dropout1d_1(prob_keep1d_1, "Dropout1D 1");
    network.schedule_layer(dropout1d_1);

    LinearLayer linear_laye1(dim_fully_connected_hidden, is_enable_bias_linear_layers, "Linear Layer 1");
    linear_laye1.enable_fixed_random_back_prop(linear_fixed_rand);
    //linear_laye1.enable_bias(is_enable_bias_linear_layers);
    network.schedule_layer(linear_laye1);

    BatchNormalization1D batch_norm1d_1(enable_gamma_beta, batch_norm_momentum, "Batch Normalization 1d 1");
    network.schedule_layer(batch_norm1d_1);
    ColumnActivationFunction column_activation_layer1(col_activation_type, "Column Activation Function 1");
    network.schedule_layer(column_activation_layer1);


    const float prob_keep1d_2 = 0.7f;
    Dropout1D dropout1d_2(prob_keep1d_2, "Dropout1D 2");
    //network.add_layer(dropout1d_2);

    LinearLayer linear_laye2(class_label_count, "Linear Layer 2");
    linear_laye2.enable_fixed_random_back_prop(linear_fixed_rand);
    network.schedule_layer(linear_laye2);

    CrossEntropyCostFunction cost_func("Cross Entropy Cost Function");
    network.schedule_layer(cost_func);
    // Set up learning rates.
    float learning_rate_weights = 1e-3f; // 1e-3
    float learning_rate_bias = 1e-3f; //
    float weight_decay = 5e-4f; // doesn't do much
    const bool enable_weight_decay = true;




    /////////////////////////////////////////////////////////////////////////////////////////
    // Train the network
    //
    MinibatchTrainer<float, int> trainer(full_train_images, 0, target_training_labels, 0, minibatch_size);
    trainer.enable_shuffling(true);

    VariableF &train_input_mini = trainer.get_input_batch();
    const MatrixI &train_output_mini = trainer.get_output_batch_mat();
    cost_func.set_target_activations(train_output_mini);
    // Initialize the network:
    network.create_input_port(train_input_mini);
    network.forward();
    // Optionally load saved parameters:
    if (false) {
        network.load_parameters("cifar_model_1");
    }
    Updater updater(network.get_params());
    updater.set_mode_constant_learning_rate(learning_rate_weights); //
    updater.set_flag_weight_decay(weight_decay, enable_weight_decay);

    // For testing:
    MinibatchTrainer<float, int> tester(full_test_images, 0, target_testing_labels, 0, minibatch_size);
    tester.enable_shuffling(true); // Not necessary to shuffle the test set but does not hurt.
    VariableF &test_input_mini = tester.get_input_batch();
    const MatrixI &test_output_mini = tester.get_output_batch_mat();
    int train_epochs = 0;
    network.set_train_mode(true);

    Accumulator train_accumulator(minibatch_size);
    Accumulator test_accumulator(minibatch_size);
    Accumulator train_loss_accumulator(minibatch_size);
    Accumulator test_loss_accumulator(minibatch_size);
    // Vectors for plotting:
    vector<float> training_errors;
    vector<float> test_errors;
    Gnuplot plot_errors;
    Gnuplot plot_activations;
    while (train_epochs < 175) { // 150-200 epochs is good
        cerr << ".";
        bool end_epoch = trainer.next(); // Get next training mini-batch

        network.forward();
        train_loss_accumulator.accumulate(cost_func.get_output_data()[0]);
        train_accumulator.accumulate(error_count(linear_laye2.get_output_data(), train_output_mini));
        network.back_propagate();
        updater.update();
        if (end_epoch) {
            cout << endl << "---------------" << endl;
            //network.print_paramater_stats(); // enable for debugging info
            cout << "Training epochs: " << train_epochs << endl;
            cout << "Train error rate: " << train_accumulator.get_mean() << endl;
            cout << "Train loss/example: " << train_loss_accumulator.get_mean() << endl;
            cout << "Train examples: " << train_accumulator.get_counter() << endl;
            training_errors.push_back(train_accumulator.get_mean());
            train_accumulator.reset();
            train_loss_accumulator.reset();
            test_accumulator.reset();
            test_loss_accumulator.reset();
            network.set_train_mode(false);
            // Now connect the test input activations and deltas to the network.
            network.create_input_port(test_input_mini);
            cost_func.set_target_activations(test_output_mini);
            bool done = false;
            while (!done) {
                done = tester.next(); // Get next test mini-batch
                network.forward();
                test_loss_accumulator.accumulate(cost_func.get_output_data()[0]);
                test_accumulator.accumulate(error_count(linear_laye2.get_output_data(), test_output_mini));
            }
            cout << "Test error rate: " << test_accumulator.get_mean() << endl;
            cout << "Test loss/example: " << test_loss_accumulator.get_mean() << endl;
            cout << "Test examples: " << test_accumulator.get_counter() << endl;
            test_errors.push_back(test_accumulator.get_mean());
            // Connect the input activations and deltas to the network.
            network.create_input_port(train_input_mini);
            cost_func.set_target_activations(train_output_mini);
            // Display plots?
            if (true) {

                // Update plot:
                plot_errors << "set multiplot layout 2,1 title 'Train/Test errors'" << endl;
                plot(plot_errors, training_errors, "Training error");
                plot(plot_errors, test_errors, "Test error");
                plot_errors << "unset multiplot" << endl;

                // Plot some activations:
                plot_activations << "set multiplot layout 4,4 title 'Network activations'" << endl;

                // Plot an input image to network:
                const int image_index = 0; // arbitrary choice
                MatrixF one_image = select(test_input_mini.data, 0, image_index);
                scale(one_image, one_image, 255.0f);
                plot_image_rgb(plot_activations, one_image, "Input test image");

                //
                //for (int n = 1; n <= 13; ++n) {
                //MatrixF images = select(network.get_layer(n).get_output_forward(), 0, image_index);
                //plot_images_greyscale_3dim(plot_activations, images, network.get_layer(n).get_name());
                //}

                // Plot conv layer 1 output:
                MatrixF images_conv_layer_1_out = select(conv_layer1.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_1_out, conv_layer1.get_name());

                // Plot conv layer 2 output:
                MatrixF images_conv_layer_2_out = select(conv_layer2.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_2_out, conv_layer2.get_name());

                // Plot conv layer 3 output:
                MatrixF images_conv_layer_3_out = select(conv_layer3.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_3_out, conv_layer3.get_name());

                // Plot conv layer 4 output:
                MatrixF images_conv_layer_4_out = select(conv_layer4.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_4_out, conv_layer4.get_name());

                // Plot conv layer 5 output:
                MatrixF images_conv_layer_5_out = select(conv_layer5.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_5_out, conv_layer5.get_name());

                // Plot conv layer 6 output:
                MatrixF images_conv_layer_6_out = select(conv_layer6.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_6_out, conv_layer6.get_name());

                // Plot conv layer 7 output:
                MatrixF images_conv_layer_7_out = select(conv_layer7.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_7_out, conv_layer7.get_name());

                // Plot conv layer 8 output:
                MatrixF images_conv_layer_8_out = select(conv_layer8.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_8_out, conv_layer8.get_name());

                // Plot conv layer 9 output:
                MatrixF images_conv_layer_9_out = select(conv_layer9.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_9_out, conv_layer9.get_name());

                // Plot conv layer 10 output:
                MatrixF images_conv_layer_10_out = select(conv_layer10.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_10_out, conv_layer10.get_name());

                // Plot conv layer 11 output:
                MatrixF images_conv_layer_11_out = select(conv_layer11.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_11_out, conv_layer11.get_name());

                // Plot conv layer 12 output:
                MatrixF images_conv_layer_12_out = select(conv_layer12.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_12_out, conv_layer12.get_name());

                // Plot conv layer 13 output:
                MatrixF images_conv_layer_13_out = select(conv_layer13.get_output_data(), 0, image_index);
                plot_images_greyscale_3dim(plot_activations, images_conv_layer_13_out, conv_layer13.get_name());


                // Plot linear layer 1:
                MatrixF linear_laye1_out = select(linear_laye1.get_output_data(), 1, image_index);
                plot_image_greyscale(plot_activations, linear_laye1_out, linear_laye1.get_name());

                // Plot linear layer 2:
                MatrixF linear_laye2_out = select(linear_laye2.get_output_data(), 1, image_index);
                plot_image_greyscale(plot_activations, linear_laye2_out, linear_laye2.get_name());

                //

                plot_activations << "unset multiplot" << endl;
            }

            network.set_train_mode(true);
        }


        if (end_epoch) {
            // This gets called at the end of each epoch.
            train_epochs++;
            // Reduce the learning rate:
            learning_rate_weights *= 0.97f;
            cout << "New learning rate weights: " << learning_rate_weights << endl;
            updater.set_mode_constant_learning_rate(learning_rate_weights);
            learning_rate_bias *= 0.97f;
            cout << "New learning rate bias: " << learning_rate_bias << endl;
            weight_decay *= 0.97f;
            cout << "New weight decay: " << weight_decay << endl;
            updater.set_flag_weight_decay(weight_decay, enable_weight_decay);
        }
    }


    /////////////////////////////////////////////////////////////////////////////////////////
    // Save parameters and stats.
    network.save_parameters("cifar_model_1");

}


}
