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

#include "ExamplesNeuralNet.h"
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

#include "MatrixT.h"
#include "MatrixIO.h"
#include "UnitTests.h"
#include "Network2DConv3F1.h"
#include "Network3DConv3F1.h"
#include "MinibatchNetworkTrainer.h"

// Uncomment following line to disable assertion checking.
//#define NDEBUG
// Uncomment to enable assertion checking.
#undef NDEBUG
#include <assert.h>

using namespace std;

namespace kumozu {

	
	void mnist_example_1() {
		cout << "MNIST Example 1" << endl << endl;
		/////////////////////////////////////////////////////////////////////////////////////////
		// Load training and test data
		//

		Matrix full_train_images = load_matrix("training_images.dat");
		Matrix test_images_full = load_matrix("testing_images.dat");
		vector<float> true_training_labels = load_matrix("array_training_labels.dat"); // fixme: float -> int 
		vector<float> true_testing_labels = load_matrix("array_testing_labels.dat"); 

		/////////////////////////////////////////////////////////////////////////////////////////
		// Set parameters for each layer of the network.
		//
		
		const int conv_filter_height1 = 5; // 9
		const int conv_filter_width1 = 5; // 9
		const int filter_count1 = 32; // 64 // Number of convolutional filters.
		// For pooling layer 1
		const vector<int> pooling_region_extents1 = {1, 3, 3};
		// (depth, height, width)
		const vector<int> pooling_output_extents1 = {filter_count1, 14, 14};

		const int conv_filter_height2 = 5; // 5
		const int conv_filter_width2 = 5; // 5
		//const int filter_count2 = 4*32; // 128
		const int filter_count2 = 2*32; // 128
		// For pooling layer 2
		const vector<int> pooling_region_extents2 = {1, 3, 3};
		// (depth, height, width)
		const vector<int> pooling_output_extents2 = {filter_count2, 14, 14};

		const int conv_filter_height3 = 3; // 5
		const int conv_filter_width3 = 3; // 5
		//const int filter_count3 = 4*4*32; // 128
		const int filter_count3 = 3*32; // 128

		// For pooling layer 3
		const vector<int> pooling_region_extents3 = {1, 3, 3};
		// (depth, height, width)
		const vector<int> pooling_output_extents3 = {filter_count3, 7, 7};

		const int dim_fully_connected_hidden = 512;

		// Amount of dropout for each layer. Expresssed as probability of keeping an activation.
		//const vector<float> dropout_keep_probabilities = {1.0f, 0.9f, 0.8f, 0.5f};
		const vector<float> dropout_keep_probabilities = {1.0f, 1.0f, 1.0f, 1.0f}; // Get it to overfit without dropout first.
		const int maxout_factor = 1;
		// Number of class labels. This is the number of digits(0 - 9).
		const int class_label_count = 1 + static_cast<int>(*max_element(true_testing_labels.begin(), true_testing_labels.end()));  
		const int image_height = full_train_images.extent(1); // Height of MNIST image
		const int image_width = full_train_images.extent(2); // Width of MNIST image
		const int minibatch_size = 50; // try 32-256
		const vector<int> input_extents = {minibatch_size, image_height, image_width};
		Network2DConv3F1 network(input_extents, 
						  filter_count1, conv_filter_height1, conv_filter_width1, 
										pooling_region_extents1, pooling_output_extents1,
									filter_count2, conv_filter_height2, conv_filter_width2, 
										pooling_region_extents2, pooling_output_extents2,
										  filter_count3, conv_filter_height3, conv_filter_width3, 
										pooling_region_extents3, pooling_output_extents3,
										class_label_count,
										  dim_fully_connected_hidden, maxout_factor,
										BoxActivationFunction::ACTIVATION_TYPE::leakyReLU, 
								 ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU,
										dropout_keep_probabilities);
		
		/////////////////////////////////////////////////////////////////////////////////////////
		// Train the network
		//
		MinibatchNetworkTrainer trainer(network, full_train_images, test_images_full, true_training_labels, 
										true_testing_labels);
		trainer.train();

		/////////////////////////////////////////////////////////////////////////////////////////
		// Save parameters and stats.

		// Save stats on learning progress.
		trainer.save_learning_info("mnist_example");

		// Save learned parameters
		network.save_learning_info("mnist_example");
	}



	void cifar10_example_1() {
		cout << "CIFAR10 Example 1" << endl << endl;
		/////////////////////////////////////////////////////////////////////////////////////////
		// Load training and test data
		//
		Matrix full_train_images = load_matrix("training_images.dat");
		Matrix test_images_full = load_matrix("testing_images.dat");
		vector<float> true_training_labels = load_matrix("array_training_labels.dat"); // fixme: float -> int
		vector<float> true_testing_labels = load_matrix("array_testing_labels.dat"); //

		/////////////////////////////////////////////////////////////////////////////////////////
		// Set parameters for each layer of the network.
		//
		
		const int minibatch_size = 20; // try 32-256.

		const int conv_filter_height1 = 5; // 9
		const int conv_filter_width1 = 5; // 9
		const int filter_count1 = 2*32; // 64 // Number of convolutional filters.
		// For pooling layer 1
		const vector<int> pooling_region_extents1 = {1, 3, 3};
		// (depth, height, width)
		const vector<int> pooling_output_extents1 = {filter_count1, 16, 16};

		const int conv_filter_height2 = 5; // 5
		const int conv_filter_width2 = 5; // 5
		//const int filter_count2 = 4*32; // 128
		const int filter_count2 = 2*2*32; // 128
		// For pooling layer 2
		const vector<int> pooling_region_extents2 = {1, 3, 3};
		// (depth, height, width)
		const vector<int> pooling_output_extents2 = {filter_count2, 8, 8};

		const int conv_filter_height3 = 5; // 5
		const int conv_filter_width3 = 5; // 5
		//const int filter_count3 = 4*4*32; // 128
		const int filter_count3 = 2*3*32; // 128
		// For pooling layer 3
		const vector<int> pooling_region_extents3 = {1, 3, 3};
		// (depth, height, width)
		const vector<int> pooling_output_extents3 = {filter_count3, 4, 4};

		const int dim_fully_connected_hidden = 512; // 512
		const int maxout_factor = 1; // Set to 1 unless using maxout activations for fully-connected layers.
		
		// Amount of dropout for each layer. Expresssed as probability of keeping an activation.
		const vector<float> dropout_keep_probabilities = {1.0f, 0.95f, 0.9f, 0.8f, 0.5f};
		//const vector<float> dropout_keep_probabilities = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
		
		const int class_label_count = 1 + static_cast<int>(*max_element(true_testing_labels.begin(), true_testing_labels.end()));  
		const int image_depth = full_train_images.extent(1); // Width of CIFAR image
		const int image_height = full_train_images.extent(2); // Height of CIFAR image
		const int image_width = full_train_images.extent(3); // Width of CIFAR image
		const vector<int> input_extents = {minibatch_size, image_depth, image_height, image_width};
		Network3DConv3F1 network(input_extents, 
						  filter_count1, conv_filter_height1, conv_filter_width1, 
										pooling_region_extents1, pooling_output_extents1,
									filter_count2, conv_filter_height2, conv_filter_width2, 
										pooling_region_extents2, pooling_output_extents2,
										  filter_count3, conv_filter_height3, conv_filter_width3, 
										pooling_region_extents3, pooling_output_extents3,
										class_label_count,
										  dim_fully_connected_hidden, maxout_factor,
										BoxActivationFunction::ACTIVATION_TYPE::leakyReLU, 
								 ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU,
										dropout_keep_probabilities);

		
		/////////////////////////////////////////////////////////////////////////////////////////
		// Train the network
		//
		MinibatchNetworkTrainer trainer(network, full_train_images, test_images_full, true_training_labels, 
										true_testing_labels);

		// Set number of iterations through training set.
		trainer.set_max_learn_epochs(300); 
		trainer.train();

		/////////////////////////////////////////////////////////////////////////////////////////
		// Save parameters and stats.

		// Save stats on learning progress.
		trainer.save_learning_info("cifar10_example");

		// Save learned parameters
		network.save_learning_info("cifar10_example");

	}




}
