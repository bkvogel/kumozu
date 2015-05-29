#ifndef _MINIBATCHNETWORKTRAINER_H
#define _MINIBATCHNETWORKTRAINER_H
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


#include "MatrixT.h"
#include <string>
#include <iostream>
#include "Constants.h"
#include "Utilities.h"
#include "Network.h"

namespace kumozu {

	/*
	 * A Utility class to train a feed-forward network (that is, an instance of "Network") in mini-batches.
	 *
	 * This class can be used to train a feed-forward network that has 2D (greyscale) or 3D (color) iamges as
	 * input and has a single class label as the output.
	 *
	 * Usage:
	 *
	 * First create an instance of this class, supplying the mini-batch size and a reference to an existing network
	 * to the constructor.
	 *
	 * Optionally call various methods to configure aspects of how the training will be performed.
	 *
	 * Call the train() method to perform the training.
	 *
	 * Call save() to save information such as training and test/validation error vs learning epoch to a file.
	 */
	class MinibatchNetworkTrainer {

	public:

		/*
		 * Create a new instance that will be associated with the supplied network.
		 *
		 * Parameters:
		 *
		 * network: an instance of a derived class of Network.
		 *
		 * train_input: Training images supplied to input activations of network. Must be either
		 *             size (training_images_count, image_height, image_width) or
		 *             size (training_images_count, image_depth, image_height, image_width).
		 *
		 * test_input: Test images supplied to input activations of network. Must be either
		 *             size (test_images_count, image_height_image, width) or
		 *             size (test_images_count, image_depth, image_height, image_width).
		 *
		 */
	MinibatchNetworkTrainer(Network& network, const Matrix& train_input,
							const Matrix& test_input,
							const std::vector<float>& train_output_labels,
							const std::vector<float>& test_output_labels): // todo: float -> int vector/Matrix.
		m_network {network},
			m_input_activations_mini{network.get_input_extents()},
			m_output_activations_mini  {network.get_output()},
			m_train_input {train_input},
					m_test_input {test_input},
							m_train_output_labels {train_output_labels},
								m_test_output_labels {test_output_labels},
									m_learn_reps {150}
		{
			/*
			std::cout << "MinibatchNetworkTrainer:" << std::endl;
			std::cout << "m_input_activations_mini.size() = " << m_input_activations_mini.size() << std::endl;
			std::cout << "m_input_activations_mini.order() = " << m_input_activations_mini.order() << std::endl;
			std::cout << "m_input_activations_mini.extent(0) = " << m_input_activations_mini.extent(0) << std::endl;
			std::vector<int> x = network.get_input_extents();
			std::cout << "x.size() = " << x.size() << std::endl;
			std::cout << "x.at(0) = " << x.at(0) << std::endl;
			std::cout << "x.at(1) = " << x.at(1) << std::endl;
			std::cout << "x.at(2) = " << x.at(2) << std::endl;
			exit(1);
			*/
		}

		/*
		 * Set the maximum number of epochs through the training set
		 * that will be performed when train() is called.
		 */
		void set_max_learn_epochs(int max_epochs);

		/*
		 * Train the network.
		 */
		void train();

		/*
		 * Save training and test error to a file withe the prefix given
		 * by the supplied name.
		 */
		void save_learning_info(std::string name) const;

	private:

		Network& m_network;
		Matrix m_input_activations_mini;
		Matrix& m_output_activations_mini;
		const Matrix& m_train_input;
		const Matrix& m_test_input;
		const std::vector<float>& m_train_output_labels; // fixme float->int
		const std::vector<float>& m_test_output_labels; // fixme float->int
		std::vector<float> m_log_training_errors;
		std::vector<float> m_log_test_errors;
		int m_learn_reps;
	};

}


#endif /* _MINIBATCHNETWORKTRAINER_H */
