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
#include "ConvLayer2D.h"
#include "Utilities.h"
#include "MinibatchNetworkTrainer.h"
#include "MatrixIO.h"

using namespace std;

namespace kumozu {

	void MinibatchNetworkTrainer::set_max_learn_epochs(int max_epochs) {
		m_learn_reps = max_epochs;
	}

	void MinibatchNetworkTrainer::train() {


		const int training_examples_count = m_train_input.extent(0);
		cout << "Number of training examples = " << training_examples_count << endl;
	
		// Size is "training label count" x "number of training examples".
		const int class_label_count = m_network.get_output().extent(0);
		const int minibatch_size = m_input_activations_mini.extent(0);
		const int minibatch_count_train = training_examples_count / minibatch_size;
		const int minibatch_remainder_train = training_examples_count % minibatch_size;
		if (minibatch_remainder_train != 0) {
			cerr << "Error: nonzero mini-batch remainder for training set!" << endl;
			cerr << "Remainder = " << minibatch_remainder_train << endl;
			cerr << "Mini-batch size = " << minibatch_size << endl;
			exit(1);
		}
		cout << "minibatch_count_train = " << minibatch_count_train << endl;
		cout << "minibatch_remainder_train = " << minibatch_remainder_train << endl;

		// Training labels for 1 mini-batch only.
		Matrix train_labels_mini(class_label_count, minibatch_size);

		const int test_examples_count = m_test_input.extent(0);
		const int minibatch_remainder_test = test_examples_count % minibatch_size;
		cout << "Test examples = " << test_examples_count << endl;
		if (minibatch_remainder_test != 0) {
			cerr << "Error: nonzero mini-batch remainder for test set!" << endl;
			exit(1);
		}
		const int minibatch_count_test = test_examples_count / minibatch_size;
		Matrix test_output(class_label_count, test_examples_count);
		Matrix train_output(class_label_count, training_examples_count);
		// Size is "training label count" x "number of training examples".
		Matrix training_labels_full = labels_to_mat(m_train_output_labels);

		int learning_examples_counter = 0;
		//const int learn_reps{ 150 }; // 100
		for (int i = 0; i < m_learn_reps; i++) {
			cout << "---------------------" << endl;
			cout << "Full data iteration " << i << endl;
			cout << "Learning examples = " << learning_examples_counter << endl;
			for (int minibatch_index = 0; minibatch_index < minibatch_count_train; ++minibatch_index) {
				int minibatch_start_col = minibatch_index*minibatch_size;
				//////////////////////////////
				// Training set:

				// Copy training images for current mini_batch into input_activations_mini.
				extract_3d_minibatch(m_input_activations_mini, m_train_input, minibatch_start_col, minibatch_size);

				// Updates output_activations_mini.
				m_network.forward_propagate(m_input_activations_mini);
				return_sub_matrix(m_output_activations_mini, train_output, minibatch_start_col); // Return predicted class labels.
				// Extract mini-batch of training inputs and outputs:
				get_sub_matrix(train_labels_mini, training_labels_full, minibatch_start_col); // Copy training labels into mini-batch
				//////////////////////////////
				// Backward pass on training set:
				m_network.back_propagate_gradients(m_input_activations_mini, train_labels_mini);
				m_network.update_weights();
				learning_examples_counter += minibatch_size;
			}
			const float train_error = compute_test_error(train_output, m_train_output_labels);
			cout << "Training error = " << train_error << endl;
			m_log_training_errors.push_back(train_error);

			for (int minibatch_index = 0; minibatch_index < minibatch_count_test; ++minibatch_index) {
				int minibatch_start_col = minibatch_index*minibatch_size;
				/////////////////////////////////////////////////////////////////////////////////////////////
				// Forward pass on test set:
				extract_3d_minibatch(m_input_activations_mini, m_test_input, minibatch_start_col, minibatch_size);

				// Updates output_activations_mini.
				m_network.disable_dropout(); 
				m_network.forward_propagate(m_input_activations_mini);
				m_network.enable_dropout();
				return_sub_matrix(m_output_activations_mini, test_output, minibatch_start_col); // Return predicted class labels.
				
			}
			const float test_error = compute_test_error(test_output, m_test_output_labels);
			cout << "Test error = " << test_error << endl;
			m_log_test_errors.push_back(test_error);

		}

	}

	void MinibatchNetworkTrainer::save_learning_info(std::string name) const {
		string train_file = name + "_train_errors.dat";
		save_vector(m_log_training_errors, train_file);
		string test_file = name + "_test_errors.dat";
		save_vector(m_log_test_errors, test_file);
	}

}
