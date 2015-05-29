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
#include "Network3DConv3F1.h"
#include "Utilities.h"
using namespace std;

namespace kumozu {

	void Network3DConv3F1::forward_propagate(const Matrix& input_activations) {
		//////////////////////////////////////////////////////////////////////////////////
		// Convolutional layer 1.

		// Dropout
		m_dropout_conv_layer0.forward_dropout(input_activations);
		// Linear convolutional layer.
		m_conv_layer1.forward_propagate(m_dropout_conv_layer0.get_output());
		// Dropout
		m_dropout_conv_layer1.forward_dropout(m_conv_layer1.get_output());
		// Non-linear activation.
		m_box_activation1.forward_activation(m_dropout_conv_layer1.get_output());
		// Pooling layer.
		m_pooling_layer1.forward_pool(m_box_activation1.get_output());
		//////////////////////////////////////////////////////////////////////////////////
		// Convolutional layer 2.

		// Linear convolutional layer.
		m_conv_layer2.forward_propagate(m_pooling_layer1.get_output());
		// Dropout.
		m_dropout_conv_layer2.forward_dropout(m_conv_layer2.get_output());
		// Non-linear activation.
		m_box_activation2.forward_activation(m_dropout_conv_layer2.get_output());
		// Pooling layer.
		m_pooling_layer2.forward_pool(m_box_activation2.get_output());
		//////////////////////////////////////////////////////////////////////////////////
		// Convolutional layer 3.

		// Linear convolutional layer.
		m_conv_layer3.forward_propagate(m_pooling_layer2.get_output()); 
		// Dropout.
		m_dropout_conv_layer3.forward_dropout(m_conv_layer3.get_output());
		// Non-linear activation.
		m_box_activation3.forward_activation(m_dropout_conv_layer3.get_output());
		// Pooling layer.
		m_pooling_layer3.forward_pool(m_box_activation3.get_output());
		//////////////////////////////////////////////////////////////////////////////////
		// Fully-connected layer 1.

		// Reformat 3D (i.e., box) activations from the convolutional layer to be compatible with fully-connected layer.
		multi_dim_minibatch_to_column_minibatch(m_full_layer1_input, m_pooling_layer3.get_output());
		// Linear fully-connected layer.
		m_full_layer1.forward_propagate(m_full_layer1_input);
		// Apply dropout
		m_dropout_full_layer1.forward_dropout(m_full_layer1.get_output());
		// Non-linear activation.
		m_full_activation_func1.forward_activation(m_dropout_full_layer1.get_output());
		//////////////////////////////////////////////////////////////////////////////////
		// Fully-connected layer 2.

		// Linear fully-connected layer.
		m_full_layer2.forward_propagate(m_full_activation_func1.get_output());
		// Non-linear activation for layer.
		m_full_activation_func2.forward_activation(m_full_layer2.get_output());
	}

	void Network3DConv3F1::back_propagate_gradients(const Matrix& input_activations, 
													 const Matrix& true_output_activations) {
		// Update gradients for all parameters but do not update parameter values.
		
		// Compute errors:
		// output_error = -(true_output_activations - output_activations)
		element_wise_difference(m_full_activation_func2.get_output_deltas(), m_full_activation_func2.get_output(), 
							  true_output_activations);
		//////////////////////////////////////////////////////////////////////////////////
		// Fully-connected layer 2.

		// Reverse non-linear activation function.
		m_full_activation_func2.reverse_activation(m_full_layer2.get_output_deltas()); 
		// Back-prop deltas.
		m_full_layer2.back_propagate_deltas(m_full_activation_func1.get_output_deltas());
		// Back-prop compute weight gradients.
		// fixme: deprecated
		m_full_layer2.back_propagate_weight_gradients(m_full_layer2.get_output_deltas(), m_full_activation_func1.get_output());
		
		//////////////////////////////////////////////////////////////////////////////////
		// Fully-connected layer 1.

		// Reverse non-linear activation function.
		m_full_activation_func1.reverse_activation(m_dropout_full_layer1.get_output_deltas());
		// Reverse dropout.
		m_dropout_full_layer1.reverse_dropout(m_full_layer1.get_output_deltas());
		// Back-prop deltas.
		m_full_layer1.back_propagate_deltas(m_full_layer1_input_error); 
		// Back-prop compute weight gradients.
		m_full_layer1.back_propagate_weight_gradients(m_full_layer1.get_output_deltas(), m_full_layer1_input);

		// Reformat each column input vector from the FC layer into a 3D box of activations for the conv layer.
		column_minibatch_to_multi_dim_minibatch(m_full_layer1_input_error, m_pooling_layer3.get_output_deltas());

		//////////////////////////////////////////////////////////////////////////////////
		// Convolutional layer 3.

		// Reverse pooling layer.
		m_pooling_layer3.reverse_pool(m_box_activation3.get_output_deltas());
		// Reverse non-linear activation function.
		m_box_activation3.reverse_activation(m_dropout_conv_layer3.get_output_deltas());
		// Reverse dropout.
		m_dropout_conv_layer3.reverse_dropout(m_conv_layer3.get_output_deltas());
		// Back-prop deltas.
		m_conv_layer3.back_propagate_deltas(m_pooling_layer2.get_output_deltas());
		// Back-prop compute weight gradients.
		m_conv_layer3.back_propagate_weight_gradients(m_conv_layer3.get_output_deltas(), m_pooling_layer2.get_output()); 

		//////////////////////////////////////////////////////////////////////////////////
		// Convolutional layer 2.

		// Reverse pooling layer.
		m_pooling_layer2.reverse_pool(m_box_activation2.get_output_deltas());
		// Reverse non-linear activation function.
		m_box_activation2.reverse_activation(m_dropout_conv_layer2.get_output_deltas());
		// Reverse dropout.
		m_dropout_conv_layer2.reverse_dropout(m_conv_layer2.get_output_deltas());
		// Back-prop deltas.
		m_conv_layer2.back_propagate_deltas(m_pooling_layer1.get_output_deltas());
		// Back-prop compute weight gradients.
		m_conv_layer2.back_propagate_weight_gradients(m_conv_layer2.get_output_deltas(), m_pooling_layer1.get_output()); 

		//////////////////////////////////////////////////////////////////////////////////
		// Convolutional layer 1.

		// Reverse pooling layer.
		m_pooling_layer1.reverse_pool(m_box_activation1.get_output_deltas());
		// Reverse non-linear activation function.
		m_box_activation1.reverse_activation(m_dropout_conv_layer1.get_output_deltas());
		// Reverse dropout.
		m_dropout_conv_layer1.reverse_dropout(m_conv_layer1.get_output_deltas());
		// Back-prop compute weight gradients.
		m_conv_layer1.back_propagate_weight_gradients(m_conv_layer1.get_output_deltas(), m_dropout_conv_layer0.get_output());
	}

	void Network3DConv3F1::back_propagate_deltas(Matrix& input_activations_deltas, 
													 const Matrix& true_output_activations)  {
		back_propagate_gradients(input_activations_deltas, true_output_activations); // the weight gradient updates are not needed but cause no harm.
		// Display error:
		cout << "back-prop deltas output rmse = " << compute_rmse(m_full_activation_func2.get_output_deltas()) << endl;

		// Back-prop deltas to input layer.
		m_conv_layer1.back_propagate_deltas(input_activations_deltas);
	}

	float Network3DConv3F1::compute_cost_function(const Matrix& input_activations, const Matrix& true_output_activations) {
		forward_propagate(input_activations); 
		// Compute error matrix:
		// m_output_error = -(true_output_activations - m_output_activations)
		Matrix& output_error = m_full_activation_func2.get_output_deltas();
		Matrix& output = m_full_activation_func2.get_output();
		element_wise_difference(output_error, output, true_output_activations);
		float cost = 0.0f;
		for (int i = 0; i != output_error.size(); ++i) {
			cost += output_error[i]*output_error[i];
		}
		cost *= 0.5;
		return cost;
	}


	void Network3DConv3F1::update_weights() {
		m_full_layer2.update_weights();
		m_full_layer1.update_weights();
		m_conv_layer3.update_weights();
		m_conv_layer2.update_weights();
		m_conv_layer1.update_weights();
	}

	/*
	 * Save training and test error to a file withe the prefix given
	 * by the supplied name.
	 */
	void Network3DConv3F1::save_learning_info(std::string name) const {
		m_full_layer2.save_learning_info(name);
		m_full_layer1.save_learning_info(name);
		m_conv_layer3.save_learning_info(name);
		m_conv_layer2.save_learning_info(name);
		m_conv_layer1.save_learning_info(name);
	}

	void Network3DConv3F1::load_parameters(std::string name) const {
		// fixme
	}

	bool Network3DConv3F1::check_gradients(const Matrix& input_activations, const Matrix& true_output_activations) {
		// Updates output_activations.
		forward_propagate(input_activations); 

		// Update all deltas and weight/bias gradients.
		back_propagate_gradients(input_activations, true_output_activations);

		// Now numerically check against the back-propagated gradients.

		// Check weight gradient from fully-connected layer 2.
		cout << "Checking gradients in fully-connected layer 2 grad_W..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_full_layer2.get_weight_gradient(),
									m_full_layer2.get_weights());
		cout << endl;
		
		// Check bias gradient from fully-connected layer 2.
		cout << "Checking gradients in fully-connected layer 2 grad_bias..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_full_layer2.get_bias_gradient(),
									m_full_layer2.get_bias());
		cout << endl;

		// Check weight gradient from fully-connected layer 1.
		cout << "Checking gradients in fully-connected layer 1 grad_W..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_full_layer1.get_weight_gradient(),
									m_full_layer1.get_weights());
		cout << endl;

		// Check bias gradient from fully-connected layer 1.
		cout << "Checking gradients in fully-connected layer 1 grad_bias..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_full_layer1.get_bias_gradient(),
									m_full_layer1.get_bias());
		cout << endl;

		// Check weight gradient from conv layer 3.
		cout << "Checking gradients in convolutional layer 3 grad_W..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_conv_layer3.get_weight_gradient(),
									m_conv_layer3.get_weights());
		cout << endl;

		// Check bias gradient from conv layer 3.
		cout << "Checking gradients in convolutional layer 3 grad_bias..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_conv_layer3.get_bias_gradient(),
									m_conv_layer3.get_bias());
		cout << endl;

		// Check weight gradient from conv layer 2.
		cout << "Checking gradients in convolutional layer 2 grad_W..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_conv_layer2.get_weight_gradient(),
									m_conv_layer2.get_weights());
		cout << endl;

		// Check bias gradient from conv layer 2.
		cout << "Checking gradients in convolutional layer 2 grad_bias..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_conv_layer2.get_bias_gradient(),
									m_conv_layer2.get_bias());
		cout << endl;

		// Check weight gradient from conv layer 1.
		cout << "Checking gradients in convolutional layer 1 grad_W..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_conv_layer1.get_weight_gradient(),
									m_conv_layer1.get_weights());
		cout << endl;

		// Check bias gradient from conv layer 1.
		cout << "Checking gradients in convolutional layer 1 grad_bias..." << endl;
		numerical_gradients(input_activations, true_output_activations, m_conv_layer1.get_bias_gradient(),
									m_conv_layer1.get_bias());
		cout << endl;
		return true;
	}




}
