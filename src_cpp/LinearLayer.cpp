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
#include "LinearLayer.h"
#include "Utilities.h"
#include "MatrixIO.h"
using namespace std;

namespace kumozu {

	void LinearLayer::set_gradient_check_mode() {
		randomize_normal(m_W, 0.0f, 1.0f);
		randomize_normal(m_bias, 0.0f, 1.0f);
	}

	// deprecated
	/*
	void LinearLayer::forward_propagate(const Matrix& input_activations, Matrix& output_activations) {
		// m_Z = m_W * m_input_activations + m_bias
		do_product_update(output_activations, m_W, input_activations, m_bias); 
	}
	*/

	void LinearLayer::forward_propagate(const Matrix& input_activations) {
		// m_Z = m_W * m_input_activations + m_bias
		do_product_update(m_output_activations, m_W, input_activations, m_bias); 
	}

	void LinearLayer::back_propagate_weight_gradients(const Matrix& output_error, const Matrix& input_activations) {
		//compute_reverse_relu(m_Z_error, output_error, m_output_activations_indices);
		// Compute grad_W = m_Z_error*m_input_activations^T
		compute_weight_grad_sgd_minibatch(output_error, m_W_grad, input_activations);
		compute_bias_grad_sgd_minibatch(output_error, m_bias_grad);

		//cout << "output_error = " << output_error << endl;
		//cout << "input_activations = " << input_activations << endl;
		//cout << "m_W_grad = " << m_W_grad << endl;

	}


	void LinearLayer::update_weights() {
		// Update m_W, m_bias in:
		// m_Z = m_W * m_input_activations + m_bias

		const float grad_scale = 1.0f/static_cast<float>(m_minibatch_size);
		// Now scale the learning rate by this factor.
		const float scaled_learning_rate_weights = grad_scale*m_learning_rate;
		const float scaled_learning_rate_bias = grad_scale*m_bias_learning_rate;
		
		m_log_weight_max_val.push_back(max_value(m_W)); // for logging
		m_log_weight_min_val.push_back(min_value(m_W)); // for logging
		copy_matrix(m_temp_size_W, m_W); // for logging.
		// Include weight decay, L1 sparsity, non-negative choice.
		//update_weights_from_gradient(m_W, m_W_grad, scaled_learning_rate_weights, m_lambda, m_sparsity_param, m_force_nonnegative); // fixme: enable after debug.
		//update_weights_from_gradient(m_W, m_W_grad, scaled_learning_rate_weights);
		//update_weights_from_gradient_rmsprop_v2(m_W, m_W_grad, m_sum_square_grad_W, m_rmsprop_learning_rate, m_rms_counter);
		float rho = 0.9f;
		float momentum = 0.9f;
		//update_weights_from_gradient_rmsprop_v3(m_W, m_W_grad, m_sum_square_grad_W, m_rmsprop_learning_rate, rho); // works well
		update_weights_from_gradient_rmsprop_momentum(m_W, m_W_grad, m_sum_square_grad_W, m_momentum_W, m_rmsprop_learning_rate, rho, momentum); // works well

		//update_weights_from_gradient_rmsprop_momentum_1d_kernel_ball(m_W, m_W_grad, m_sum_square_grad_W, m_momentum_W, m_rmsprop_learning_rate, rho, momentum);
		//update_weights_from_gradient_adagrad(m_W, m_W_grad, m_sum_square_grad_W, m_rmsprop_learning_rate);

		element_wise_difference(m_temp_size_W, m_temp_size_W, m_W); // for logging
		m_log_weight_updates.push_back(max_value(m_temp_size_W)); // for logging
		m_log_bias_max_val.push_back(max_value(m_bias)); // for logging
		m_log_bias_min_val.push_back(min_value(m_bias)); // for logging

		copy_matrix(m_temp_size_bias, m_bias); // for logging
		// Update bias:
		//compute_moving_average_mean_square(m_bias_rms_mean_square, m_bias_grad, m_rms_grad_keep_weight);
		//update_weights_from_gradient(m_bias, m_bias_grad, scaled_learning_rate_bias);
		//update_weights_from_gradient_rmsprop_v2(m_bias, m_bias_grad, m_sum_square_grad_bias, m_rmsprop_learning_rate, m_rms_counter);
		//update_weights_from_gradient_rmsprop_v3(m_bias, m_bias_grad, m_sum_square_grad_bias, m_rmsprop_learning_rate, rho); // works well
		update_weights_from_gradient_rmsprop_momentum(m_bias, m_bias_grad, m_sum_square_grad_bias, m_momentum_bias, m_rmsprop_learning_rate, rho, momentum);
		//update_weights_from_gradient_adagrad(m_bias, m_bias_grad, m_sum_square_grad_bias, m_rmsprop_learning_rate);
		element_wise_difference(m_temp_size_bias, m_temp_size_bias, m_bias); // for logging
		m_log_bias_updates.push_back(max_value(m_temp_size_bias)); // for logging

		++m_rms_counter;
	}

	// deprecated
	/*
	void LinearLayer::back_propagate_deltas(Matrix& input_error, const Matrix& output_error) {
		
		do_backprop_update_sgd_minibatch(output_error, m_W, input_error); // Usual backpropagation
		
		// OR

		// New method: Use fixed random matrix to back-propagate the errors.
		//do_backprop_update_sgd_minibatch_amp(m_Z_error, m_W_fixed_random, m_input_error); 
	}
	*/

	void LinearLayer::back_propagate_deltas(Matrix& input_error) {
		
		do_backprop_update_sgd_minibatch(m_output_error, m_W, input_error); // Usual backpropagation
		
		// OR

		// New method: Use fixed random matrix to back-propagate the errors.
		//do_backprop_update_sgd_minibatch_amp(m_Z_error, m_W_fixed_random, m_input_error); 
	}

	/*
	void LinearLayer::reverse_activation() {
		// Perform a "reverse maxout" to update the errors in m_Z_error.
		// Update m_Z_error in:
		// (m_output_error, m_output_activations_indices) = kmax(m_Z_error)
		compute_reverse_relu(m_Z_error, m_output_error, m_output_activations_indices);
		//compute_reverse_leaky_relu(m_Z_error, m_output_error, m_output_activations_indices);
	}
	*/

	Matrix& LinearLayer::get_weights() {
		return m_W;
	}

	Matrix& LinearLayer::get_bias() {
		return m_bias;
	}

	Matrix& LinearLayer::get_weight_gradient() {
		return m_W_grad;
	}

	Matrix& LinearLayer::get_bias_gradient() {
		return m_bias_grad;
	}

	void LinearLayer::save_learning_info(std::string name) const {
		save_matrix(m_W, name + "_" + m_layer_name + "_W.dat");
		save_vector(m_log_weight_updates, name + "_" + m_layer_name + "_weight_updates.dat");
		save_vector(m_log_weight_max_val, name + "_" + m_layer_name + "_weight_max_val.dat");
		save_vector(m_log_weight_min_val, name + "_" + m_layer_name + "_weight_min_val.dat");
		save_vector(m_log_bias_updates, name + "_" + m_layer_name + "_bias_updates.dat");
		save_vector(m_log_bias_max_val, name + "_" + m_layer_name + "_bias_max_val.dat");
		save_vector(m_log_bias_min_val, name + "_" + m_layer_name + "_bias_min_val.dat");
	}

	float LinearLayer::compute_cost_function(const Matrix& input_activations, const Matrix& true_output_activations) {
		//forward_propagate(input_activations, m_output_activations); 
		forward_propagate(input_activations); 
		// Compute error matrix:
		// m_output_error = -(true_output_activations - m_output_activations)
		element_wise_difference(m_output_error, m_output_activations, true_output_activations);
		float cost = 0.0f;
		for (int i = 0; i != m_output_error.size(); ++i) {
			cost += m_output_error[i]*m_output_error[i];
		}
		cost *= 0.5;
		return cost;
	}

}
