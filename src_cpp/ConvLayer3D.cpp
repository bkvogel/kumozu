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
#include "ConvLayer3D.h"
#include "Utilities.h"
#include "MatrixIO.h"
namespace kumozu {

	void ConvLayer3D::set_gradient_check_mode() {
		randomize_normal(m_W, 0.0f, 1.0f);
		randomize_normal(m_bias, 0.0f, 1.0f);
	}

	void ConvLayer3D::forward_propagate(const Matrix& input_activations) {
		// Let's perform convolution on one mini-batch.
		// Compute m_Z = m_W (convolve with) m_input_activations + m_bias:

		// naive version
		//convolve_3d_filter_with_bias_minibatch(output_activations, m_W, m_bias, input_activations);

		// optimized version
		convolve_3d_filter_with_bias_minibatch_optimized(m_output_activations, m_W, m_bias, input_activations,
														 m_temp_Z2, m_temp_A1, m_temp_W);
	}

	void ConvLayer3D::back_propagate_weight_gradients(const Matrix& output_error, const Matrix& input_activations) {
		// naive version
		//compute_3d_weight_grad_convolutive_minibatch(m_W_grad, output_error, input_activations);

		// optimized version
		compute_3d_weight_grad_convolutive_minibatch_optimized(m_W_grad, output_error, input_activations,
															m_temp_Z2, m_temp_A1, m_temp_W);

		compute_bias_grad_convolutive_minibatch(m_bias_grad, output_error);
	}


	void ConvLayer3D::update_weights() {
		// Compute scale factor for the gradient to get mean gradient:

		// fixme: get the scaling right! If we use adaptive learning rate like Adagrad, don't need to worry about this.

		//const float grad_scale = 1.0f/(static_cast<float>(m_minibatch_size)*static_cast<float>(m_image_height)*static_cast<float>(m_image_width));
		//const float grad_scale = 1.0f/static_cast<float>(m_minibatch_size);// fixme: is this correct?

		const float grad_scale = 1.0f/static_cast<float>(m_minibatch_size*m_image_height*m_image_width);
		//std::cout << "m_minibatch_size*m_image_height*m_image_width = " << m_minibatch_size*m_image_height*m_image_width << std::endl;
		// Now scale the learning rate by this factor.
		const float scaled_learning_rate_weights = grad_scale*m_learning_rate;
		const float scaled_learning_rate_bias = grad_scale*m_bias_learning_rate;

		//std::cout << "scaled_learning_rate_weights = " << scaled_learning_rate_weights << std::endl;

		m_log_weight_max_val.push_back(max_value(m_W)); // for logging
		m_log_weight_min_val.push_back(min_value(m_W)); // for logging
		copy_matrix(m_temp_size_W, m_W); // for logging.
		// Include weight decay, L1 sparsity, non-negative choice.
		//update_weights_from_gradient(m_W, m_W_grad, scaled_learning_rate_weights, m_lambda, m_sparsity_param, m_force_nonnegative);
		//update_weights_from_gradient(m_W, m_W_grad, scaled_learning_rate_weights); // vanilla SGD
		
		
		//update_weights_from_gradient_rmsprop_v2(m_W, m_W_grad, m_sum_square_grad_W, m_rmsprop_learning_rate, m_rms_counter); // don't use
		float rho = 0.9f;
		float momentum = 0.9f;
		//update_weights_from_gradient_rmsprop_v3(m_W, m_W_grad, m_sum_square_grad_W, m_rmsprop_learning_rate, rho); // works well
		//update_weights_from_gradient_rmsprop_kernel_ball_1(m_W, m_W_grad, m_sum_square_grad_W, m_rmsprop_learning_rate, rho); 

		update_weights_from_gradient_rmsprop_momentum(m_W, m_W_grad, m_sum_square_grad_W, m_momentum_W, m_rmsprop_learning_rate, rho, momentum); // works well
		//update_weights_from_gradient_adagrad(m_W, m_W_grad, m_sum_square_grad_W, m_rmsprop_learning_rate);
		

		element_wise_difference(m_temp_size_W, m_temp_size_W, m_W); // for logging
		m_log_weight_updates.push_back(max_value(m_temp_size_W)); // for logging
		m_log_bias_max_val.push_back(max_value(m_bias)); // for logging
		m_log_bias_min_val.push_back(min_value(m_bias)); // for logging

		copy_matrix(m_temp_size_bias, m_bias); // for logging
		// Update bias:
		//update_weights_from_gradient(m_bias, m_bias_grad, scaled_learning_rate_bias);
		//update_weights_from_gradient_rmsprop_v2(m_bias, m_bias_grad, m_sum_square_grad_bias, m_rmsprop_learning_rate, m_rms_counter); // don't use
		update_weights_from_gradient_rmsprop_v3(m_bias, m_bias_grad, m_sum_square_grad_bias, m_rmsprop_learning_rate, rho); // works well
		//update_weights_from_gradient_rmsprop_momentum(m_bias, m_bias_grad, m_sum_square_grad_bias, m_momentum_bias, m_rmsprop_learning_rate, rho, momentum); // works well
		//update_weights_from_gradient_adagrad(m_bias, m_bias_grad, m_sum_square_grad_bias, m_rmsprop_learning_rate);
		element_wise_difference(m_temp_size_bias, m_temp_size_bias, m_bias); // for logging
		m_log_bias_updates.push_back(max_value(m_temp_size_bias)); // for logging

		++m_rms_counter;
	}

	void ConvLayer3D::back_propagate_deltas(Matrix& input_error) {
		// Update m_input_error.

		// naive version
		//compute_3d_convolutive_deltas_minibatch(input_error, m_W, m_output_error); 

		// optimized version
		compute_3d_convolutive_deltas_minibatch_optimized(input_error, m_W, m_output_error, m_temp_Z2, m_temp_A1, m_temp_W); 
	}


	Matrix& ConvLayer3D::get_weights() {
		return m_W;
	}


	Matrix& ConvLayer3D::get_bias() {
		return m_bias;
	}

	Matrix& ConvLayer3D::get_weight_gradient() {
		return m_W_grad;
	}

	Matrix& ConvLayer3D::get_bias_gradient() {
		return m_bias_grad;
	}

	void ConvLayer3D::save_learning_info(std::string name) const {
		save_matrix(m_W, name + "_" + m_layer_name + "_W.dat");
		save_vector(m_log_weight_updates, name + "_" + m_layer_name + "_weight_updates.dat");
		save_vector(m_log_weight_max_val, name + "_" + m_layer_name + "_weight_max_val.dat");
		save_vector(m_log_weight_min_val, name + "_" + m_layer_name + "_weight_min_val.dat");
		save_vector(m_log_bias_updates, name + "_" + m_layer_name + "_bias_updates.dat");
		save_vector(m_log_bias_max_val, name + "_" + m_layer_name + "_bias_max_val.dat");
		save_vector(m_log_bias_min_val, name + "_" + m_layer_name + "_bias_min_val.dat");
	}

	float ConvLayer3D::compute_cost_function(const Matrix& input_activations, const Matrix& true_output_activations) {
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
