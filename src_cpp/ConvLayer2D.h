#ifndef _CONVLAYER2D_H
#define _CONVLAYER2D_H
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


namespace kumozu {

	/*
	 * A convolutional layer that operates on a mini-batch of 2D greyscale images.
	 *
	 */
	class ConvLayer2D {

	public:

				/*
		 *
		 * Parameters:
		 *
		 * input_extents: Dimensions of the input activations, which are 
		 *                (minibatch_size, image_height, image_width)
		 *
		 * filter_count: Number of convolutional filters.
		 *
		 * conv_filter_height: Height of a convolutional filter.
		 *
		 * conv_filter_width: Width of a convolutional filter.
		 *
		 */
	ConvLayer2D(const std::vector<int>& input_extents, int filter_count, int conv_filter_height, 
				int conv_filter_width, std::string name) :
		m_layer_name {name},
		m_minibatch_size {input_extents.at(0)},
			m_image_height {input_extents.at(1)},
				m_image_width {input_extents.at(2)},
		m_output_activations {m_minibatch_size, filter_count, m_image_height, m_image_width},
			m_output_error {m_minibatch_size, filter_count, m_image_height, m_image_width},
				m_W(filter_count, conv_filter_height, conv_filter_width),
					m_temp_size_W(filter_count, conv_filter_height, conv_filter_width),
					m_sum_square_grad_W(filter_count, conv_filter_height, conv_filter_width),
					m_momentum_W(filter_count, conv_filter_height, conv_filter_width),
					m_W_grad(filter_count, conv_filter_height, conv_filter_width),
							m_bias(filter_count),
							m_temp_size_bias(filter_count),
					m_sum_square_grad_bias(filter_count),
					m_momentum_bias(filter_count),
							m_bias_grad(filter_count),
					m_temp_Z2 {m_image_height*m_image_width*m_minibatch_size, filter_count},
					m_temp_A1 {m_image_height*m_image_width*m_minibatch_size, conv_filter_height*conv_filter_width + 1},
						m_temp_W {conv_filter_height*conv_filter_width + 1, filter_count}
			 {



				 const float std_dev_init = 0.5*sqrt(2.0f/(conv_filter_height*conv_filter_width));
			std::cout << "ConvLayer2D std_dev_init = " << std_dev_init << std::endl;
			randomize_normal(m_W, 0.0f, std_dev_init);

			std::cout << "m_temp_A1 rows = " << m_temp_A1.extent(0) << std::endl;
			std::cout << "m_temp_A1 cols = " << m_temp_A1.extent(1) << std::endl;
			std::cout << "m_temp_W rows = " << m_temp_W.extent(0) << std::endl;
			std::cout << "m_temp_W cols = " << m_temp_W.extent(1) << std::endl;

			//randomize_normal(m_W, 0.0f, 1.0f); // for gradient checking, use larger values for the weights.
			//randomize_normal(m_W_fixed_random, 0.0f, r);
		}


		/*
		 *
		 * Parameters:
		 *
		 * input_extents: Dimensions of the input activations, which are 
		 *                (minibatch_size, image_height, image_width)
		 *
		 * filter_count: Number of convolutional filters.
		 *
		 * conv_filter_height: Height of a convolutional filter.
		 *
		 * conv_filter_width: Width of a convolutional filter.
		 *
		 */
			 // deprecated
	ConvLayer2D(const std::vector<int>& input_extents, int filter_count, int conv_filter_height, int conv_filter_width) :
		m_minibatch_size {input_extents.at(0)},
			m_image_height {input_extents.at(1)},
				m_image_width {input_extents.at(2)},
		m_output_activations {m_minibatch_size, filter_count, m_image_height, m_image_width},
			m_output_error {m_minibatch_size, filter_count, m_image_height, m_image_width},
				m_W(filter_count, conv_filter_height, conv_filter_width),
					m_temp_size_W(filter_count, conv_filter_height, conv_filter_width),
					m_sum_square_grad_W(filter_count, conv_filter_height, conv_filter_width),
					m_momentum_W(filter_count, conv_filter_height, conv_filter_width),
					m_W_grad(filter_count, conv_filter_height, conv_filter_width),
							m_bias(filter_count),
							m_temp_size_bias(filter_count),
					m_sum_square_grad_bias(filter_count),
					m_momentum_bias(filter_count),
							m_bias_grad(filter_count),
					m_temp_Z2 {m_image_height*m_image_width*m_minibatch_size, filter_count},
					m_temp_A1 {m_image_height*m_image_width*m_minibatch_size, conv_filter_height*conv_filter_width + 1},
						m_temp_W {conv_filter_height*conv_filter_width + 1, filter_count}
			 {

				 //check_dimensions(input_activations, input_error);

				 //check_dimensions(output_activations, output_error);
				 /*
			if (input_activations.extent(0) != output_activations.extent(0)) {
				std::cerr << "Error: input_activations and output_activations have a different mini-batch size!" << std::endl;
				exit(1);
			}
			if (input_activations.extent(1) != output_activations.extent(1)) {
				std::cerr << "Error: input_activations and output_activations have a different image height!" << std::endl;
				exit(1);
			}
			if (input_activations.extent(2) != output_activations.extent(2)) {
				std::cerr << "Error: input_activations and output_activations have a different image width!" << std::endl;
				exit(1);
			}
				 */
			// Use advice from:
			// http://arxiv.org/pdf/1206.5533.pdf
			//float fan_in = static_cast<float>(m_W.extent(1));
			//float r = 0.5*sqrt(1.0f /fan_in);
			//std::cout << "r = " << r << std::endl;
			//r = 0.1f; // 0.01
			//m_W.randomize(-r, r);

			//randomize_normal(m_W, 0.0f, r);

			//randomize_normal(m_W, 0.0f, 0.01f);
			//randomize_normal(m_bias, 0.0f, 0.01f);


			// Suggested in: http://arxiv.org/pdf/1502.01852.pdf (does not seem to work well)
			//const float std_dev_init = sqrt(2.0f/(image_height*image_width));
			//const float std_dev_init = sqrt(2.0f/(conv_filter_height*conv_filter_width));


				 const float std_dev_init = 0.5*sqrt(2.0f/(conv_filter_height*conv_filter_width));
			std::cout << "ConvLayer2D std_dev_init = " << std_dev_init << std::endl;
			randomize_normal(m_W, 0.0f, std_dev_init);

			std::cout << "m_temp_A1 rows = " << m_temp_A1.extent(0) << std::endl;
			std::cout << "m_temp_A1 cols = " << m_temp_A1.extent(1) << std::endl;
			std::cout << "m_temp_W rows = " << m_temp_W.extent(0) << std::endl;
			std::cout << "m_temp_W cols = " << m_temp_W.extent(1) << std::endl;

			//randomize_normal(m_W, 0.0f, 1.0f); // for gradient checking, use larger values for the weights.
			//randomize_normal(m_W_fixed_random, 0.0f, r);
		}

			 /*
			  * Call this method to initialize parameters before performing numerical gradient checking.
			  *
			  * This initializes the weight and bias parameters to large values which may be needed for
			  * gradient checking to work.
			  */
			 void set_gradient_check_mode();

		/*
		* Compute the output activations as a function of input activations.
		*/
		void forward_propagate(const Matrix& input_activations);

		/*
		 * Compute the gradients for W and bias.
		 *
		 * This updates m_W_grad and m_bias_grad.
		 */
		void back_propagate_weight_gradients(const Matrix& output_error, const Matrix& input_activations);

		/*
		* Update weights matrix and bias vector.
		*
		* This does not compute the gradients. This method should therefore be called after calling
		* compute_gradients().
		*/
		void update_weights();

		/*
		* Back-propagate errors to compute new values for input_error.
		*/
		void back_propagate_deltas(Matrix& input_error);

		/*
		* Perform a reverse ReLU operation. This function must be called before calling update_weights() or
		* back_propagate().
		*/
		//void reverse_activation();

		//void set_learning_rate(float rate);

		/*
		* Return a reference to the weights matrix.
		*/
		Matrix& get_weights();

		/*
		* Return a reference to the bias vector.
		*/
		Matrix& get_bias();

		/*
		* Return a reference to the weight gradient matrix.
		*/
		Matrix& get_weight_gradient();

		/*
		* Return a reference to the bias gradient vector.
		*/
		Matrix& get_bias_gradient();

				/*
		 * Return a reference to this layer's output activations.
		 */
		const Matrix& get_output() const {
			return m_output_activations;
		}

		/*
		 * Return a reference to this layer's output activations.
		 */
		Matrix& get_output() {
			return m_output_activations;
		}

		/*
		 * Return a reference to this layer's output deltas. These activations represent the
		 * gradient of the output activations (that is, errors for the output activations)
		 * that are computed during the back-propagation step.
		 */
		const Matrix& get_output_deltas() const {
			return m_output_error;
		}

		/*
		 * Return a reference to this layer's output deltas. These activations represent the
		 * gradient of the output activations (that is, errors for the output activations)
		 * that are computed during the back-propagation step.
		 */
		Matrix& get_output_deltas() {
			return m_output_error;
		}

				/*
		 * Return height of output activations which is also the
		 * height of input activations.
		 */
		int get_height() const {
			return m_output_activations.extent(2);
		}

		/*
		 * Return width of output activations which is also the
		 * width of input activations.
		 */
		int get_width() const {
			return m_output_activations.extent(3);
		}

		/*
		 * Return depth of output activations.
		 */
		int get_output_depth() const {
			return m_output_activations.extent(1);
		}


		/*
		 * Return the extents of the output activations.
		 * This information is typically passed to the constructor of the next layer in the network.
		 */
		std::vector<int> get_output_extents() const {
			return m_output_activations.get_extents();
		}

		/*
		 * Save learned parameters, stats etc. to a file withe the prefix given
		 * by the supplied name.
		 */
		void save_learning_info(std::string name) const;

		/*
		 * Compute the cost function J(W, bias, true_output_activations, current input activations)
		 *
		 * where J = 0.5*(||actual_output_activations - true_output_activations||^2)
		 *
		 * and where true_output_activations is the same size as output_activations.
		 *
		 * 
		 */ 
		float compute_cost_function(const Matrix& input_activations, const Matrix& true_output_activations);

		/*
		* Return a reference to the bias vector.
		*/
		//const concurrency::array<float, 1>& get_bias() const;

	// Make private again after debug.
	//private:
		std::string m_layer_name;
		const int m_minibatch_size;
		const int m_image_height;
		const int m_image_width;
		Matrix m_output_activations;
		Matrix m_output_error;
		Matrix m_W; // Parameter weights matrix.
		Matrix m_temp_size_W;
		Matrix m_sum_square_grad_W;
		Matrix m_momentum_W;
		//Matrix m_W_fixed_random; // experimental 
		Matrix m_W_grad; // Gradient of parameter weights matrix.
		//Matrix m_W_rms_mean_square; // Mean square average gradient for Rmsprop.
		//Matrix m_W_momentum; // For momentum.
		Matrix m_bias; // Bias parameters.
		Matrix m_temp_size_bias;
		Matrix m_sum_square_grad_bias;
		Matrix m_momentum_bias;
		Matrix m_bias_grad; // Gradient of bias parameters.

		// Temp matrices for optimized convolution functions.
		Matrix m_temp_Z2;
		Matrix m_temp_A1;
		Matrix m_temp_W;

		// Parameters

		float m_learning_rate = 0.3f; // 0.3f
		float m_bias_learning_rate = 0.3f; //m_learning_rate/5.0f;
		float m_lambda = 0.0f; //0.00001f;
		float m_sparsity_param = 0.0f;//1.0e-6f; //1.0e-6f // scale by 1/(number of elements in X) first.
		float m_force_nonnegative = false;
		float m_rmsprop_learning_rate = 1e-4f; // 2e-4f

		// Log the learning rate update magnitude for the weights vs mini-batch index. 
		// This will be constant unless adaptive learning rates are enabled.
		std::vector<float> m_log_weight_updates;

		// Log the max value of the weights matrix vs mini-batch index.
		std::vector<float> m_log_weight_max_val;

		// Log the min value of the weights matrix vs mini-batch index.
		std::vector<float> m_log_weight_min_val;

		// Log the learning rate update magnitude for the bias. This will be constant unless adaptive learning rates are enabled.
		std::vector<float> m_log_bias_updates;

		// Log the max value of the bias matrix vs mini-batch index.
		std::vector<float> m_log_bias_max_val;

		// Log the min value of the bias matrix vs mini-batch index.
		std::vector<float> m_log_bias_min_val;
		
		private:

	};

}

#endif /* _CONVLAYER2D_H */
