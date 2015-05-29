#ifndef _LINEARLAYER_H
#define _LINEARLAYER_H
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
	* An instance of this class represents 1 fully-connected linear layer in a neural network. A "layer" represents
	* the weight and bias parameters and activation function that computes the output activations as a function of the input
	* activations, as well as the methods for performing parameter learning, including the
	* parameters themselves.
	*
	* The caller is also responsible for creating all data matrices that are shared between network
	* layers. This shared data typically consists of the network activations themselves. Therefore, an
	* instance of this class will need to be provided a reference to both the input and output
	* activations in the constructor.
	*
	* An instance of this class will create the corresponding parameter matrices (weights) and bias vector.
	*
	* This class is intended to be used in a network that operates on "mini-batches" of data.
	*
	* Usage:
	*
	* Caller decides on mini-batch size N, and input_activatins dimension dim_input, and an output
	* activations dimension dim_output, and creates several matrices:
	*
	* input_activations: dim_input x N matrix.
	*
	* output_activations: dim_output x N matrix.
	*
	* input_error: dim_input x N matrix.
	*
	* output_error: dim_output x N matrix.
	*
	* The caller also decides on the maxout factor "maxout_factor" which specifies the number of piecewise
	*	linearities in each activation.
	*
	* This constructor will then create the internal parameters for this layer:
	*
	* W: The weights matrix, which is of dimension (dim_output*maxout_factor) x dim_input.
	*
	* bias: The bias vector, which is of size dim_output*maxout_factor) x 1.
	*
	* A reference to these parameters can be obtained by calling:
	*
	* get_weights() to return a reference to "W" or
	*
	* get_bias() to return a reference to "bias."
	*
	* To compute the output activations as a function of input activations, call forward_propagate().
	*
	* To compute the input activations error as a function of output activations error and parameters, 
	* call back_propagate().
	*
	* To update the parameter values based on activations error, call update_weights().
	*
	*/
	class LinearLayer {

	public:

		/*
		* 
		* 
		*
		* Parameters:
		*
		* input_extents: Dimensions of the input activations, which are 
		*                (dim_input, minibatch_size)
		* 
		*/
	LinearLayer(const std::vector<int>& input_extents, int dim_output, std::string name) :
		m_layer_name {name},
			 m_input_layer_units{ input_extents.at(0) },
			m_output_layer_units {dim_output},
				m_minibatch_size {input_extents.at(1)},
		m_output_activations { dim_output, m_minibatch_size },
			m_output_error { dim_output, m_minibatch_size },
			m_W(dim_output, m_input_layer_units),
			m_temp_size_W(dim_output, m_input_layer_units),
				m_sum_square_grad_W(dim_output, m_input_layer_units),
				m_momentum_W(dim_output, m_input_layer_units),
				//m_W_fixed_random(dim_output, dim_input),
			m_W_grad(dim_output, m_input_layer_units),
				//m_W_rms_mean_square(dim_output, dim_input),
				//m_W_momentum(dim_output, dim_input),
			m_bias(dim_output),
			m_temp_size_bias(dim_output),
				m_sum_square_grad_bias(dim_output),
				m_momentum_bias(dim_output),
			m_bias_grad(dim_output),
				//m_bias_rms_mean_square(dim_output),
			m_output_activations_indices { dim_output, m_minibatch_size }
			 {
				 //check_dimensions(input_activations, input_error);
				 //check_dimensions(output_activations, output_error);
				 /*
			if (input_activations.extent(1) != output_activations.extent(1)) {
				std::cerr << "Error: input_activations and output_activations have a different number of columns!" << std::endl;
				exit(1);
			}
				 */
			// Use advice from:
			// http://arxiv.org/pdf/1206.5533.pdf
			//float fan_in = static_cast<float>(m_W.extent(1));
			//float r = 0.5*sqrt(1.0f /fan_in);

			//std::cout << "LinearLayer r = " << r << std::endl;
			//float r = 0.05f; // 0.01
			//m_W.randomize(-r, r);
			
			//randomize_normal(m_W, 0.0f, r);

			//randomize_normal(m_W, 0.0f, 0.01f);
			//randomize_normal(m_bias, 0.0f, 0.01f);

			// Suggested in: http://arxiv.org/pdf/1502.01852.pdf
				 // Must also scale input to have 0 mean and unit std dev.
			//const float std_dev_init = sqrt(2.0f/m_W.extent(1));
				 //const float std_dev_init = sqrt(2.0f/m_W.extent(1));

				 // Old method.
				 //const float std_dev_init = 0.5*sqrt(1.0f/m_W.extent(1));
				 // Set to std deviation of the input layer data.

				 const float data_std_dev = 0.5; // 0.5, 0.35 // todo: move this to somewhere else?
				 const float std_dev_init = data_std_dev*sqrt(2.0f/m_W.extent(1));

				 //const float std_dev_init = 1.0*sqrt(1.0f/m_W.extent(1));
				 //const float std_dev_init = 0.01f;
				 std::cout << "LinearLayer std_dev_init = " << std_dev_init << std::endl;
			//randomize_normal(m_W, 0.0f, std_dev_init);
			randomize_normal(m_W, 0.0f, 1.0f);
			scale(m_W, m_W, std_dev_init);

			//randomize_normal(m_bias, 0.0f, 0.01f);


			//m_W.copy_cpu_to_gpu();
			
			//m_W_fixed_random.randomize(-r, r);
			//randomize_normal(m_W_fixed_random, 0.0f, r);
			//m_W_fixed_random.copy_cpu_to_gpu();
			//;
			//std::cout << "LinearLayer: debug: end of constructor." << std::endl;
		}




	   /*
		* 
		* 
		*
		* Parameters:
		*
		* input_extents: Dimensions of the input activations, which are 
		*                (dim_input, minibatch_size)
		* 
		*/
		// deprecated: add string layer name to parameters.
	LinearLayer(const std::vector<int>& input_extents, int dim_output) :
		m_layer_name {""},
			 m_input_layer_units{ input_extents.at(0) },
			m_output_layer_units {dim_output},
				m_minibatch_size {input_extents.at(1)},
		m_output_activations { dim_output, m_minibatch_size },
			m_output_error { dim_output, m_minibatch_size },
			m_W(dim_output, m_input_layer_units),
			m_temp_size_W(dim_output, m_input_layer_units),
				m_sum_square_grad_W(dim_output, m_input_layer_units),
				m_momentum_W(dim_output, m_input_layer_units),
				//m_W_fixed_random(dim_output, dim_input),
			m_W_grad(dim_output, m_input_layer_units),
				//m_W_rms_mean_square(dim_output, dim_input),
				//m_W_momentum(dim_output, dim_input),
			m_bias(dim_output),
			m_temp_size_bias(dim_output),
				m_sum_square_grad_bias(dim_output),
				m_momentum_bias(dim_output),
			m_bias_grad(dim_output),
				//m_bias_rms_mean_square(dim_output),
			m_output_activations_indices { dim_output, m_minibatch_size }
			 {
				 //check_dimensions(input_activations, input_error);
				 //check_dimensions(output_activations, output_error);
				 /*
			if (input_activations.extent(1) != output_activations.extent(1)) {
				std::cerr << "Error: input_activations and output_activations have a different number of columns!" << std::endl;
				exit(1);
			}
				 */
			// Use advice from:
			// http://arxiv.org/pdf/1206.5533.pdf
			//float fan_in = static_cast<float>(m_W.extent(1));
			//float r = 0.5*sqrt(1.0f /fan_in);

			//std::cout << "LinearLayer r = " << r << std::endl;
			//float r = 0.05f; // 0.01
			//m_W.randomize(-r, r);
			
			//randomize_normal(m_W, 0.0f, r);

			//randomize_normal(m_W, 0.0f, 0.01f);
			//randomize_normal(m_bias, 0.0f, 0.01f);

			// Suggested in: http://arxiv.org/pdf/1502.01852.pdf
				 // Must also scale input to have 0 mean and unit std dev.
			//const float std_dev_init = sqrt(2.0f/m_W.extent(1));
				 //const float std_dev_init = sqrt(2.0f/m_W.extent(1));

				 // Old method.
				 //const float std_dev_init = 0.5*sqrt(1.0f/m_W.extent(1));
				 // Set to std deviation of the input layer data.

				 const float data_std_dev = 0.5; // 0.5, 0.35 // todo: move this to somewhere else?
				 const float std_dev_init = data_std_dev*sqrt(2.0f/m_W.extent(1));

				 //const float std_dev_init = 1.0*sqrt(1.0f/m_W.extent(1));
				 //const float std_dev_init = 0.01f;
				 std::cout << "LinearLayer std_dev_init = " << std_dev_init << std::endl;
			//randomize_normal(m_W, 0.0f, std_dev_init);
			randomize_normal(m_W, 0.0f, 1.0f);
			scale(m_W, m_W, std_dev_init);

			//randomize_normal(m_bias, 0.0f, 0.01f);


			//m_W.copy_cpu_to_gpu();
			
			//m_W_fixed_random.randomize(-r, r);
			//randomize_normal(m_W_fixed_random, 0.0f, r);
			//m_W_fixed_random.copy_cpu_to_gpu();
			//;
			//std::cout << "LinearLayer: debug: end of constructor." << std::endl;
		}


			 /*
			  * Call this method to initialize parameters before performing numerical gradient checking.
			  *
			  * This initializes the weight and bias parameters to large values which may be needed for
			  * gradient checking to work.
			  */
			 // deprecated
			 void set_gradient_check_mode();

	

		/*
		* Compute the output activations as a function of input activations.
		*
		* The output activations can then be obtained by calling get_output().
		*/
		void forward_propagate(const Matrix& input_activations);

		/*
		 * Compute the gradients for W and bias.
		 *
		 * This updates m_W_grad and m_bias_grad.
		 */
		// deprecated
		void back_propagate_weight_gradients(const Matrix& output_error, const Matrix& input_activations);

		/*
		* Update weights matrix and bias vector using the weight and bias gradients that were computed
		* by back_propagate_weight_gradients().
		*
		* Be sure to call back_propagate_weight_gradients() before calling this method.
		*/
		void update_weights();

		/*
		* Back-propagate errors to compute new values for input_error.
		*/
		// deprecated.
		//void back_propagate_deltas(Matrix& input_error, const Matrix& output_error);

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
		 * Return the input dimension.
		 */
		int get_dim_input() const {
			return m_W.extent(1);
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
		const int m_input_layer_units;
		const int m_output_layer_units;
		const int m_minibatch_size;
		Matrix m_output_activations;
		Matrix m_output_error; 
		Matrix m_W; // Parameter weights matrix.
		Matrix m_temp_size_W;
		Matrix m_sum_square_grad_W;
		Matrix m_momentum_W;
		//Matrix m_W_fixed_random; // experimental 
		Matrix m_W_grad; // Gradient of parameter weights matrix.
		Matrix m_bias; // Bias parameters.
		Matrix m_temp_size_bias;
		Matrix m_sum_square_grad_bias;
		Matrix m_momentum_bias;
		Matrix m_bias_grad; // Gradient of bias parameters.
		MatrixT<int> m_output_activations_indices;

		// Parameters
		float m_rms_grad_keep_weight = 0.9f; // 0.5 - 0.9;
		float m_rmsprop_learning_rate = 1.0e-4f; // 2.0e-4;
		//float m_learning_rate = 0.3f; // 0.5f
		float m_learning_rate = 0.3f; // 0.3f
		float m_bias_learning_rate = 0.3f; //m_learning_rate/5.0f;
		bool m_enable_updates = false;
		float m_lambda = 0.00000f; //0.00001f;
		float m_sparsity_param = 0.0f;//1.0e-6f; //1.0e-6f // scale by 1/(number of elements in X) first.
		float m_force_nonnegative = false;
		float m_rms_counter = 1.0f;

		// Monitoring the learning process
		//
		// We will monitor several quantities during the learning process. These are implemented as lists (i.e., vectors)
		// that will grow by 1 on each mini-batch of data.

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

#endif /* _LINEARLAYER_H */
