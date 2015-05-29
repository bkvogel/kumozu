#ifndef _NETWORK_H
#define _NETWORK_H
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

namespace kumozu {

	/*
	 * This is a base class for a multi-layer feed-forward network.
	 *
	 */
	class Network {

	public:

	Network(const std::vector<int>& input_extents) : m_input_extents {input_extents}, 
			m_epsilon {1e-3f}, m_pass_relative_error {5e-2f} 
		{
			/*
			auto c = m_input_extents;
			std::cout << "Network:" << std::endl;
			std::cout << "c.size() = " << c.size() << std::endl;
			std::cout << "c.at(0) = " << c.at(0) << std::endl;
			std::cout << "c.at(1) = " << c.at(1) << std::endl;
			std::cout << "c.at(2) = " << c.at(2) << std::endl;
			//
			auto d = get_input_extents();
			std::cout << "d.size() = " << d.size() << std::endl;
			std::cout << "d.at(0) = " << d.at(0) << std::endl;
			std::cout << "d.at(1) = " << d.at(1) << std::endl;
			std::cout << "d.at(2) = " << d.at(2) << std::endl;
			*/
		}

		virtual ~Network(){}

		/*
		 * Compute the output activations as a function of input activations.
		 */
		virtual void forward_propagate(const Matrix& input_activations) = 0;
		
		/*
		 * Perform back-propagation to compute all gradients: weights, biases, and activation deltas.
		 *
		 * Call forward_propagate() before this method to compute output activations.
		 *
		 * This updates m_W_grad and m_bias_grad.
		 */
		virtual void back_propagate_gradients(const Matrix& input_activations, 
													 const Matrix& true_output_activations) = 0;

		/*
		 * Compute the cost function J(W, bias, true_output_activations, current input activations)
		 *
		 * where J = 0.5*(||actual_output_activations - true_output_activations||^2)
		 *
		 * true_output_activations: The desired output activations, such as the training activations values.
		 * 
		 */ 
		virtual float compute_cost_function(const Matrix& input_activations, const Matrix& true_output_activations) = 0;

		/*
		 * Update weights and bias parameters.
		 *
		 * This does not compute the gradients. This method should therefore be called after calling
		 * compute_gradients().
		 */
		virtual void update_weights() = 0;

		/*
		 * Perform numerical gradient checks using finite differences method and compare to
		 * gradients computed by back-propagation. Return true if the checks pass. Otherwise,
		 * return false.
		 */
		virtual bool check_gradients(const Matrix& input_activations, const Matrix& true_output_activations) = 0;

		/*
		 * Return the input extents for this network.
		 */
		std::vector<int> get_input_extents() const {
			/*
			std::cout << "get_input_extents():" << std::endl;
			std::cout << "size() = " << m_input_extents.size() << std::endl;
			std::cout << "at(0) = " << m_input_extents.at(0) << std::endl;
			std::cout << "at(1) = " << m_input_extents.at(1) << std::endl;
			std::cout << "at(2) = " << m_input_extents.at(2) << std::endl;
			*/
			return m_input_extents;
		}

		/*
		 * Return a reference to this layer's output activations.
		 */
		virtual const Matrix& get_output() const = 0;

		/*
		 * Return a reference to this layer's output activations.
		 */
		virtual Matrix& get_output() = 0;

		/*
		 * Enable dropout.
		 *
		 * Dropout should only be enabled in "train mode."
		 * Dropout should be disabled before running in "test mode."
		 *
		 * This default implementation does nothing so that derived classes do not need to
		 * override.
		 */
		virtual void enable_dropout() {}

		/*
		 * Disable dropout. 
		 *
		 * Dropout should only be enabled in "train mode."
		 * Dropout should be disabled before running in "test mode."
		 *
		 * To enable again, call enable_dropout().
		 *
		 * This default implementation does nothing so that derived classes do not need to
		 * override.
		 */
		virtual void disable_dropout() {}

		/*
		 * Save parameters and stats to a file withe the prefix given
		 * by the supplied name.
		 */
		virtual void save_learning_info(std::string name) const = 0;

		/*
		 * Load learned parameters from a file. The string name should be
		 * the same that was used to save the parameters when
		 * save_learning_info() was called.
		 */
		virtual void load_parameters(std::string name) const = 0;

		/*
		 * Compute gradients numerically using finite differnces method and compare to supplied "parameters" matrix. If the numerically-
		 * computed values are close enough, return true. Otherwise, return false.
		 *
		 * This function iterates through the supplied "parameters" matrix, making small changes in the values and calling
		 * compute_cost_function() after each change in order to compute the numerical gradients. After doing this, if
		 * the numerical gradients matrix is close enough to the supplied "gradients_back_prop" matrix, return. Otherwise,
		 * exit with an error message.
		 *
		 * layer: an object that has a member function of type:
		 *           float compute_cost_function(input_activations, true_output_activations)
		 *
		 * input_activations: Input to compute_cost_function().
		 *
		 * true_output_activations: Input to compute_cost_function().
		 *
		 * gradients_back_prop: The gradients that were already computed by back-propagation.
		 *
		 * parameters: Either the weights matrix or the bias matrix.
		 *
		 */
		void numerical_gradients(const Matrix& input_activations, 
								 const Matrix& true_output_activations, const Matrix& gradients_back_prop, 
								 Matrix& parameters);

	private:

		const std::vector<int>& m_input_extents;
		const float m_epsilon; // For finite difference approximation to gradient.
		const float m_pass_relative_error; // // Relative error must be below this to pass.

	};

}


#endif /* _NETWORK_H */
