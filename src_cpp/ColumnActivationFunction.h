#ifndef _COLUMNACTIVATIONFUNCTION_H
#define _COLUMNACTIVATIONFUNCTION_H
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
	 * An instance of this class represents an activation function that operates on a mini-batch
	 * of input column vectors and outputs a corresponding mini-batch of column vectors.
	 *
	 * The activation function is applied independently to each column. Specifically,
	 * the inputs to the activation are supplied as a Matrix of size (dim_input x minibatch_size).
	 * The output of the activation function is a Matrix of size (dim_output x minibatch_size).
	 *
	 * When using the class, a reference to the input matrix must be supplied. However, this class
	 * creates its output matrix and error (i.e., deltas) matrix when it is instantiated and methods are provided to get a reference
	 * to the output matrices.
	 *
	 * It should be noted that this class corresponds to two output matrices of the same size, "output" and
	 * "output_deltas." These two matrices are member variables of the class instance that are allocated
	 * by the constructor. A forward activation call will compute "output" as a function of the supplied
	 * "input" matrix. A reverse activation call will compute "input" as a function of the member "output_deltas"
	 * matrix, which is intended to be called as part of the back-propagation procedure.
	 */
	class ColumnActivationFunction {

	public:

		enum class ACTIVATION_TYPE { ReLU, leakyReLU, linear, maxout, kmax };

		/*
		 * Create an instance of an activation function that operates on a mini-batch of
		 * column activation vectors.
		 */
		// deprecated
	ColumnActivationFunction(int minibatch_size, int dim_input, int dim_output,
							 ACTIVATION_TYPE activation_type) : 
		m_output {dim_output, minibatch_size},
			m_output_deltas {dim_output, minibatch_size},
			m_state {dim_output, minibatch_size},
				m_activation_type {activation_type}
		{
			std::cout << "ColumnActivationFunction:" << std::endl;
			std::cout << "dim_input = " << dim_input << std::endl;
			std::cout << "dim_output = " << dim_output << std::endl;
			// fixme: add error checks.
			if (m_activation_type == ACTIVATION_TYPE::ReLU) {
				std::cout << "Using ReLU activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
				std::cout << "Using leakyReLU activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::linear) {
				std::cout << "Using linear activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::maxout) {
				std::cout << "Using maxout activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::kmax) {
				std::cout << "Using kmax activation:" << std::endl;
			}

			// Default values for kmax:
			m_partition_count = 1;
			m_k = 1;
		}

		/*
		 * Create an instance of an activation function that operates on a mini-batch of
		 * column activation vectors.
		 */
	ColumnActivationFunction(const std::vector<int>& input_extents, int dim_output,
							 ACTIVATION_TYPE activation_type) : 
		// Note: input_extents.at(1) is mini-batch size.
		// input_extents.at(1) is dim_input.
		m_output {dim_output, input_extents.at(1)},
			m_output_deltas {dim_output, input_extents.at(1)},
			m_state {dim_output, input_extents.at(1)},
				m_activation_type {activation_type}
		{
			std::cout << "ColumnActivationFunction:" << std::endl;
			std::cout << "dim_input = " << input_extents.at(1) << std::endl;
			std::cout << "dim_output = " << dim_output << std::endl;
			// fixme: add error checks.
			if (m_activation_type == ACTIVATION_TYPE::ReLU) {
				std::cout << "Using ReLU activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
				std::cout << "Using leakyReLU activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::linear) {
				std::cout << "Using linear activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::maxout) {
				std::cout << "Using maxout activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::kmax) {
				std::cout << "Using kmax activation:" << std::endl;
			}

			// Default values for kmax:
			m_partition_count = 1;
			m_k = 1;
		}


		/*
		 * Compute the activation function in the forward direction.
		 *
		 * The activation function of the supplied input activations is computed.
		 * The results are stored in the output activations member variable, which
		 * can be obtained by calling get_output().
		 *
		 * Depending on the activation type, calling this method may modify internal
		 * state which will be needed by reverse_activation().
		 */
		void forward_activation(const Matrix& input);

		/*
		 * Compute the activation function in the reverse direction.
		 *
		 * The reverse activation function of the output deltas activations member variable
		 * is computed and the result is stored in the supplied input activations variable.
		 * This method is typically used during the back-propigation step to back-propagate
		 * deltas (errors) through the activation function.
		 */
		void reverse_activation(Matrix& input);

		/*
		 * Return a reference to the output activations.
		 */
		const Matrix& get_output() const {
			return m_output;
		}

		/*
		 * Return a reference to the output activations.
		 */
		Matrix& get_output() {
			return m_output;
		}

		/*
		 * Return a reference to the output deltas (back-prop errors) activations.
		 */
		const Matrix& get_output_deltas() const {
			return m_output_deltas;
		}

		/*
		 * Return a reference to the output deltas (back-prop errors) activations.
		 */
		Matrix& get_output_deltas() {
			return m_output_deltas;
		}

		
		/*
		 * Return the extents of the output activations.
		 * This information is typically passed to the constructor of the next layer in the network.
		 */
		std::vector<int> get_output_extents() const {
			return m_output.get_extents();
		}

		//private: // fixme

		Matrix m_output;
		Matrix m_output_deltas;
		MatrixT<int> m_state;
		ACTIVATION_TYPE m_activation_type;

		int m_partition_count;
		int m_k;

	};

}

#endif /* _COLUMNACTIVATIONFUNCTION_H */
