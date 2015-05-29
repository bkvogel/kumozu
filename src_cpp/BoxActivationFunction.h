#ifndef _BOXACTIVATIONFUNCTION_H
#define _BOXACTIVATIONFUNCTION_H
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
	 * of input 3D (i.e., box-shaped) matrices and outputs a corresponding mini-batch of 3D
	 * output matrices. Typically, the input corresponds to a mini-batch of activations from the
	 * output of a linear convolutional layer.
	 *
	 * The activation function is applied independently to each 3D matrix in the mini-batch. Specifically,
	 * the inputs to the activation are supplied as a Matrix of size (minibatch_size, depth, height, width).
	 * The output of the activation function is a matrix of size as the input matrix. Therefore, this class
	 * does not include pooling (a sperate class is available for that).
	 *
	 * When using the class, a reference to the input matrix must be supplied. However, this class
	 * creates its own output matrix and error (i.e., deltas) matrix when instantiated and methods are provided to obtain a reference
	 * to the output matrices.
	 *
	 * It should be noted that this class corresponds to two output matrices of the same size: "output" and
	 * "output_deltas." These two matrices are member variables of the class instance that are allocated
	 * by the constructor. A forward activation call will compute "output" as a function of the supplied
	 * "input" matrix. A reverse activation call will compute "input" as a function of the member "output_deltas"
	 * matrix, which is intended to be called as part of the back-propagation procedure.
	 */
	class BoxActivationFunction {

	public:

		enum class ACTIVATION_TYPE { ReLU, leakyReLU, linear, kmax };


		/*
		 * Create an instance of an activation function that operates on a mini-batch of
		 * data at a time.
		 *
		 * activation_extents: Dimensions of the input and output activations which are 
		 *                (minibatch_size x depth x height x width). 
		 */
	BoxActivationFunction(const std::vector<int>& activation_extents,
							 ACTIVATION_TYPE activation_type) : 
		m_minibatch_size {activation_extents.at(0)},
			m_depth {activation_extents.at(1)},
				m_height {activation_extents.at(2)},
					m_width {activation_extents.at(3)},
		m_output { m_minibatch_size, m_depth, m_height, m_width },
			m_output_deltas { m_minibatch_size, m_depth, m_height, m_width},
			m_state { m_minibatch_size, m_depth, m_height, m_width},
				m_activation_type {activation_type},
				// Default values for kmax:
					m_box_depth {1},
						m_box_height {1},
							m_box_width {1},
								m_k {1}
		{
			std::cout << "BoxActivationFunction:" << std::endl;
			// fixme: add error checks.
			if (m_activation_type == ACTIVATION_TYPE::ReLU) {
				std::cout << "Using ReLU activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
				std::cout << "Using leakyReLU activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::linear) {
				std::cout << "Using linear activation:" << std::endl;
			} else if (m_activation_type == ACTIVATION_TYPE::kmax) {
				std::cout << "Using kmax activation:" << std::endl;
			}

		}

		/*
		 * Set parameters for kmax activations function.
		 *
		 * If this function is not called. The default value of 1 is used for all parameters.
		 */
		void set_kmax_parameters(int box_depth, int box_height, int box_width, int k) {
			m_box_depth = box_depth;
			m_box_height = box_height;
			m_box_width = box_width;
			m_k = k;
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
		 * The reverse activation function of the output_deltas activations member variable
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


	private:

		int m_minibatch_size;
		int	m_depth;
		int	m_height;
		int	m_width;
		Matrix m_output;
		Matrix m_output_deltas;
		MatrixT<int> m_state;
		ACTIVATION_TYPE m_activation_type;

		int m_box_depth;
		int m_box_height;
		int m_box_width;
		int m_k; // When using kmax, number of activations to keep in each box partition.

	};

}

#endif /* _BOXACTIVATIONFUNCTION_H */
