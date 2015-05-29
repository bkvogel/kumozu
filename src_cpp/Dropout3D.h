#ifndef _DROPOUT3D_H
#define _DROPOUT3D_H
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
	 * An instance of this class represents an dropout function that operates on a mini-batch
	 * of input 3D (i.e., box-shaped) matrices and outputs a corresponding mini-batch of 3D
	 * output matrices. Typically, the input corresponds to a mini-batch of activations from the
	 * output of a linear convolutional layer.
	 *
	 * The dropout function is applied independently to each 3D matrix in the mini-batch. Specifically,
	 * the inputs to the function are supplied as a Matrix of size (minibatch_size, depth, height, width).
	 * The output of the dropout function is a matrix of size as the input matrix. 
	 *
	 * When using the class, a reference to the input matrix must be supplied. However, this class
	 * creates its own output matrix and error (i.e., deltas) matrix when instantiated and methods are provided to obtain a reference
	 * to the output matrices.
	 *
	 * It should be noted that this class corresponds to two output matrices of the same size: "output" and
	 * "output_deltas." These two matrices are member variables of the class instance that are allocated
	 * by the constructor. A forward dropout call will compute "output" as a function of the supplied
	 * "input" matrix. A reverse dropout call will compute "input" as a function of the member "output_deltas"
	 * matrix, which is intended to be called as part of the back-propagation procedure.
	 */
	class Dropout3D {

	public:




		/*
		 * Create an instance of a dropout function that operates on a mini-batch of
		 * data at a time.
		 *
		 * activation_extents: Dimensions of the input and output activations which are 
		 *                (minibatch_size x depth x height x width). 
		 */
	Dropout3D(const std::vector<int>& activation_extents,
							 float prob_keep) : 
		m_minibatch_size {activation_extents.at(0)},
			m_depth {activation_extents.at(1)},
				m_height {activation_extents.at(2)},
					m_width {activation_extents.at(3)},
		m_output { m_minibatch_size, m_depth, m_height, m_width },
			m_output_deltas { m_minibatch_size, m_depth, m_height, m_width},
			m_prob_keep {prob_keep},
					m_prob_keep_current {prob_keep},
						m_dropout_mask { m_depth, m_height, m_width},
						m_temp_rand { m_depth, m_height, m_width}
				
		{
			std::cout << "Dropout3D:" << std::endl;
			m_mersenne_twister_engine.seed(static_cast<unsigned long>(time(NULL)));

		}


		/*
		 * Compute the dropout function in the forward direction.
		 *
		 * The dropout function of the supplied input activations is computed.
		 * The results are stored in the output activations member variable, which
		 * can be obtained by calling get_output().
		 *
		 */
		void forward_dropout(const Matrix& input);

		/*
		 * Compute the dropout function in the reverse direction.
		 *
		 * The reverse dropout function of the output deltas activations member variable
		 * is computed and the result is stored in the supplied input activations variable.
		 * This method is typically used during the back-propigation step to back-propagate
		 * deltas (errors) through the reverse dropout function.
		 */
		void reverse_dropout(Matrix& input);

		
		/*
		 * Enable dropout.
		 */
		void enable_dropout() {
			m_prob_keep_current = m_prob_keep;
		}

		/*
		 * Disable droput. 
		 *
		 * To enable again, call enable_dropout().
		 */
		void disable_dropout() {
			m_prob_keep_current = 1.0f;
		}

		/*
		 * Set the probability of keeping any given element.
		 */
		void set_prob_keep(float prob_keep) {
			m_prob_keep = prob_keep;
			m_prob_keep_current = prob_keep;
		}

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

		int m_minibatch_size;
		int	m_depth;
		int	m_height;
		int	m_width;
		Matrix m_output;
		Matrix m_output_deltas;
		float m_prob_keep;
		float m_prob_keep_current;
		MatrixT<int> m_dropout_mask;
		Matrix m_temp_rand;
		std::mt19937 m_mersenne_twister_engine;

	};

}

#endif /* _DROPOUT3D_H */
