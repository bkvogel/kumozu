#ifndef _DROPOUT1D_H
#define _DROPOUT1D_H
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
#include <random>

namespace kumozu {

	

	/*
	 * An instance of this class represents a dropout function that operates on a mini-batch
	 * of input column vectors and outputs a corresponding mini-batch of column vectors.
	 *
	 * A single dropout mask is generated and then applied independently to each column in the mini-batch. Specifically,
	 * the inputs to the dropout function are supplied as a Matrix of size (dim_input x minibatch_size).
	 * The output of the dropout function is a Matrix of size (dim_output x minibatch_size).
	 *
	 * When using the class, a reference to the input matrix must be supplied. However, this class
	 * creates its output matrix and error (i.e., deltas) matrix when it is instantiated and methods are provided to get a reference
	 * to the output matrices.
	 *
	 * It should be noted that this class corresponds to two output matrices of the same size, "output" and
	 * "output_deltas." These two matrices are member variables of the class instance that are allocated
	 * by the constructor. A forward dropout call will compute "output" as a function of the supplied
	 * "input" matrix. A reverse dropout call will compute "input" as a function of the member "output_deltas"
	 * matrix, which is intended to be called as part of the back-propagation procedure.
	 *
	 * Note that the dropout function can be thought of as a type of activation function since it computes an output
	 * as a non-linear function of the input.
	 *
	 * This class implements "inverted dropout" as described in:
	 * http://cs231n.github.io/neural-networks-2/
	 *
	 * Thus, we perform dropout and scale the activations in the forward training pass so that scaling is not performed
	 * at test time.
	 */
	class Dropout1D {

	public:


		/*
		 * Create an instance of a dropout function that operates on a mini-batch of
		 * column activation vectors.
		 *
		 * Parameters
		 *
		 * prob_keep: Probability of keeping a unit active. Must be in the range (0, 1].
		 */
	Dropout1D(const std::vector<int>& input_extents,
							 float prob_keep) : 
		// Note: input_extents.at(1) is mini-batch size.
		// input_extents.at(0) is dim_input.
		m_output {input_extents.at(0), input_extents.at(1)},
			m_output_deltas {input_extents.at(0), input_extents.at(1)},
				m_prob_keep {prob_keep},
					m_prob_keep_current {prob_keep},
						m_dropout_mask(input_extents.at(0)),
						m_temp_rand(input_extents.at(0))
		{
			std::cout << "Dropout1D:" << std::endl;
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

		//private: 

		Matrix m_output;
		Matrix m_output_deltas;

		float m_prob_keep;
		float m_prob_keep_current;
		std::vector<int> m_dropout_mask;
		std::vector<float> m_temp_rand;
		std::mt19937 m_mersenne_twister_engine;
	};

}

#endif /* _DROPOUT1D_H */
