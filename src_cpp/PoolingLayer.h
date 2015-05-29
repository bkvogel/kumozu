#ifndef _POOLINGLAYER_H
#define _POOLINGLAYER_H
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
	 * An instance of this class represents a pooling layer in a network.
	 *
	 * The input to the pooling layer is a mini-batch of 3D matrices of size (depth_in x height_in x width_in).
	 * Adding the mini-batch index as a dimension results in a 4D matrix.
	 *
	 * Input: 4D matrix of size (minibatch_size x depth_in x height_in x width_in).
	 * Note that fixing the first dimension to a particular mini-batch results
	 * in a 3D matrix.
	 *
	 * The output of the pooling layer is a mini-batch of 3d matrices:
	 * Output: 4D matrix of size (minibatch_size x depth_out x height_out x width_out).
	 *
	 * This class currently supports only max-pooling.
	 *
	 * For the i'th mini-batch, pooling is performed by partitioning the 3D (depth_in x weight_in x width_in).
	 * input matrix into boxes of size (depth_in/depth_out x height_in/height_out x width_in/width_out). There
	 * must be no remainder when performing these divisions. Otherwise, the program will exit with an error.
	 * Note that the stride is the same as the box size so that the boxes do not overlap.
	 * Note also that (depth_in/depth_out) > 1 corresponds to a maxout activation function when max-pooling is used.
	 * 
	 * When using the class, a reference to the input matrix must be supplied. However, this class
	 * creates its own output matrix and error (i.e., deltas) matrix when instantiated and methods are provided to obtain a reference
	 * to the output matrices.
	 *
	 * It should be noted that this class corresponds to two output matrices of the same size: "output" and
	 * "output_deltas." These two matrices are member variables of the class instance that are allocated
	 * by the constructor. A forward pooling call will compute "output" as a function of the supplied
	 * "input" matrix:
	 *
	 * Forward pool:
	 *
	 * input parameter -> output activations (member variable, call get_output() to get a reference)
	 *
	 * Reverse pool:
	 *
	 * output_deltas (member variable, call get_output_deltas() to get a reference) -> input parameter
	 *
	 * A reverse activation call will compute "input" as a function of the member "output_deltas"
	 * matrix, which is intended to be called as part of the back-propagation procedure.
	 */
	class PoolingLayer {

	public:

		//enum class ACTIVATION_TYPE { ReLU, leakyReLU, linear, kmax };

		

		/*
		 * Create an instance of an pooling function that operates on a mini-batch of
		 * data at a time.
		 *
		 * Use this consstructor to create non-overlapping polling regions.
		 *
		 * The output activations will have size 
		 * (minibatch_size, depth_in/depth_factor, height_in/height_factor, width_in/width_factor).
		 *
		 * Parameters:
		 *
		 * input_extents: Extents for input activations of size (minibatch_size, depth_in, height_in, width_in).
		 *
		 * height_factor: Reduction factor along height dimension.
		 *
		 * width_factor: Reduction factor along width dimension.
		 * 
		 * depth_factor: Reduction factor along depth dimension.
		 */
		// deprecated: this constructor does not allow overlapping pooling regions.
		/*
	PoolingLayer(const std::vector<int>& input_extents,
				 int height_factor, int width_factor, int depth_factor) : 
		m_output { input_extents.at(0), input_extents.at(1)/depth_factor, input_extents.at(2)/height_factor, input_extents.at(3)/width_factor },
			m_output_deltas { input_extents.at(0), input_extents.at(1)/depth_factor, input_extents.at(2)/height_factor, input_extents.at(3)/width_factor},
				m_state { input_extents.at(0), input_extents.at(1)/depth_factor, input_extents.at(2)/height_factor, input_extents.at(3)/width_factor, 3}
		{
			std::cout << "PoolingLayer:" << std::endl;
			if (input_extents.size() != 4) {
				std::cerr << "Input extents have wrong order. Exiting." << std::endl;
				exit(1);
			}
		}
		*/

		/*
		 * Create an instance of an pooling function that operates on a mini-batch of
		 * data at a time.
		 *
		 * The approximate stride is given as the ratio of the input_extents to the output_extents. For example,
		 * the stride in the height dimension is given as height_in/height_out.
		 *
		 * Parameters:
		 *
		 * input_extents: Extents for input activations of size (minibatch_size, depth_in, height_in, width_in).
		 *
		 * pooling_region_extents: Extents that specify the size of each pooling region. This must correspond to a 3-D matrix of
		 *                         size (pooling_size_depth, pooling_size_height, pooling_size_width).
		 *
		 * output_extents: Extents for the output activations for exluding mini-batch size: (depth_out, height_out, width_out). fixme: change name
		 *
		 * fixme: add parameter to specify pooling method. Max, mean, fractional max (Ben Graham's method, http://arxiv.org/pdf/1412.6071.pdf) etc.
		 */
		PoolingLayer(const std::vector<int>& input_extents,
					 const std::vector<int>& pooling_region_extents, const std::vector<int>& output_extents) : 
		m_output { input_extents.at(0), output_extents.at(0), output_extents.at(1), output_extents.at(2) },
			m_output_deltas { input_extents.at(0), output_extents.at(0), output_extents.at(1), output_extents.at(2) },
				m_state { input_extents.at(0), output_extents.at(0), output_extents.at(1), output_extents.at(2), 3},
					m_pooling_region_extents {pooling_region_extents}
		{
			std::cout << "PoolingLayer:" << std::endl;
			if (input_extents.size() != 4) {
				std::cerr << "Input extents have wrong size. Exiting." << std::endl;
				exit(1);
			}
			//if (input_extents.at(0) != output_extents.at(0)) {
			//	std::cerr << "Input and output extents have inconsistent mini-batch size." << std::endl;
			//	exit(1);
			//}
		}

		

		/*
		 * Compute the pooling function in the forward direction.
		 *
		 * The pooling function of the supplied input activations is computed.
		 * The results are stored in the output activations member variable, which
		 * can be obtained by calling get_output().
		 *
		 * Depending on the activation type, calling this method may modify internal
		 * state which will be needed by reverse_pool().
		 */
		void forward_pool(const Matrix& input);

		/*
		 * Compute the pooling function in the reverse direction.
		 *
		 * The reverse pooling function of the output_deltas activations member variable
		 * is computed and the result is stored in the supplied input activations variable.
		 * This method is typically used during the back-propigation step to back-propagate
		 * deltas (errors) through the activation function.
		 */
		void reverse_pool(Matrix& input);

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
		 * Return height of output activations.
		 */
		int get_output_height() const {
			return m_output.extent(2);
		}

		/*
		 * Return width of output activations.
		 */
		int get_output_width() const {
			return m_output.extent(3);
		}

		/*
		 * Return depth of output activations.
		 */
		int get_output_depth() const {
			return m_output.extent(1);
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
		MatrixT<int> m_state;
		const std::vector<int>& m_pooling_region_extents;
	};

}

#endif /* _POOLINGLAYER_H */
