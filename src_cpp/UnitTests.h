#ifndef _UNITTESTS_H
#define	_UNITTESTS_H
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
#include "Utilities.h"

namespace kumozu {

	void test_mat_mult();

	void benchmark_mat_mult();

	void stress_test_forward_prop();

	void test_mat_multiply_left_transpose();

	void test_mat_multiply_right_transpose();

	void benchmark_mat_multiply_right_transpose();

	void test_Matrix1D();

	void test_Matrix2D();

	void test_Matrix3D();

	void test_Matrix4D();

	void test_Matrix5D();

	void test_Matrix6D();


	
	



	/*
	 * Check gradients for conventional convolutional model.
	 */
	//void test_gradients_2();

	/*
	 * Check gradients for conventional convolutional model with bias.
	 */
	//void test_gradients_3();

	/*
	 * Check gradients for conventional convolutional model with bias, minibatch version.
	 */
	void test_gradients_convolutional_minibatch();




	/*
	 * Check gradients for the ReLULayer class.
	 */
	void test_gradients_ReLULayer();

	/*
	 * Check gradients for the Network2DConvFull class.
	 */
	void test_gradients_Network2DConvFull();

	/*
	 * Check gradients for the Network2DConvFull2L class.
	 */
	void test_gradients_Network2DConvFull2L();

	/*
	 * Check gradients for the Network2DConvFull3L class.
	 */
	void test_gradients_Network2DConvFull3L();

	/*
	 * Test compute_forward_kmax() and
	 * compute_reverse_kmax().
	 */
	void test_compute_kmax();

	/*
	 * Test compute_forward_kmax_v2() and
	 * compute_reverse_kmax_v2().
	 */
	void test_compute_kmax_v2();

	/*
	 * Test ReLU activation function.
	 */
	void test_relu();

	/*
	 * Test compute_maxout_3d().
	 */
	void test_compute_maxout_3d();

	void test_compute_maxout_3d_minibatch();

	/*
	 * Test optimized 2D convolution using BLAS sgemm.
	 *
	 */
	//void test_optimized_convolve_2d_minibatch_deprecated();

	/*
	 * Test optimized 2D convolution using BLAS sgemm.
	 *
	 */
	void test_optimized_convolve_2d_minibatch();

	/*
	 * Test optimized convolutive deltas (back-propagation)
	 * using BLAS sgemm.
	 */
	void test_optimized_convolutive_deltas();

	/*
	 * Test optimized convolutive wieght gradient (back-propagation)
	 * using BLAS sgemm.
	 */
	void test_optimized_weight_grad_convolutive();

	/*
	 * Test optimized 2D convolution using BLAS sgemm.
	 *
	 */
	void test_optimized_convolve_3d_minibatch();

	/*
	 * Test optimized 3d convolutive deltas (back-propagation)
	 * using BLAS sgemm.
	 */
	void test_optimized_3d_convolutive_deltas();

	/*
	 * Test optimized 3d convolutive wieght gradient (back-propagation)
	 * using BLAS sgemm.
	 */
	void test_optimized_3d_weight_grad_convolutive();

	/*
	* Run all unit tests.
	*/
	void run_all_tests();

	/*
	 * Check gradients for the Network3DConvFull3L class.
	 */
	void test_gradients_Network3DConvFull3L();

	/*
	 * Check gradients for the Network3DConvFull3LMaxout class.
	 */
	void test_gradients_Network3DConvFull3LMaxout();

	/*
	 * Check gradients for the Network3DConvFull3LMaxout class.
	 */
	void test_gradients_Network3DConvFull4LMaxout();


	/*
	 * Check 3D kmax functions.
	 */
	void test_compute_3d_kmax();

	/*
	 * Check gradients for the Network3DConvFull4LExp2 class.
	 */
	void test_gradients_Network3DConvFul4LExp2();


	/*
	 * Test Dropout1D class.
	 */
	void test_Dropout1D();

	/*
	 * Test Dropout3D class.
	 */
	void test_Dropout3D();

	/*
	 * Check gradients for the Network2DConv3F1 class.
	 */
	void test_gradients_Network2DConv3F1();

	/*
	 * Check gradients for the Network3DConv3F1 class.
	 */
	void test_gradients_Network3DConv3F1();

}
#endif	/* _UNITTESTS_H */
