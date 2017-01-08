#ifndef _UNITTESTS_H
#define _UNITTESTS_H
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

#include "Matrix.h"
#include "Utilities.h"

namespace kumozu {


  void test_mat_mult();

  void benchmark_mat_mult();

  void stress_test_forward_prop();

  void test_mat_multiply_left_transpose();

  void test_mat_multiply_left_transpose_accumulate();

  void test_mat_multiply_right_transpose();

  void test_mat_multiply_right_transpose_accumulate();

  void benchmark_mat_multiply_right_transpose();

  void test_Matrix1D();

  void test_Matrix2D();

  void test_Matrix3D();

  void test_Matrix4D();

  void test_Matrix5D();

  void test_Matrix6D();

  void test_view();

  void test_MatrixResize();



  /*
   * Check gradients for the ReLULayer class.
   */
  void test_gradients_ReLULayer();

  void test_copy_to_from_submatrix();

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

  void test_leaky_kmax();

  /*
   * Test ReLU activation function.
   */
  void test_relu();




  void test_PoolingLayer();


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
   * Check 3D kmax functions.
   */
  void test_compute_3d_kmax();




  /*
   * Numercally compute Jacobian with respect to both paramaters and
   * input activations to verify that that the gradients are being
   * computed correctly.
   *
   */
  void test_jacobian_LinearLayer();

  /*
   * Numercally compute Jacobian with respect to both paramaters and
   * input activations to verify that that the gradients are being
   * computed correctly.
   *
   */
  void test_jacobian_LinearLayer_Node();

  void test_jacobian_MaxProduct();

  /*
   * Numercally compute Jacobian with respect to both paramaters and
   * input activations to verify that that the gradients are being
   * computed correctly.
   *
   */
  void test_jacobian_ConvLayer3D();




  void test_select();



  void test_SequentialLayer();


  void test_SequentialLayer2();

  void test_SequentialLayer3();

  void test_SequentialLayer4();

  void test_SequentialLayer_shared_parameters();

  void test_jacobian_ImageToColumnLayer();

  void test_jacobian_BoxActivationFunction();

  void test_jacobian_ColumnActivationFunction();

  void test_jacobian_PoolingLayer();


  void test_MSECostFunction();

  void test_CrossEntropyCostFunction();

  void test_Dropout1D();

  void test_Dropout3D();

  void test_BatchNormalization1D();

  void test_BatchNormalization3D();
  
  void test_Node_shared_parameters();

  void test_Node_shared_parameters2();

  void test_Node_copy_paramaters();

  void test_multi_port_node();

  void test_AdderNode();

  void test_MeanNode();

  void test_MeanNode2();  

  void test_ConcatNode();

  void test_SubtractorNode();

  void test_MultiplyerNode();

  void test_SplitterNode();

  void test_SplitterNode2();

  void test_ExtractorNode();

  void test_rnn_slice();

  void test_simple_rnn();

  void test_char_rnn_minibatch_getter();

  void test_multi_rgb_plots();

  void test_variable();

}
#endif  /* _UNITTESTS_H */
