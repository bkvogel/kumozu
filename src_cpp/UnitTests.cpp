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



#include "UnitTests.h"
#include "Utilities.h"
#include "Matrix.h"

#include "Constants.h"
#include <string>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include "MatrixIO.h"
#include <chrono>
#include <ctime>

#include "SequentialNetwork.h"
#include "ConvLayer3D.h"
#include "LinearLayer.h"
#include "ImageToColumnLayer.h"
#include "BoxActivationFunction.h"
#include "ColumnActivationFunction.h"
#include "PoolingLayer.h"
#include "MSECostFunction.h"
#include "CrossEntropyCostFunction.h"
#include "Dropout1D.h"
#include "Dropout3D.h"
#include "BatchNormalization1D.h"
#include "BatchNormalization3D.h"

// Uncomment following line to disable assertion checking.
// #define NDEBUG
#include <assert.h>
using namespace std;

namespace kumozu {

  void test_mat_mult() {
    std::cout << "test_mat_mult()..." << std::endl;
    const int rows_A = 5;
    const int cols_A = 4;
    const int cols_B = 3;
    // Compute A = B * C
    MatrixF A(rows_A, cols_A);

    MatrixF B(rows_A, cols_B);
    randomize_uniform(B,-1.0f, 1.0f);
    MatrixF C(cols_B, cols_A);
    randomize_uniform(C,-1.0f, 1.0f);

    // Make copies:
    MatrixF Ac = A;
    MatrixF Bc = B;
    MatrixF Cc = C;

    // Compute using BLAS:
    mat_multiply_blas(A, B, C);
    std::cout << "A = " << std::endl << A << std::endl;
    //cout << "B = " << endl << B << endl;

    // Compute using simple but likely correct version:
    mat_multiply_naive(Ac, Bc, Cc);
    std::cout << "Ac = " << std::endl << Ac << std::endl;
    //cout << "Bc = " << endl << Bc << endl;
    float tolerance = 1.0e-6;
    assert_almost_equal(A, Ac, tolerance);

    std::cout << "done" << std::endl;
  }

  void benchmark_mat_mult()  {
    std::cout << "benchmark_mat_mult()..." << std::endl;

    const int rows_A = 1024; // 511
    const int cols_A = 1024; // 1024
    const int cols_B = 1024; // 2048
    // Compute A = B * C
    MatrixF A(rows_A, cols_A);

    MatrixF B(rows_A, cols_B);
    randomize_uniform(B,-1.0f, 1.0f);
    MatrixF C(cols_B, cols_A);
    randomize_uniform(C,-1.0f, 1.0f);

    // Make copies:
    MatrixF Ac = A;
    MatrixF Bc = B;
    MatrixF Cc = C;

    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    int loop_count = 1000; // 1000
    for (int n = 0; n != loop_count; ++n) {
      mat_multiply_blas(A, B, C); //
      if ((n % 500) == 0) {
        //std::cout << "n = " << n << std::endl;
      }
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds" << std::endl;
    std::cout << flops << " GFLOPS" << std::endl;


    // Check result with slow version:
    mat_multiply_naive(Ac, Bc, Cc);

    //cout << "Ac = " << endl << Ac << endl;


    float tolerance = 1.0e-4;
    assert_almost_equal(A, Ac, tolerance);
    std::cout << "done" << std::endl;
  }


  void stress_test_forward_prop()   {
    std::cout << "stress_test_forward_prop()..." << std::endl;

    const int rows_A = 1*1024; // 511
    const int cols_A = 1*1024; // 1024
    const int cols_B = 1*1024; // 2048
    // Compute A = B * C
    MatrixF A(rows_A, cols_A);

    MatrixF B(rows_A, cols_B);
    randomize_uniform(B,-1.0f, 1.0f);
    MatrixF C(cols_B, cols_A);
    randomize_uniform(C,-1.0f, 1.0f);

    // Make copies:
    MatrixF Ac = A;
    MatrixF Bc = B;
    MatrixF Cc = C;
    Matrix<int> out_indices(A.extent(0), A.extent(1));
    MatrixF out_values(A.extent(0), A.extent(1));

    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    int loop_count = 1000; // 1000
    for (int n = 0; n != loop_count; ++n) {
      mat_multiply_blas(A, B, C); //
      //scale(A, A, 0.98765f);
      compute_forward_relu(A, out_values, out_indices);
      //compute_reverse_relu(Ac, out_values, out_indices);
      if ((n % 500) == 0) {
        std::cout << "n = " << n << std::endl;
      }
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds" << std::endl;
    std::cout << flops << " GFLOPS not including activation function." << std::endl;


    std::cout << "done" << std::endl;
  }


  void test_mat_multiply_left_transpose() {
    std::cout << "test_mat_multiply_left_transpose()..." << std::endl;
    const int rows_A = 16;
    const int cols_A = 16 * 2;
    const int cols_B = 16 * 3;
    // Compute A = B^T * C
    MatrixF A(rows_A, cols_A);

    MatrixF B(cols_B, rows_A);
    randomize_uniform(B,-1.0f, 1.0f);
    MatrixF C(cols_B, cols_A);
    randomize_uniform(C,-1.0f, 1.0f);

    // Make copies:
    MatrixF Ac = A;
    MatrixF Bc = B;
    MatrixF Cc = C;

    mat_multiply_left_transpose(A, B, C);
    std::cout << "A = " << std::endl << A << std::endl;
    //cout << "B = " << endl << B << endl;

    mat_multiply_left_transpose_naive(Ac, Bc, Cc);
    std::cout << "Ac = " << std::endl << Ac << std::endl;
    //cout << "Bc = " << endl << Bc << endl;
    float tolerance = 1.0e-4;
    assert_almost_equal(A, Ac, tolerance);

    std::cout << "done" << std::endl;
  }

  void test_mat_multiply_right_transpose() {
    std::cout << "test_mat_multiply_right_transpose()..." << std::endl;
    const int rows_A = 16;
    const int cols_A = 16 * 2;
    const int cols_B = 16 * 3;
    // Compute A = B * C^T
    MatrixF A(rows_A, cols_A);

    MatrixF B(rows_A, cols_B);
    randomize_uniform(B,-1.0f, 1.0f);
    MatrixF C(cols_A, cols_B);
    randomize_uniform(C,-1.0f, 1.0f);

    // Make copies:
    MatrixF Ac = A;
    MatrixF Bc = B;
    MatrixF Cc = C;

    mat_multiply_right_transpose(A, B, C);
    std::cout << "A = " << std::endl << A << std::endl;
    //cout << "B = " << endl << B << endl;

    mat_multiply_right_transpose_naive(Ac, Bc, Cc);
    std::cout << "Ac = " << std::endl << Ac << std::endl;
    //cout << "Bc = " << endl << Bc << endl;
    float tolerance = 1.0e-4;
    assert_almost_equal(A, Ac, tolerance);

    std::cout << "done" << std::endl;
  }


  void benchmark_mat_multiply_right_transpose() {
    //const int rows_A = 784;
    //const int cols_A = 1*100;
    //const int cols_B = 1*50;

    const int rows_A = 784;
    const int cols_A = 1*100;
    const int cols_B = 1*50;
    // Compute A = B * C^T
    MatrixF A(rows_A, cols_A);

    MatrixF B(rows_A, cols_B);
    randomize_uniform(B,-1.0f, 1.0f);
    MatrixF C(cols_A, cols_B);
    randomize_uniform(C,-1.0f, 1.0f);

    // Make copies:
    MatrixF Ac = A;
    MatrixF Bc = B;
    MatrixF Cc = C;

    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    int loop_count = 10000; // 1000
    for (int n = 0; n != loop_count; ++n) {
      //mat_multiply_blas(B, A, C); //
      mat_multiply_right_transpose(A, B, C);
      //mat_multiply_right_transpose_naive(Ac, Bc, Cc);
      if ((n % 500) == 0) {
        std::cout << "n = " << n << std::endl;
      }
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds" << std::endl;
    std::cout << flops << " GFLOPS" << std::endl;


    //std::cout << "A = " << std::endl << A << std::endl;
    //cout << "B = " << endl << B << endl;

    mat_multiply_right_transpose_naive(Ac, Bc, Cc);
    //std::cout << "Ac = " << std::endl << Ac << std::endl;
    //cout << "Bc = " << endl << Bc << endl;
    float tolerance = 1.0e-3;
    assert_almost_equal(A, Ac, tolerance);

    std::cout << "done" << std::endl;
  }

  void test_MatrixF1D() {
    cout << "Testing test_MatrixF1D...";
    const int dim0 = 5;
    MatrixF mat_1d(dim0);
    // Fill with data:
    for (int i=0; i < dim0; ++i) {
      mat_1d(i) = i*i - 1234;
    }
    // Read and verify data:
    for (int i=0; i < dim0; ++i) {
      float read_val = mat_1d(i);
      float true_val = i*i - 1234;
      assert_almost_equal(read_val, true_val, 1e-4f);
    }
    cout << "PASSED" << endl;
  }

  void test_MatrixF2D() {
    cout << "Testing test_MatrixF2D...";
    const int dim0 = 3;
    const int dim1 = 4;
    MatrixF mat_2d(dim0, dim1);
    // Fill with data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        mat_2d(i,j) = i*i + j*j - 1234;
      }
    }
    // Read and verify data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        float read_val = mat_2d(i,j);
        float true_val = i*i + j*j - 1234;
        assert_almost_equal(read_val, true_val, 1e-4f);
      }
    }
    cout << "PASSED" << endl;
  }

  void test_MatrixF3D() {
    cout << "Testing test_MatrixF3D...";
    const int dim0 = 2;
    const int dim1 = 3;
    const int dim2 = 4;
    MatrixF mat_3d(dim0, dim1, dim2);
    // Fill with data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          mat_3d(i,j,k) = i*i + j*j + k*k - 1234;
        }
      }
    }
    // Read and verify data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          float read_val = mat_3d(i,j,k);
          float true_val = i*i + j*j + k*k - 1234;
          assert_almost_equal(read_val, true_val, 1e-4f);
        }
      }
    }

    cout << "PASSED" << endl;

  }

  void test_MatrixF4D() {
    cout << "Testing test_MatrixF4D...";
    const int dim0 = 2;
    const int dim1 = 3;
    const int dim2 = 4;
    const int dim3 = 5;
    MatrixF mat_4d(dim0, dim1, dim2, dim3);
    // Fill with data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          for (int l = 0; l < dim3; ++l) {
            mat_4d(i,j,k,l) = i*i + j*j + k*k + l*l - 1234;
          }
        }
      }
    }
    //cout << "matrix data = " << endl << mat_4d << endl;
    // Read and verify data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          for (int l = 0; l < dim3; ++l) {
            float read_val = mat_4d(i,j,k,l);
            float true_val = i*i + j*j + k*k + l*l - 1234;
            assert_almost_equal(read_val, true_val, 1e-4f);
          }
        }
      }
    }

    cout << "PASSED" << endl;

  }

  void test_MatrixF5D() {
    cout << "Testing test_MatrixF5D...";
    const int dim0 = 2;
    const int dim1 = 3;
    const int dim2 = 4;
    const int dim3 = 5;
    const int dim4 = 6;
    MatrixF mat_5d(dim0, dim1, dim2, dim3, dim4);
    // Fill with data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          for (int l = 0; l < dim3; ++l) {
            for (int m = 0; m < dim4; ++m) {
              mat_5d(i,j,k,l,m) = i*i + j*j + k*k + l*l + m*m - 1234;
            }
          }
        }
      }
    }
    //cout << "matrix data = " << endl << mat_4d << endl;
    // Read and verify data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          for (int l = 0; l < dim3; ++l) {
            for (int m = 0; m < dim4; ++m) {
              float read_val = mat_5d(i,j,k,l,m);
              float true_val = i*i + j*j + k*k + l*l + m*m - 1234;
              assert_almost_equal(read_val, true_val, 1e-4f);
            }
          }
        }
      }
    }
    cout << "PASSED" << endl;
  }

  void test_MatrixF6D() {
    cout << "Testing test_MatrixF6D...";
    const int dim0 = 2;
    const int dim1 = 3;
    const int dim2 = 4;
    const int dim3 = 5;
    const int dim4 = 6;
    const int dim5 = 7;
    MatrixF mat_6d(dim0, dim1, dim2, dim3, dim4, dim5);
    // Fill with data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          for (int l = 0; l < dim3; ++l) {
            for (int m = 0; m < dim4; ++m) {
              for (int n = 0; n < dim5; ++n) {
                mat_6d(i,j,k,l,m,n) = i*i + j*j + k*k + l*l + m*m + n*n - 1234;
              }
            }
          }
        }
      }
    }
    //cout << "matrix data = " << endl << mat_4d << endl;
    // Read and verify data:
    for (int i=0; i < dim0; ++i) {
      for (int j = 0; j < dim1; ++j) {
        for (int k = 0; k < dim2; ++k) {
          for (int l = 0; l < dim3; ++l) {
            for (int m = 0; m < dim4; ++m) {
              for (int n = 0; n < dim5; ++n) {
                float read_val = mat_6d(i,j,k,l,m,n);
                float true_val = i*i + j*j + k*k + l*l + m*m + n*n - 1234;
                assert_almost_equal(read_val, true_val, 1e-4f);
              }
            }
          }
        }
      }
    }
    cout << "PASSED" << endl;
  }



  void test_MatrixResize() {
    cout << "test_MatrixResize()..." << endl;

    MatrixF A; // size 0.

    vector<int> extents = {2, 3};
    A.resize(extents);
    set_value(A, 4.8f);
    cout << "A size = " << A.size() << endl;
    cout << "A = " << endl << A << endl;

    A.resize(12);
    cout << "A size = " << A.size() << endl;
    cout << "A = " << endl << A << endl;

    A.resize(3, 2);
    cout << "A size = " << A.size() << endl;
    cout << "A = " << endl << A << endl;

    A.resize(3, 2, 3);
    set_value(A, 1.0f);
    cout << "A size = " << A.size() << endl;
    cout << "A = " << endl << A << endl;

    cout << "PASSED" << endl;    
  }



  void test_compute_kmax() {
    cout << "test_compute_kmax()..." << endl;
    const int M = 1024;
    const int N = 256;
    const int partition_count = 1;
    const int k = 512; // 294
    MatrixF kmax_in(M, N);
    randomize_uniform(kmax_in, -1.0f, 1.0f);
    //cout << "kmax_in = " << endl << kmax_in << endl;
    MatrixF kmax_out_values(M, N);
    Matrix<int> kmax_out_indices(k*partition_count, N);
    // Compute forward-direction kmax:
    compute_forward_kmax(kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
    //cout << "kmax_out_values = " << endl << kmax_out_values << endl;
    //cout << "kmax_out_indices = " << endl << kmax_out_indices << endl;

    MatrixF other_kmax_in(M, N);
    // Compute reverse-direction kmax:
    compute_reverse_kmax(other_kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
    //cout << "Updated kmax_in = " << endl << other_kmax_in << endl;
    assert_almost_equal(kmax_out_values, other_kmax_in, 1e-3f);

    cout << "PASSED" << endl;
  }

  void test_compute_kmax_v2() {
    cout << "test_compute_kmax_v2()..." << endl;
    const int M = 1024;
    const int N = 256;
    const int partition_count = 1;
    const int k = 512; // 294
    MatrixF kmax_in(M, N);
    randomize_uniform(kmax_in, -1.0f, 1.0f);
    //cout << "kmax_in = " << endl << kmax_in << endl;
    MatrixF kmax_out_values(M, N);
    Matrix<int> kmax_out_indices(M, N);
    // Compute forward-direction kmax:
    compute_forward_kmax_v2(kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
    //cout << "kmax_out_values = " << endl << kmax_out_values << endl;
    //cout << "kmax_out_indices = " << endl << kmax_out_indices << endl;

    MatrixF other_kmax_in(M, N);
    // Compute reverse-direction kmax:
    compute_reverse_kmax_v2(other_kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
    //cout << "Updated kmax_in = " << endl << other_kmax_in << endl;
    assert_almost_equal(kmax_out_values, other_kmax_in, 1e-3f);

    cout << "PASSED" << endl;
  }



  void test_relu() {
    cout << "test_relu()..." << endl;
    const int M = 50;
    const int N = 40;
    MatrixF in_vals(M, N);
    randomize_uniform(in_vals, -1.0f, 1.0f);
    //cout << "in_vals = " << endl << in_vals << endl;
    MatrixF out_values(M, N);
    Matrix<int> out_indices(M, N);
    // Compute forward-direction ReLU:
    compute_forward_relu(in_vals, out_values, out_indices);
    //cout << "out_values = " << endl << out_values << endl;
    //cout << "out_indices = " << endl << out_indices << endl;

    MatrixF other_in_vals(M, N);
    // Compute reverse-direction ReLU:
    compute_reverse_relu(other_in_vals, out_values, out_indices);
    //cout << "Updated in_vals = " << endl << other_in_vals << endl;
    assert_almost_equal(out_values, other_in_vals, 1e-3f);

    cout << "PASSED" << endl;
  }


  void test_PoolingLayer() {
    cout << "test_PoolingLayer..." << endl;



    cout << "PASSED" << endl;
  }




  void test_optimized_convolutive_deltas() {
    cout << "test_optimized_convolutive_deltas()..." << endl;
    const float pass_relative_error = 5e-3f; // Relative error must be below this to pass.
    /*
    // Number of samples in a mini-batch.
    const int minibatch_size = 128;
    // Create X with random values.
    const int image_height = 32;
    const int image_width = 32;
    const int conv_filter_height = 5; // Convolution filter height
    const int conv_filter_width = 5; // Convolution filter width
    const int filter_count = 64; // 1 // Number of convolutional filters.
    */

    // Number of samples in a mini-batch.
    const int minibatch_size = 32;
    // Create X with random values.
    const int image_height = 16;
    const int image_width = 16;
    const int conv_filter_height = 3; // Convolution filter height
    const int conv_filter_width = 3; // Convolution filter width
    const int filter_count = 2; // 1 // Number of convolutional filters.


    MatrixF deltas_Z2(minibatch_size, filter_count, image_height, image_width);
    randomize_uniform(deltas_Z2, -1.0f, 1.0f);
    MatrixF W(filter_count, conv_filter_height, conv_filter_width);
    randomize_uniform(W, -1.0f, 1.0f);
    MatrixF bias(filter_count);
    randomize_uniform(bias, -1.0f, 1.0f);
    MatrixF deltas_A1_true(minibatch_size, image_height, image_width);


    cout << "Running naive convolution..." << endl;
    // Warm up:
    compute_convolutive_deltas_minibatch(deltas_A1_true, W, deltas_Z2);
    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    const int loop_count = 10; // 1000
    for (int n = 0; n != loop_count; ++n) {
      //cout << "naive: n = " << n << endl;
      // Assume this version is correct.
      compute_convolutive_deltas_minibatch(deltas_A1_true, W, deltas_Z2);
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    //double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds for naive version." << std::endl;
    //std::cout << flops << " GFLOPS" << std::endl;


    // This will be the result of the optimized version.
    MatrixF deltas_A1_optimized(minibatch_size, image_height, image_width);

    // Allocate temporary matrices that are needed by the optimized convolution function.
    MatrixF temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
    randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
    MatrixF temp_deltas_A1(image_height*image_width*minibatch_size, conv_filter_height*conv_filter_width + 1);
    randomize_uniform(temp_deltas_A1, -1.0f, 1.0f);
    MatrixF temp_W(conv_filter_height*conv_filter_width + 1, filter_count);
    randomize_uniform(temp_W, -1.0f, 1.0f);
    cout << "Running optimized convolutive back-progatation to compute deltas_A1..." << endl;
    auto t0_opt = high_resolution_clock::now();
    for (int n = 0; n != loop_count; ++n) {
      // Compute deltas_A1_optimized.
      //compute_convolutive_deltas_minibatch(deltas_A1_optimized, W, deltas_Z2);
      compute_convolutive_deltas_minibatch_optimized(deltas_A1_optimized, W, deltas_Z2, temp_deltas_Z2,
                                                     temp_deltas_A1, temp_W);
    }
    auto t1_opt = high_resolution_clock::now();
    auto time_in_msec_opt = duration_cast<milliseconds>(t1_opt - t0_opt).count();
    std::cout << time_in_msec_opt << " milliseconds for optimized version." << std::endl;


    //cout << "deltas_A1_true = " << endl << deltas_A1_true << endl;
    //cout << "deltas_A1_optimized = " << endl << deltas_A1_optimized << endl;
    // Check deltas_A1_optimized against the assumed correct value in deltas_A1_true.
    const float rel_error = relative_error(deltas_A1_true, deltas_A1_optimized);
    cout << "relative error = " << rel_error << endl;
    assert_almost_equal(rel_error, 0.0f, pass_relative_error);

    cout << "PASSED" << endl;

  }


  void test_optimized_weight_grad_convolutive()  {
    cout << "test_optimized_weight_grad_convolutive()..." << endl;
    const float pass_relative_error = 5e-3f; // Relative error must be below this to pass.

    /*
    // Number of samples in a mini-batch.
    const int minibatch_size = 128;
    // Create X with random values.
    const int image_height = 32;
    const int image_width = 32;
    const int conv_filter_height = 5; // Convolution filter height
    const int conv_filter_width = 5; // Convolution filter width
    const int filter_count = 64; // 1 // Number of convolutional filters.
    */


    // Number of samples in a mini-batch.
    const int minibatch_size = 32;
    // Create X with random values.
    const int image_height = 16;
    const int image_width = 16;
    const int conv_filter_height = 3; // Convolution filter height
    const int conv_filter_width = 3; // Convolution filter width
    const int filter_count = 2; // 1 // Number of convolutional filters.


    MatrixF deltas_Z2(minibatch_size, filter_count, image_height, image_width);
    randomize_uniform(deltas_Z2, -1.0f, 1.0f);
    MatrixF grad_W_true(filter_count, conv_filter_height, conv_filter_width);
    //MatrixF bias(filter_count);
    //randomize_uniform(bias, -1.0f, 1.0f); //
    MatrixF A1(minibatch_size, image_height, image_width);
    randomize_uniform(A1, -1.0f, 1.0f);

    cout << "Running naive weight gradient convolutive back-propagation..." << endl;
    // Warm up:
    compute_weight_grad_convolutive_minibatch(grad_W_true, deltas_Z2, A1);
    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    const int loop_count = 10; // 1000
    for (int n = 0; n != loop_count; ++n) {
      //cout << "naive: n = " << n << endl;
      // Assume this version is correct.
      compute_weight_grad_convolutive_minibatch(grad_W_true, deltas_Z2, A1);
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    //double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds for naive version." << std::endl;
    //std::cout << flops << " GFLOPS" << std::endl;


    // This will be the result of the optimized version.
    MatrixF grad_W_optimized(filter_count, conv_filter_height, conv_filter_width);

    // Allocate temporary matrices that are needed by the optimized convolution function.
    MatrixF temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
    randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
    MatrixF temp_A1(image_height*image_width*minibatch_size, conv_filter_height*conv_filter_width + 1);
    randomize_uniform(temp_A1, -1.0f, 1.0f);
    MatrixF temp_grad_W(conv_filter_height*conv_filter_width + 1, filter_count);
    randomize_uniform(temp_grad_W, -1.0f, 1.0f);
    cout << "Running optimized convolutive back-progatation to compute deltas_A1..." << endl;
    auto t0_opt = high_resolution_clock::now();
    for (int n = 0; n != loop_count; ++n) {
      compute_weight_grad_convolutive_minibatch_optimized(grad_W_optimized, deltas_Z2, A1,
                                                          temp_deltas_Z2, temp_A1, temp_grad_W);
    }
    auto t1_opt = high_resolution_clock::now();
    auto time_in_msec_opt = duration_cast<milliseconds>(t1_opt - t0_opt).count();
    std::cout << time_in_msec_opt << " milliseconds for optimized version." << std::endl;

    //cout << "grad_W_optimized = " << grad_W_optimized << endl;
    //cout << "deltas_A1_true = " << endl << deltas_A1_true << endl;
    //cout << "deltas_A1_optimized = " << endl << deltas_A1_optimized << endl;
    // Check deltas_A1_optimized against the assumed correct value in deltas_A1_true.
    const float rel_error = relative_error(grad_W_true, grad_W_optimized);
    cout << "relative error = " << rel_error << endl;
    assert_almost_equal(rel_error, 0.0f, pass_relative_error);

    cout << "PASSED" << endl;

  }

  void test_optimized_convolve_3d_minibatch() {
    cout << " test_optimized_convolve_3d_minibatch()..." << endl;
    const float pass_relative_error = 5e-3f; // Relative error must be below this to pass.

    // Benchmarking parameters:
    /*
    // Number of samples in a mini-batch.
    const int minibatch_size = 128;
    // Create X with random values.
    const int image_height = 32;
    const int image_width = 32;
    const int image_depth = 8;
    const int conv_filter_height = 5; // Convolution filter height
    const int conv_filter_width = 5; // Convolution filter width
    const int filter_count = 64; // 1 // Number of convolutional filters.
    */

    // Correctness testing parameters:
    // Number of samples in a mini-batch.
    const int minibatch_size = 2;
    // Create X with random values.
    const int image_height = 5;
    const int image_width = 5;
    const int image_depth = 2;
    const int conv_filter_height = 3; // Convolution filter height
    const int conv_filter_width = 3; // Convolution filter width
    const int filter_count = 2; // 1 // Number of convolutional filters.

    // Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
    MatrixF Z2_true(minibatch_size, filter_count, image_height, image_width);
    MatrixF W(filter_count, image_depth, conv_filter_height, conv_filter_width);
    randomize_uniform(W, -1.0f, 1.0f);
    MatrixF bias(filter_count);
    randomize_uniform(bias, -1.0f, 1.0f);
    MatrixF A1(minibatch_size, image_depth, image_height, image_width);
    randomize_uniform(A1, -1.0f, 1.0f);

    cout << "Running naive convolution..." << endl;
    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    const int loop_count = 10; // 10
    for (int n = 0; n != loop_count; ++n) {
      //cout << "naive: n = " << n << endl;
      // Assume this version is correct.
      // Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
      convolve_3d_filter_with_bias_minibatch(Z2_true, W, bias, A1);
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    //double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds for naive version." << std::endl;
    //std::cout << flops << " GFLOPS" << std::endl;


    // This will be the result of the optimized version.
    MatrixF Z2_optimized(minibatch_size, filter_count, image_height, image_width);

    // Allocate temporary matrices that are needed by the optimized convolution function.
    MatrixF temp_Z2(image_height*image_width*minibatch_size, filter_count);
    MatrixF temp_A1(image_height*image_width*minibatch_size, image_depth*conv_filter_height*conv_filter_width + 1);
    MatrixF temp_W(image_depth*conv_filter_height*conv_filter_width + 1, filter_count);
    cout << "Running optimized convolution..." << endl;
    auto t0_opt = high_resolution_clock::now();
    for (int n = 0; n != loop_count; ++n) {
      //cout << "opt: n = " << n << endl;
      //convolve_2d_filter_with_bias_minibatch(Z2_true, W, bias, A1);
      // Compute Z2_optimized
      convolve_3d_filter_with_bias_minibatch_optimized(Z2_optimized, W, bias, A1, temp_Z2, temp_A1, temp_W);
    }
    auto t1_opt = high_resolution_clock::now();
    auto time_in_msec_opt = duration_cast<milliseconds>(t1_opt - t0_opt).count();
    std::cout << time_in_msec_opt << " milliseconds for optimized version." << std::endl;


    // Check Z2_optimized against the assumed correct value in Z2_true.
    //cout << "Z2_true = " << endl << Z2_true << endl;
    //cout << "Z2_optimized = " << endl << Z2_optimized << endl;
    const float rel_error = relative_error(Z2_true, Z2_optimized);
    cout << "relative error = " << rel_error << endl;
    assert_almost_equal(rel_error, 0.0f, pass_relative_error);

    cout << "PASSED" << endl;
  }


  void test_optimized_3d_convolutive_deltas()  {
    cout << "test_optimized_3d_convolutive_deltas()..." << endl;
    const float pass_relative_error = 5e-3f; // Relative error must be below this to pass.

    // Benchmarking parameters:
    /*
    // Number of samples in a mini-batch.
    const int minibatch_size = 128;
    // Create X with random values.
    const int image_height = 32;
    const int image_width = 32;
    const int image_depth = 8;
    const int conv_filter_height = 5; // Convolution filter height
    const int conv_filter_width = 5; // Convolution filter width
    const int filter_count = 64; // 1 // Number of convolutional filters.
    */


    // Correctness testing parameters:
    // Number of samples in a mini-batch.
    const int minibatch_size = 2;
    // Create X with random values.
    const int image_height = 5;
    const int image_width = 5;
    const int image_depth = 2;
    const int conv_filter_height = 3; // Convolution filter height
    const int conv_filter_width = 3; // Convolution filter width
    const int filter_count = 2; // 1 // Number of convolutional filters.


    MatrixF deltas_Z2(minibatch_size, filter_count, image_height, image_width);
    randomize_uniform(deltas_Z2, -1.0f, 1.0f);
    MatrixF W(filter_count, image_depth, conv_filter_height, conv_filter_width);
    randomize_uniform(W, -1.0f, 1.0f);
    MatrixF bias(filter_count);
    randomize_uniform(bias, -1.0f, 1.0f);
    MatrixF deltas_A1_true(minibatch_size, image_depth, image_height, image_width);


    cout << "Running naive convolution..." << endl;
    // Warm up:
    compute_3d_convolutive_deltas_minibatch(deltas_A1_true, W, deltas_Z2);
    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    const int loop_count = 10; // 1000
    for (int n = 0; n != loop_count; ++n) {
      //cout << "naive: n = " << n << endl;
      // Assume this version is correct.
      compute_3d_convolutive_deltas_minibatch(deltas_A1_true, W, deltas_Z2);
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    //double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds for naive version." << std::endl;
    //std::cout << flops << " GFLOPS" << std::endl;


    // This will be the result of the optimized version.
    MatrixF deltas_A1_optimized(minibatch_size, image_depth, image_height, image_width);

    // Allocate temporary matrices that are needed by the optimized convolution function.
    MatrixF temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
    randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
    MatrixF temp_deltas_A1(image_height*image_width*minibatch_size, image_depth*conv_filter_height*conv_filter_width + 1);
    randomize_uniform(temp_deltas_A1, -1.0f, 1.0f);
    MatrixF temp_W(image_depth*conv_filter_height*conv_filter_width + 1, filter_count);
    randomize_uniform(temp_W, -1.0f, 1.0f);
    cout << "Running optimized convolutive back-progatation to compute deltas_A1..." << endl;
    auto t0_opt = high_resolution_clock::now();
    for (int n = 0; n != loop_count; ++n) {
      // Compute deltas_A1_optimized.
      compute_3d_convolutive_deltas_minibatch_optimized(deltas_A1_optimized, W, deltas_Z2, temp_deltas_Z2,
                                                        temp_deltas_A1, temp_W);
    }
    auto t1_opt = high_resolution_clock::now();
    auto time_in_msec_opt = duration_cast<milliseconds>(t1_opt - t0_opt).count();
    std::cout << time_in_msec_opt << " milliseconds for optimized version." << std::endl;


    //cout << "deltas_A1_true = " << endl << deltas_A1_true << endl;
    //cout << "deltas_A1_optimized = " << endl << deltas_A1_optimized << endl;
    // Check deltas_A1_optimized against the assumed correct value in deltas_A1_true.
    const float rel_error = relative_error(deltas_A1_true, deltas_A1_optimized);
    cout << "relative error = " << rel_error << endl;
    assert_almost_equal(rel_error, 0.0f, pass_relative_error);

    cout << "PASSED" << endl;

  }

  void test_optimized_3d_weight_grad_convolutive()   {
    cout << "test_optimized_3d_weight_grad_convolutive()..." << endl;
    const float pass_relative_error = 5e-3f; // Relative error must be below this to pass.

    // Benchmarking parameters
    /*
    // Number of samples in a mini-batch.
    const int minibatch_size = 128;
    // Create X with random values.
    const int image_height = 32;
    const int image_width = 32;
    const int image_depth = 8;
    const int conv_filter_height = 5; // Convolution filter height
    const int conv_filter_width = 5; // Convolution filter width
    const int filter_count = 64; // 1 // Number of convolutional filters.
    */


    // Correctness testing parameters:

    // Number of samples in a mini-batch.
    const int minibatch_size = 32;
    // Create X with random values.
    const int image_height = 16;
    const int image_width = 16;
    const int image_depth = 8;
    const int conv_filter_height = 3; // Convolution filter height
    const int conv_filter_width = 3; // Convolution filter width
    const int filter_count = 2; // 1 // Number of convolutional filters.


    MatrixF deltas_Z2(minibatch_size, filter_count, image_height, image_width);
    randomize_uniform(deltas_Z2, -1.0f, 1.0f);
    MatrixF grad_W_true(filter_count, image_depth, conv_filter_height, conv_filter_width);
    //MatrixF bias(filter_count);
    //randomize_uniform(bias, -1.0f, 1.0f); //
    MatrixF A1(minibatch_size, image_depth, image_height, image_width);
    randomize_uniform(A1, -1.0f, 1.0f);

    cout << "Running naive weight gradient convolutive back-propagation..." << endl;
    // Warm up:
    compute_3d_weight_grad_convolutive_minibatch(grad_W_true, deltas_Z2, A1);
    // Start timer here.
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    const int loop_count = 10; // 1000
    for (int n = 0; n != loop_count; ++n) {
      //cout << "naive: n = " << n << endl;
      // Assume this version is correct.
      compute_3d_weight_grad_convolutive_minibatch(grad_W_true, deltas_Z2, A1);
    }
    // Stop timer here.
    auto t1 = high_resolution_clock::now();

    auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
    //double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
    std::cout << time_in_msec << " milliseconds for naive version." << std::endl;
    //std::cout << flops << " GFLOPS" << std::endl;


    // This will be the result of the optimized version.
    MatrixF grad_W_optimized(filter_count, image_depth, conv_filter_height, conv_filter_width);

    // Allocate temporary matrices that are needed by the optimized convolution function.
    MatrixF temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
    randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
    MatrixF temp_A1(image_height*image_width*minibatch_size, image_depth*conv_filter_height*conv_filter_width + 1);
    randomize_uniform(temp_A1, -1.0f, 1.0f);
    MatrixF temp_grad_W(image_depth*conv_filter_height*conv_filter_width + 1, filter_count);
    randomize_uniform(temp_grad_W, -1.0f, 1.0f);
    cout << "Running optimized convolutive back-progatation to compute deltas_A1..." << endl;
    auto t0_opt = high_resolution_clock::now();
    for (int n = 0; n != loop_count; ++n) {
      compute_3d_weight_grad_convolutive_minibatch_optimized(grad_W_optimized, deltas_Z2, A1,
                                                             temp_deltas_Z2, temp_A1, temp_grad_W);
    }
    auto t1_opt = high_resolution_clock::now();
    auto time_in_msec_opt = duration_cast<milliseconds>(t1_opt - t0_opt).count();
    std::cout << time_in_msec_opt << " milliseconds for optimized version." << std::endl;

    //cout << "grad_W_optimized = " << grad_W_optimized << endl;
    //cout << "deltas_A1_true = " << endl << deltas_A1_true << endl;
    //cout << "deltas_A1_optimized = " << endl << deltas_A1_optimized << endl;
    // Check deltas_A1_optimized against the assumed correct value in deltas_A1_true.
    const float rel_error = relative_error(grad_W_true, grad_W_optimized);
    cout << "relative error = " << rel_error << endl;
    assert_almost_equal(rel_error, 0.0f, pass_relative_error);

    cout << "PASSED" << endl;
  }




  void test_compute_3d_kmax() {
    cout << "test_compute_3d_kmax()..." << endl;
    const int minibatch_size = 2;
    const int depth = 4;
    const int height = 4;
    const int width = 4;
    const int box_depth = 3;
    const int box_height = 3;
    const int box_width = 3;
    const int k = 2;

    MatrixF kmax_in(minibatch_size, depth, height, width);
    MatrixF kmax_out(minibatch_size, depth, height, width);
    Matrix<int> kmax_state(minibatch_size, depth, height, width);

    randomize_uniform(kmax_in, -1.0f, 1.0f);
    cout << "kmax_in = " << endl << kmax_in << endl;

    // Compute forward-direction kmax:
    forward_3d_kmax(kmax_in, kmax_out, kmax_state, box_depth, box_height, box_width, k);
    cout << "kmax_out = " << endl << kmax_out << endl;
    cout << "kmax_state = " << endl << kmax_state << endl;

    MatrixF other_kmax_in(minibatch_size, depth, height, width);
    // Compute reverse-direction kmax:
    //compute_reverse_kmax(other_kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
    reverse_3d_kmax(other_kmax_in, kmax_out, kmax_state);
    cout << "Updated kmax_in = " << endl << other_kmax_in << endl;
    assert_almost_equal(kmax_out, other_kmax_in, 1e-3f);

    cout << "PASSED" << endl;
  }

  void test_select() {
    cout << "test_select()..." << endl;

    MatrixF X(4,5);
    randomize_uniform(X, 0.0f, 0.01f);
    cout << "X = " << endl << X << endl;
    MatrixF Y = select(X, 1, 2);
    cout << "Y = " << endl << Y << endl;
  }

  void test_jacobian_ConvLayer3D() {
    cout << "test_jacobian_ConvLayer3D()..." << endl;
    const int minibatch_size = 4;
    const int image_depth = 3;
    const int image_height = 13;
    const int image_width = 15;
    const int filter_count = 5;
    const int filter_height = 3;
    const int filter_width = 4;

    const vector<int> input_extents = {minibatch_size, image_depth, image_height, image_width};
    ConvLayer3D layer(filter_count, filter_height, filter_width, "Conv Layer");

    // Check weights gradients.
    layer.check_jacobian_weights(input_extents);
    // Now check bias gradients
    layer.check_jacobian_bias(input_extents);
    // Now check input error gradients
    layer.check_jacobian_input_error(input_extents);

  }


  void test_SequentialNetwork() {
    cout << "test_SequentialNetwork()..." << endl;
    const int minibatch_size = 4;
    const int image_depth = 3;
    const int image_height = 13;
    const int image_width = 15;
    const int filter_count = 5;
    const int filter_height = 3;
    const int filter_width = 4;

    SequentialNetwork seq_net("sequential network 1");
    ConvLayer3D conv_layer1(filter_count, filter_height, filter_width, "Conv Layer 1");
    seq_net.add_layer(conv_layer1);

    const vector<int> input_extents = {minibatch_size, image_depth, image_height, image_width};
    // Check weights gradients.
    seq_net.check_jacobian_weights(input_extents);
    // Now check bias gradients
    seq_net.check_jacobian_bias(input_extents);
    // Now check input error gradients
    seq_net.check_jacobian_input_error(input_extents);

    //MatrixF input_activations(minibatch_size, image_depth, image_height, image_width);
    //seq_net.forward(input_activations);
    //seq_net.forward(input_activations);

    //cout << "PASSED" << endl;
  }

  void test_SequentialNetwork2() {
    cout << "test_SequentialNetwork2()..." << endl;
    const int minibatch_size = 4;
    const int image_depth = 3;
    const int image_height = 13;
    const int image_width = 15;
    const int filter_count = 5;
    const int filter_height = 3;
    const int filter_width = 4;

    const int dim_output = 7;

    SequentialNetwork net("sequential network 1");
    ConvLayer3D conv_layer1(filter_count, filter_height, filter_width, "Conv Layer 1");
    net.add_layer(conv_layer1);
    BoxActivationFunction box_activation_layer1(BoxActivationFunction::ACTIVATION_TYPE::leakyReLU, "Box Activation Function 1");
    net.add_layer(box_activation_layer1);
    const vector<int> pooling_region_extents = {1, 3, 3};
    const vector<int> pooling_region_step_sizes = {1, 2, 2};
    PoolingLayer pooling_layer1(pooling_region_extents, pooling_region_step_sizes, "Pooling Layer 1");
    net.add_layer(pooling_layer1);
    ImageToColumnLayer layer2("Image To Column Layer 1");
    net.add_layer(layer2);
    LinearLayer linear_laye1(dim_output, "Linear Layer 1");
    net.add_layer(linear_laye1);
    ColumnActivationFunction column_activation_layer1(ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU, "Column Activation Function 1");
    net.add_layer(column_activation_layer1);

    const vector<int> input_extents = {minibatch_size, image_depth, image_height, image_width};
    // Check weights gradients.
    net.check_jacobian_weights(input_extents);
    // Now check bias gradients
    net.check_jacobian_bias(input_extents);
    // Now check input error gradients
    net.check_jacobian_input_error(input_extents);

  }

  void test_jacobian_LinearLayer() {
    cout << "test_jacobian_LinearLayer()..." << endl;
    const int minibatch_size = 4;
    const int dim_input = 5;
    const int dim_output = 7;

    const vector<int> input_extents = {dim_input, minibatch_size};
    LinearLayer layer(dim_output, "Linear Layer 1");

    // Check weights gradients.
    layer.check_jacobian_weights(input_extents);
    // Now check bias gradients
    layer.check_jacobian_bias(input_extents);
    // Now check input error gradients
    layer.check_jacobian_input_error(input_extents);

  }

  void test_jacobian_ImageToColumnLayer() {
    cout << "test_jacobian_ImageToColumnLayer()..." << endl;
    const int minibatch_size = 4;
    const int dim1 = 3;
    const int dim2 = 7;
    const int dim3 = 4;

    const vector<int> input_extents = {minibatch_size, dim1, dim2, dim3};
    ImageToColumnLayer layer("Image To Column Layer 1");

    // Check weights gradients.
    layer.check_jacobian_weights(input_extents);
    // Now check bias gradients
    layer.check_jacobian_bias(input_extents);
    // Now check input error gradients
    layer.check_jacobian_input_error(input_extents);

  }

  void test_jacobian_BoxActivationFunction() {
    cout << "test_jacobian_BoxActivationFunction()..." << endl;
    const int minibatch_size = 4;
    const int depth = 3;
    const int height = 4;
    const int width = 5;

    const vector<int> input_extents = {minibatch_size, depth, height, width};
    //BoxActivationFunction::ACTIVATION_TYPE box_activation_type = BoxActivationFunction::ACTIVATION_TYPE::leakyReLU;
    BoxActivationFunction layer(BoxActivationFunction::ACTIVATION_TYPE::leakyReLU, "Box Activation Function 1");

    // Check weights gradients.
    layer.check_jacobian_weights(input_extents);
    // Now check bias gradients
    layer.check_jacobian_bias(input_extents);
    // Now check input error gradients
    layer.check_jacobian_input_error(input_extents);

  }

  void test_jacobian_ColumnActivationFunction() {
    cout << "test_jacobian_ColumnActivationFunction()..." << endl;
    const int minibatch_size = 4;
    const int dim_input = 5;

    const vector<int> input_extents = {dim_input, minibatch_size};
    ColumnActivationFunction layer(ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU, "Column Activation Function 1");

    // Check weights gradients.
    layer.check_jacobian_weights(input_extents);
    // Now check bias gradients
    layer.check_jacobian_bias(input_extents);
    // Now check input error gradients
    layer.check_jacobian_input_error(input_extents);

  }

  void test_jacobian_PoolingLayer() {
    cout << "test_jacobian_PoolingLayer()..." << endl;
    //const int minibatch_size = 4;
    //const int depth = 8;
    //const int height = 10;
    //const int width = 12;

    const int minibatch_size = 4;
    const int depth = 8;
    const int height = 6; // 4 works
    const int width = 4;

    const vector<int> input_extents = {minibatch_size, depth, height, width};
    //const vector<int> pooling_region_extents = {2, 2, 2};
    const vector<int> pooling_region_extents = {1, 3, 3};
    const vector<int> pooling_region_step_sizes = {1, 2, 2};
    PoolingLayer layer(pooling_region_extents, pooling_region_step_sizes, "Pooling Layer 1");

    // Check weights gradients.
    layer.check_jacobian_weights(input_extents);
    // Now check bias gradients
    layer.check_jacobian_bias(input_extents);
    // Now check input error gradients
    layer.check_jacobian_input_error(input_extents);

  }

  void debug_PoolingLayer() {
    cout << "debug_PoolingLayer()" << endl;
    //const int minibatch_size = 1;
    //const int depth = 2;
    //const int height = 4;
    //const int width = 4;

    const int minibatch_size = 2;
    const int depth = 3;
    const int height = 4; // 4 works
    const int width = 4;
    const vector<int> pooling_region_extents = {1, 3, 3};
    const vector<int> pooling_region_step_sizes = {1, 3, 2};

    const vector<int> input_extents = {minibatch_size, depth, height, width};
    //const vector<int> pooling_region_extents = {2, 2, 2};
    //const vector<int> pooling_region_step_sizes = {2, 2, 2};
    PoolingLayer layer(pooling_region_extents, pooling_region_step_sizes, "Pooling Layer 1");


    MatrixF input_activations(input_extents);
    //randomize_uniform(input_activations, -1.0f, 1.0f);
    set_value(input_activations, 0.12f);
    cout << "input_activations: " << endl << input_activations << endl;
    layer.forward(input_activations);

    MatrixF& output_activations = layer.get_output();
    cout << "output_activations: " << endl << output_activations << endl;

    MatrixF& output_deltas = layer.get_output_deltas();
    randomize_uniform(output_deltas, -1.0f, 1.0f);
    cout << "output_deltas: " << endl << output_deltas << endl;
    MatrixF input_error(input_extents);
    layer.back_propagate(input_error, input_activations);
    cout << "input_error: " << endl << input_error << endl;

  }

  void test_MSECostFunction() {
    cout << "test_MSECostFunction()" << endl;
    MSECostFunction cost_func("MSE Cost Function");
    const int unit_count = 8;
    const int minibatch_size = 2;
    const vector<int> input_extents = {unit_count, minibatch_size};
    //MatrixF input_activations(input_extents);
    //randomize_uniform(input_activations, 0.0f, 1.0f);
    //MatrixF target_activations(input_extents);
    //randomize_uniform(target_activations, 0.0f, 1.0f);
    cost_func.check_gradients(input_extents);
  }

  void test_CrossEntropyCostFunction() {
    cout << "test_CrossEntropyCostFunction()" << endl;
    CrossEntropyCostFunction cost_func("Cross Entropy Cost Function");
    const int unit_count = 8;
    const int minibatch_size = 2;
    const vector<int> input_extents = {unit_count, minibatch_size};
    cost_func.check_gradients(input_extents);
  }


  void test_Dropout1D() {
    cout << "test_Dropout1D()" << endl;

    const float prob_keep = 0.3f;
    Dropout1D drop1d(prob_keep, "Dropout1D");
    drop1d.set_train_mode(true);

    int unit_count = 5;
    int minibatch_size = 4;
    MatrixF input_activations(unit_count, minibatch_size);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    cout << "input_activations:" << endl << input_activations << endl;
    MatrixF input_errors(unit_count, minibatch_size);
    randomize_uniform(input_errors, 0.0f, 1.0f);



    cout << "dropout forward:" << endl;
    drop1d.forward(input_activations);
    MatrixF& output_activations = drop1d.get_output();
    cout << "output_activations:" << endl << output_activations << endl;

    MatrixF& output_errors = drop1d.get_output_deltas();
    randomize_uniform(output_errors, 0.0f, 1.0f);
    cout << "Random output_errors:" << endl << output_errors << endl;

    cout << "dropout forward:" << endl;
    drop1d.forward(input_activations);
    cout << "output_activations:" << endl << output_activations << endl;

    cout << "dropout forward:" << endl;
    drop1d.forward(input_activations);
    cout << "output_activations:" << endl << output_activations << endl;

    cout << "Random output_errors:" << endl << output_errors << endl;

    cout << "dropout backward:" << endl;
    drop1d.back_propagate(input_errors, input_activations);
    cout << "input_errors:" << endl << input_errors << endl;


  }

  void test_Dropout3D() {
    cout << "test_Dropout3D()" << endl;

    const float prob_keep = 0.3f;
    Dropout3D dropout(prob_keep, "Dropout3D");
    dropout.set_train_mode(true);

    const int minibatch_size = 2;
    const int depth = 2;
    const int height = 2;
    const int width = 2;
    MatrixF input_activations(minibatch_size, depth, height, width);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    cout << "input_activations:" << endl << input_activations << endl;
    MatrixF input_errors(input_activations.get_extents());
    randomize_uniform(input_errors, 0.0f, 1.0f);

    cout << "dropout forward:" << endl;
    dropout.forward(input_activations);
    MatrixF& output_activations = dropout.get_output();
    cout << "output_activations:" << endl << output_activations << endl;

    MatrixF& output_errors = dropout.get_output_deltas();
    randomize_uniform(output_errors, 0.0f, 1.0f);
    cout << "Random output_errors:" << endl << output_errors << endl;

    cout << "dropout forward:" << endl;
    dropout.forward(input_activations);
    cout << "output_activations:" << endl << output_activations << endl;

    cout << "dropout forward:" << endl;
    dropout.forward(input_activations);
    cout << "output_activations:" << endl << output_activations << endl;

    cout << "Random output_errors:" << endl << output_errors << endl;

    cout << "dropout backward:" << endl;
    dropout.back_propagate(input_errors, input_activations);
    cout << "input_errors:" << endl << input_errors << endl;


  }

  void test_BatchNormalization1D() {
    cout << "test_BatchNormalization1D()" << endl;
    const int minibatch_size = 4;
    const int dim_input = 10;

    const vector<int> input_extents = {dim_input, minibatch_size};
    // If set to false, mean/var checks will work.
    const bool enable_gamma_beta = false;
    const float momentum = 0.1f;
    BatchNormalization1D normalizer(enable_gamma_beta, momentum, "Batch Normalization 1D");
    normalizer.set_train_mode(true);
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    cout << "input_activations:" << endl << input_activations << endl;
    MatrixF input_errors(input_extents);

    //for (int i = 0; i < 200; ++i) {
    normalizer.forward(input_activations);
    //}
    MatrixF& output_activations = normalizer.get_output();
    MatrixF& output_errors = normalizer.get_output_deltas();
    randomize_uniform(output_errors, 0.0f, 1.0f);
    cout << "Random output errors:" << endl << output_errors << endl;
    cout << "output_activations:" << endl << output_activations << endl;

    // Compute mean of output:
    //float mean = sum(output_activations)/output_activations.size();
    MatrixF actual_means(dim_input);
    for (int i = 0; i < dim_input; ++i) {
      for (int j = 0; j < minibatch_size; ++j) {
        actual_means(i) += output_activations(i, j);
      }
      actual_means(i) /= static_cast<float>(minibatch_size);
    }
    cout << "Mean of output activations (should be close to 0): " << endl << actual_means << endl;
    assert_almost_equal(compute_rmse(actual_means), 0, 1e-2f);


    MatrixF actual_std_dev(dim_input);
    for (int i = 0; i < dim_input; ++i) {
      for (int j = 0; j < minibatch_size; ++j) {
        actual_std_dev(i) += (output_activations(i,j) - actual_means(i))*(output_activations(i,j) - actual_means(i));
      }
      actual_std_dev(i) /= static_cast<float>(minibatch_size);
      actual_std_dev(i) = std::sqrt(actual_std_dev(i));
    }

    cout << "Standard deviation of output activations (should be close to 1): " << endl << actual_std_dev << endl;
    MatrixF ones(actual_means.get_extents());
    set_value(ones, 1.0f);
    assert_almost_equal(actual_std_dev, ones, 1e-2f);

    cout << "Back prop:" << endl;
    normalizer.back_propagate(input_errors, input_activations);
    cout << "input_errors:" << endl << input_errors << endl;

    // enable gamma/beta for jacobian checking.
    BatchNormalization1D normalizer2(true, 0.1f, "Batch Normalization 1D");
    normalizer2.set_train_mode(true);

    // Check weights gradients.
    normalizer2.check_jacobian_weights(input_extents); // Will pass if momentum set to 1.0
    // Now check bias gradients
    normalizer2.check_jacobian_bias(input_extents); // Will pass if momentum set to 1.0
    // Now check input error gradients
    normalizer2.check_jacobian_input_error(input_extents);
  }

  void test_BatchNormalization3D() {
    cout << "test_BatchNormalization3D()" << endl;
    const int minibatch_size = 4;
    const int image_depth = 5;
    const int image_height = 3;
    const int image_width = 4;

    const vector<int> input_extents = {minibatch_size, image_depth, image_height, image_width};

    // If set to false, mean/var checks will work.
    const bool enable_gamma_beta = false;
    const float momentum = 0.1f; // 0.1
    BatchNormalization3D normalizer(enable_gamma_beta, momentum, "Batch Normalization 3D");
    normalizer.set_train_mode(true);
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    cout << "input_activations:" << endl << input_activations << endl;
    MatrixF input_errors(input_extents);

    //for (int i = 0; i < 200; ++i) {
    normalizer.forward(input_activations);
    //}
    MatrixF& output_activations = normalizer.get_output();
    MatrixF& output_errors = normalizer.get_output_deltas();
    randomize_uniform(output_errors, 0.0f, 1.0f);
    cout << "Random output errors:" << endl << output_errors << endl;
    cout << "output_activations:" << endl << output_activations << endl;

    // Compute mean of output:

    MatrixF actual_means(image_depth);
    for (int i = 0; i < image_depth; ++i) {
      for (int j = 0; j < minibatch_size; ++j) {
        for (int k = 0; k < image_height; ++k) {
          for (int l = 0; l < image_width; ++l) {
            actual_means(i) += output_activations(j,i,k,l);
          }
        }
      }
      actual_means(i) /= static_cast<float>(minibatch_size*image_height*image_width);
    }
    cout << "Mean of output activations (should be close to 0): " << endl << actual_means << endl;
    assert_almost_equal(compute_rmse(actual_means), 0, 1e-2f);

    MatrixF actual_std_dev(image_depth);
    for (int i = 0; i < image_depth; ++i) {
      for (int j = 0; j < minibatch_size; ++j) {
        for (int k = 0; k < image_height; ++k) {
          for (int l = 0; l < image_width; ++l) {
            actual_std_dev(i) += (output_activations(j,i,k,l) - actual_means(i))*(output_activations(j,i,k,l) - actual_means(i));
          }
        }
      }
      actual_std_dev(i) /= static_cast<float>(minibatch_size*image_height*image_width);
      actual_std_dev(i) = std::sqrt(actual_std_dev(i));
    }

    cout << "Standard deviation of output activations (should be close to 1): " << endl << actual_std_dev << endl;
    MatrixF ones(actual_means.get_extents());
    set_value(ones, 1.0f);
    assert_almost_equal(actual_std_dev, ones, 1e-2f);

    cout << "Back prop:" << endl;
    normalizer.back_propagate(input_errors, input_activations);
    cout << "input_errors:" << endl << input_errors << endl;

    // Set momentum to 1.0 and enable gamma/beta for jacobian checking.
    BatchNormalization3D normalizer2(true, 0.1f, "Batch Normalization 3D");
    normalizer2.set_train_mode(true);

    // Check weights gradients.
    normalizer2.check_jacobian_weights(input_extents); // Will pass if momentum set to 1.0
    // Now check bias gradients
    normalizer2.check_jacobian_bias(input_extents); // Will pass if momentum set to 1.0
    // Now check input error gradients
    normalizer2.check_jacobian_input_error(input_extents);
  }


  void run_all_tests() {
    test_mat_mult();
    test_mat_multiply_left_transpose();
    test_mat_multiply_right_transpose();
    test_MatrixF1D();
    test_MatrixF2D();
    test_MatrixF3D();
    test_MatrixF4D();
    test_MatrixF5D();
    test_MatrixF6D();
    test_MatrixResize();
    test_compute_kmax();
    test_compute_kmax_v2();
    test_relu();
    test_PoolingLayer();
    test_optimized_convolutive_deltas();
    test_optimized_weight_grad_convolutive();

    test_optimized_convolve_3d_minibatch();
    test_optimized_3d_convolutive_deltas();
    test_optimized_3d_weight_grad_convolutive();

    test_compute_3d_kmax();


    test_Dropout1D();
    test_Dropout3D();
    test_jacobian_LinearLayer();
    test_jacobian_ConvLayer3D();
    test_select();
    test_jacobian_ConvLayer3D();
    test_SequentialNetwork();
    test_SequentialNetwork2();
    test_jacobian_ImageToColumnLayer();
    test_jacobian_BoxActivationFunction();
    test_jacobian_ColumnActivationFunction();
    test_MSECostFunction();
    test_CrossEntropyCostFunction();
    test_BatchNormalization1D();
  }



}
