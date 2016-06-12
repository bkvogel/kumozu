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
#include "SequentialLayer.h"
#include "AdderNode.h"
//#include "MeanNode.h"
#include "SubtractorNode.h"
#include "MultiplyerNode.h"
#include "SplitterNode.h"
#include "CharRNNMinibatchGetter.h"
#include "ConcatNode.h"
#include "ExtractorNode.h"

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

  void test_mat_multiply_left_transpose_accumulate() {
    std::cout << "test_mat_multiply_left_transpose_accumulate()..." << std::endl;
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

    mat_multiply_left_transpose_accumulate(A, B, C);
    mat_multiply_left_transpose_accumulate(A, B, C);
    std::cout << "A = " << std::endl << A << std::endl;
    //cout << "B = " << endl << B << endl;

    mat_multiply_left_transpose_naive_accumulate(Ac, Bc, Cc);
    mat_multiply_left_transpose_naive_accumulate(Ac, Bc, Cc);
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

void test_mat_multiply_right_transpose_accumulate() {
    std::cout << "test_mat_multiply_right_transpose_accumulate()..." << std::endl;
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

    mat_multiply_right_transpose_accumulate(A, B, C);
    mat_multiply_right_transpose_accumulate(A, B, C);
    std::cout << "A = " << std::endl << A << std::endl;
    //cout << "B = " << endl << B << endl;

    mat_multiply_right_transpose_accumulate_naive(Ac, Bc, Cc);
    mat_multiply_right_transpose_accumulate_naive(Ac, Bc, Cc);
    std::cout << "Ac = " << std::endl << Ac << std::endl;
    //cout << "Bc = " << endl << Bc << endl;
    assert_almost_equal(A, Ac);
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

  void test_copy_to_from_submatrix() {
    cout << "test_copy_to_from_submatrix()" << endl;
    MatrixF fullmat(7,5);
    MatrixF submat1(3,4);
    randomize_uniform(submat1, 0.0f, 1.0f);
    cout << "submat1 now:" << endl << submat1 << endl;
    MatrixF submat2(4,4);
    randomize_uniform(submat2, 0.0f, 1.0f);
    cout << "submat2 now:" << endl << submat2 << endl;
    copy_from_submatrix(submat1, fullmat, 0,1);
    cout << "fullmat now: " << endl << fullmat << endl;
    copy_from_submatrix(submat2, fullmat, 3, 0);
    cout << "fullmat now: " << endl << fullmat << endl;
    set_value(submat1, 0.0f);
    cout << "submat1 now:" << endl << submat1 << endl;
    copy_to_submatrix(submat1, fullmat, 0,1);
    cout << "submat1 now:" << endl << submat1 << endl;
    set_value(submat2, 0.0f);
    cout << "submat2 now:" << endl << submat2 << endl;
    copy_to_submatrix(submat2, fullmat, 3, 0);
    cout << "submat2 now:" << endl << submat2 << endl;
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
      set_value(grad_W_true, 0.0f);
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
      set_value(grad_W_optimized, 0.0f);
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
    layer.check_jacobian_input_backward(input_extents);

  }



  void test_SequentialLayer() {
    cout << "test_SequentialLayer()..." << endl;
    const int minibatch_size = 4;
    const int dim_input = 5;
    const int dim_output = 7;

    SequentialLayer seq_net("sequential layer 1");
    const vector<int> input_extents = {dim_input, minibatch_size};
    //MatrixF input_activations(input_extents);
    //randomize_uniform(input_activations, 0.0f, 1.0f);
    //MatrixF input_backwards(input_extents);
    //seq_net.create_input_port(input_activations, input_backwards);
    LinearLayer lin_layer(dim_output, "Linear Layer 1");
    seq_net.add_layer(lin_layer);
    
    // Check weights gradients.
    seq_net.check_jacobian_weights(input_extents);
    // Now check bias gradients
    seq_net.check_jacobian_bias(input_extents);
    // Now check input error gradients
    seq_net.check_jacobian_input_backward(input_extents);
  }



  void test_SequentialLayer2() {
    cout << "test_SequentialLayer2()..." << endl;
    const int minibatch_size = 4;
    const int image_depth = 3;
    const int image_height = 13;
    const int image_width = 15;
    const int filter_count = 5;
    const int filter_height = 3;
    const int filter_width = 4;

    const int dim_output = 7;

    SequentialLayer net("sequential network 1");
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
    net.check_jacobian_input_backward(input_extents);

  }

void test_SequentialLayer3() {
    cout << "test_SequentialLayer3()..." << endl;
    // Test a network of sequentially-connected layers that also includes the 
    // MSE cost function.
    const int minibatch_size = 4;
    const int dim_input = 5;
    const int dim_output = 7;

    MatrixF target_activations(dim_output, minibatch_size);
    randomize_uniform(target_activations, -1.0f, 1.0f);

    SequentialLayer seq_net("sequential layer 1");
    const vector<int> input_extents = {dim_input, minibatch_size};
    LinearLayer lin_layer(dim_output, "Linear Layer 1");
    seq_net.add_layer(lin_layer);
    MSECostFunction cost_func("MSE Cost Function");
    seq_net.add_layer(cost_func);

    cost_func.set_target_activations(target_activations);

    // Check weights gradients.
    seq_net.check_jacobian_weights(input_extents);
    // Now check bias gradients
    seq_net.check_jacobian_bias(input_extents);
    // Now check input error gradients
    seq_net.check_jacobian_input_backward(input_extents);
  }

void test_SequentialLayer4() {
    cout << "test_SequentialLayer4()..." << endl;
    // Test a network of sequentially-connected layers that also includes the 
    // MSE cost function.
    const int minibatch_size = 4;
    const int dim_input = 5;
    const int dim_output = 7;

    MatrixI target_activations(minibatch_size);

    SequentialLayer seq_net("sequential layer 1");
    const vector<int> input_extents = {dim_input, minibatch_size};
    LinearLayer lin_layer(dim_output, "Linear Layer 1");
    seq_net.add_layer(lin_layer);
    CrossEntropyCostFunction cost_func("Cross Entropy Cost Function");
    seq_net.add_layer(cost_func);

    cost_func.set_target_activations(target_activations);

    // Check weights gradients.
    seq_net.check_jacobian_weights(input_extents);
    // Now check bias gradients
    seq_net.check_jacobian_bias(input_extents);
    // Now check input error gradients
    seq_net.check_jacobian_input_backward(input_extents);
  }


    void test_SequentialLayer_shared_parameters() {
    cout << "test_SequentialLayer_shared_parameters()" << endl;
    const int minibatch_size = 4;
    const int dim_input = 5;
    const int dim_output = 5;

    SequentialLayer seq_net("sequential layer 1");
    const vector<int> input_extents = {dim_input, minibatch_size};
    //MatrixF input_activations(input_extents);
    //randomize_uniform(input_activations, 0.0f, 1.0f);
    //MatrixF input_backwards(input_extents);
    //seq_net.create_input_port(input_activations, input_backwards);
    LinearLayer lin_layer(dim_output, "Linear Layer 1");
    seq_net.add_layer(lin_layer);

    LinearLayer lin_layer2(dim_output, "Linear Layer 2");
    lin_layer2.set_shared(lin_layer);
    seq_net.add_layer(lin_layer2);

    // Check weights gradients.
    seq_net.check_jacobian_weights(input_extents);
    // Now check bias gradients
    seq_net.check_jacobian_bias(input_extents);
    // Now check input error gradients
    seq_net.check_jacobian_input_backward(input_extents);
    //
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
    layer.check_jacobian_input_backward(input_extents);

  }

  void test_jacobian_LinearLayer_Node() {
    cout << "test_jacobian_LinearLayer_Node()..." << endl;
    const int minibatch_size = 4;
    const int dim_input = 5;
    const int dim_output = 7;

    const vector<int> input_extents = {dim_input, minibatch_size};
    MatrixF input_activations(input_extents);
    LinearLayer layer(dim_output, "Linear Layer 1");

    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["0"] = input_extents;

    // Check weights gradients.
    //layer.check_jacobian_weights(input_extents);
    layer.check_jacobian_weights(input_port_extents_map);
    // Now check bias gradients
    layer.check_jacobian_bias(input_port_extents_map);
    // Now check input error gradients
    layer.check_jacobian_input_backward(input_port_extents_map);

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
    layer.check_jacobian_input_backward(input_extents);

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
    layer.check_jacobian_input_backward(input_extents);

  }

  void test_jacobian_ColumnActivationFunction() {
    cout << "test_jacobian_ColumnActivationFunction()..." << endl;
    const int minibatch_size = 4;
    const int dim_input = 5;

    const vector<int> input_extents = {dim_input, minibatch_size};
    ColumnActivationFunction layer(ColumnActivationFunction::ACTIVATION_TYPE::identity, "Column Activation Function 1");

    // Now check input error gradients
    layer.check_jacobian_input_backward(input_extents);

    layer.set_activation_type(ColumnActivationFunction::ACTIVATION_TYPE::ReLU);

    // Now check input error gradients
    layer.check_jacobian_input_backward(input_extents);

    layer.set_activation_type(ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU);

    // Now check input error gradients
    layer.check_jacobian_input_backward(input_extents);

    layer.set_activation_type(ColumnActivationFunction::ACTIVATION_TYPE::tanh);

    // Now check input error gradients
    layer.check_jacobian_input_backward(input_extents);

    layer.set_activation_type(ColumnActivationFunction::ACTIVATION_TYPE::sigmoid);

    // Now check input error gradients
    layer.check_jacobian_input_backward(input_extents);

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
    layer.check_jacobian_input_backward(input_extents);

  }

  void test_MSECostFunction() {
    cout << "test_MSECostFunction()" << endl;
    MSECostFunction cost_func("MSE Cost Function");
    const int unit_count = 8;
    const int minibatch_size = 2;
    const vector<int> input_extents = {unit_count, minibatch_size};
    MatrixF target_activations(input_extents);
    randomize_uniform(target_activations, -1.0f, 1.0f);
    cost_func.set_target_activations(target_activations);
    cost_func.check_jacobian_input_backward(input_extents);
  }

  void test_CrossEntropyCostFunction() {
    cout << "test_CrossEntropyCostFunction()" << endl;
    CrossEntropyCostFunction cost_func("Cross Entropy Cost Function");
    const int unit_count = 8;
    const int minibatch_size = 2;
    const vector<int> input_extents = {unit_count, minibatch_size};
    MatrixI target_activations(minibatch_size);
    
    cost_func.set_target_activations(target_activations);
    cost_func.check_jacobian_input_backward(input_extents);
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
    MatrixF input_backwards(unit_count, minibatch_size);
    randomize_uniform(input_backwards, 0.0f, 1.0f);
    drop1d.create_input_port(input_activations, input_backwards);


    cout << "dropout forward:" << endl;
    drop1d.forward();
    const MatrixF& output_forward = drop1d.get_output_forward();
    cout << "output_forward:" << endl << output_forward << endl;

    MatrixF& output_backwards = drop1d.get_output_backward();
    randomize_uniform(output_backwards, 0.0f, 1.0f);
    cout << "Random output_backwards:" << endl << output_backwards << endl;

    cout << "dropout forward:" << endl;
    drop1d.forward();
    cout << "output_forward:" << endl << output_forward << endl;

    cout << "dropout forward:" << endl;
    drop1d.forward();
    cout << "output_forward:" << endl << output_forward << endl;

    cout << "Random output_backwards:" << endl << output_backwards << endl;

    cout << "dropout backward:" << endl;
    drop1d.back_propagate();
    cout << "input_backwards:" << endl << input_backwards << endl;


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
    MatrixF input_backwards(input_activations.get_extents());
    randomize_uniform(input_backwards, 0.0f, 1.0f);
    dropout.create_input_port(input_activations, input_backwards);
    
    cout << "dropout forward:" << endl;
    dropout.forward();
    const MatrixF& output_forward = dropout.get_output_forward();
    cout << "output_forward:" << endl << output_forward << endl;

    MatrixF& output_backwards = dropout.get_output_backward();
    randomize_uniform(output_backwards, 0.0f, 1.0f);
    cout << "Random output_backwards:" << endl << output_backwards << endl;

    cout << "dropout forward:" << endl;
    dropout.forward();
    cout << "output_forward:" << endl << output_forward << endl;

    cout << "dropout forward:" << endl;
    dropout.forward();
    cout << "output_forward:" << endl << output_forward << endl;

    cout << "Random output_backwards:" << endl << output_backwards << endl;

    cout << "dropout backward:" << endl;
    dropout.back_propagate();
    cout << "input_backwards:" << endl << input_backwards << endl;


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
    MatrixF input_backwards(input_extents);
    normalizer.create_input_port(input_activations, input_backwards);

    //for (int i = 0; i < 200; ++i) {
    normalizer.forward();
    //}
    const MatrixF& output_forward = normalizer.get_output_forward();
    MatrixF& output_backwards = normalizer.get_output_backward();
    randomize_uniform(output_backwards, 0.0f, 1.0f);
    cout << "Random output errors:" << endl << output_backwards << endl;
    cout << "output_forward:" << endl << output_forward << endl;

    // Compute mean of output:
    //float mean = sum(output_forward)/output_forward.size();
    MatrixF actual_means(dim_input);
    for (int i = 0; i < dim_input; ++i) {
      for (int j = 0; j < minibatch_size; ++j) {
        actual_means(i) += output_forward(i, j);
      }
      actual_means(i) /= static_cast<float>(minibatch_size);
    }
    cout << "Mean of output activations (should be close to 0): " << endl << actual_means << endl;
    assert_almost_equal(compute_rmse(actual_means), 0, 1e-2f);


    MatrixF actual_std_dev(dim_input);
    for (int i = 0; i < dim_input; ++i) {
      for (int j = 0; j < minibatch_size; ++j) {
        actual_std_dev(i) += (output_forward(i,j) - actual_means(i))*(output_forward(i,j) - actual_means(i));
      }
      actual_std_dev(i) /= static_cast<float>(minibatch_size);
      actual_std_dev(i) = std::sqrt(actual_std_dev(i));
    }

    cout << "Standard deviation of output activations (should be close to 1): " << endl << actual_std_dev << endl;
    MatrixF ones(actual_means.get_extents());
    set_value(ones, 1.0f);
    assert_almost_equal(actual_std_dev, ones, 1e-2f);

    cout << "Back prop:" << endl;
    normalizer.back_propagate();
    cout << "input_backwards:" << endl << input_backwards << endl;

    // enable gamma/beta for jacobian checking.
    BatchNormalization1D normalizer2(true, 0.1f, "Batch Normalization 1D");
    normalizer2.set_train_mode(true);

    // Check weights gradients.
    normalizer2.check_jacobian_weights(input_extents); // Will pass if momentum set to 1.0
    // Now check bias gradients
    normalizer2.check_jacobian_bias(input_extents); // Will pass if momentum set to 1.0
    // Now check input error gradients
    normalizer2.check_jacobian_input_backward(input_extents);
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
    MatrixF input_backwards(input_extents);
    normalizer.create_input_port(input_activations, input_backwards);

    //for (int i = 0; i < 200; ++i) {
    normalizer.forward();
    //}
    const MatrixF& output_forward = normalizer.get_output_forward();
    MatrixF& output_backwards = normalizer.get_output_backward();
    randomize_uniform(output_backwards, 0.0f, 1.0f);
    cout << "Random output errors:" << endl << output_backwards << endl;
    cout << "output_forward:" << endl << output_forward << endl;

    // Compute mean of output:

    MatrixF actual_means(image_depth);
    for (int i = 0; i < image_depth; ++i) {
      for (int j = 0; j < minibatch_size; ++j) {
        for (int k = 0; k < image_height; ++k) {
          for (int l = 0; l < image_width; ++l) {
            actual_means(i) += output_forward(j,i,k,l);
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
            actual_std_dev(i) += (output_forward(j,i,k,l) - actual_means(i))*(output_forward(j,i,k,l) - actual_means(i));
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
    normalizer.back_propagate();
    cout << "input_backwards:" << endl << input_backwards << endl;

    // Set momentum to 1.0 and enable gamma/beta for jacobian checking.
    BatchNormalization3D normalizer2(true, 0.1f, "Batch Normalization 3D");
    normalizer2.set_train_mode(true);

    // Check weights gradients.
    normalizer2.check_jacobian_weights(input_extents); // Will pass if momentum set to 1.0
    // Now check bias gradients
    normalizer2.check_jacobian_bias(input_extents); // Will pass if momentum set to 1.0
    // Now check input error gradients
    normalizer2.check_jacobian_input_backward(input_extents);
  }


  void test_Node_shared_parameters() {
    cout << "test_Node_shared_parameters()" << endl;

    const int minibatch_size = 4;
    const int dim_input = 5;
    const int dim_output = 7;

    const vector<int> input_extents = {dim_input, minibatch_size};
    MatrixF input_activations(input_extents);
    MatrixF input_backwards(input_extents);
    LinearLayer layer1(dim_output, "Linear Layer 1");
    layer1.create_input_port(input_activations, input_backwards);

    LinearLayer layer2(dim_output, "Linear Layer 2");
    layer2.create_input_port(input_activations, input_backwards);
    // Make layer2 share layer1's parameters:
    layer2.set_shared(layer1);

    layer1.forward(); // Initialize
    auto& layer1_forward_out = layer1.get_output_forward();
    layer2.forward(); // Initialize
    auto& layer2_forward_out = layer2.get_output_forward();
    assert_almost_equal(layer1_forward_out, layer2_forward_out);

    cout << "PASSED" << endl;
  }

  void test_Node_shared_parameters2() {
    cout << "test_Node_shared_parameters2()..." << endl;
    

    // Reason for making a custom class:
    // If you want to make multiple copies of the same network, such as 1 for training and
    // 1 validation etc, it is best to wrap the network in a class for convinience. This
    // allows us to avoid dealing with pointers for the nodes that are added using "add_layer()".
    class TestNet : public Node {

    public:

      TestNet(int filter_count, int filter_height, int filter_width, int dim_output, std::string name) :
	Node(name),
	m_net {"sequential network: " + name},
	m_conv_layer1 {filter_count, filter_height, filter_width, "Conv Layer: " + name},
	m_box_activation_layer1 {BoxActivationFunction::ACTIVATION_TYPE::leakyReLU, "Box Activation Function: " + name},
	m_pooling_layer1 {{1, 3, 3}, {1, 2, 2}, "Pooling Layer: " + name},
	m_layer2 {"Image To Column Layer: " + name},
	m_linear_laye1 {dim_output, "Linear Layer:" + name},
	m_column_activation_layer1 {ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU, "Column Activation Function: " + name}
      {
	connect_input_to_contained_node(m_net);
	add_node(m_net);
	m_net.add_layer(m_conv_layer1);
	m_net.add_layer(m_box_activation_layer1);
	m_net.add_layer(m_pooling_layer1);
	m_net.add_layer(m_layer2);
	m_net.add_layer(m_linear_laye1);
	m_net.add_layer(m_column_activation_layer1);
	create_output_port(m_net);
      }
	
    private:
      SequentialLayer m_net;
      ConvLayer3D m_conv_layer1;
      BoxActivationFunction m_box_activation_layer1;
      PoolingLayer m_pooling_layer1;
      ImageToColumnLayer m_layer2;
      LinearLayer m_linear_laye1;
      ColumnActivationFunction m_column_activation_layer1;

    };

    const int minibatch_size = 4;
    const int image_depth = 3;
    const int image_height = 13;
    const int image_width = 15;
    const int filter_count = 5;
    const int filter_height = 3;
    const int filter_width = 4;

    const int dim_output = 7;
   

    TestNet net1(filter_count, filter_height, filter_width, dim_output, "1");
    TestNet net2(filter_count, filter_height, filter_width, dim_output, "2");

    const vector<int> input_extents = {minibatch_size, image_depth, image_height, image_width};
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    MatrixF input_backwards(input_extents);

    net1.create_input_port(input_activations, input_backwards);
    net2.create_input_port(input_activations, input_backwards);

    // Make net2 share net1's parameters:
    net2.set_shared(net1);

    net1.forward(); // Initialize
    auto& net1_out = net1.get_output_forward();
    net2.forward(); // Initialize
    auto& net2_out = net2.get_output_forward();
    assert_almost_equal(net1_out, net2_out);

    auto& net1_out_backward = net1.get_output_backward();
    randomize_uniform(net1_out_backward, 0.0f, 1.0f);
    set_value(input_backwards, 0.0f);
    net1.back_propagate();
    MatrixF net1_input_backward = input_backwards;

    set_value(input_backwards, 0.0f);
    auto& net2_out_backward = net2.get_output_backward();
    net2_out_backward = net1_out_backward;
    net2.back_propagate();
    assert_almost_equal(net1_input_backward, input_backwards);

    cout << "PASSED" << endl;
  }


  void test_multi_port_node() {
    cout << "test_multi_port_node()" << endl;
    // Test a simple Node with two input ports and two output ports.

    Node composite_node("Composite Node");

    // First internal node.
    const int dim_output1 = 5;
    LinearLayer linear_laye1(dim_output1, "Linear Layer 1");
    composite_node.connect_input_to_contained_node("in1", linear_laye1);
    composite_node.create_output_port_this_name(linear_laye1, "out1");
    
    composite_node.add_node(linear_laye1);

    // Second internal node.
    const int dim_output2 = 7;
    LinearLayer linear_laye2(dim_output2, "Linear Layer 2");
    composite_node.connect_input_to_contained_node("in2", linear_laye2);
    composite_node.create_output_port_this_name(linear_laye2, "out2");

    composite_node.add_node(linear_laye2);

    // Add input extents for each input port of the composite node:
    const int minibatch_size = 4;
    const int dim_in1 = 3;
    const int dim_in2 = 6;
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["in1"] = {dim_in1, minibatch_size};
    input_port_extents_map["in2"] = {dim_in2, minibatch_size};

    // Check weights gradients.
    composite_node.check_jacobian_weights(input_port_extents_map);
    // Now check bias gradients
    composite_node.check_jacobian_bias(input_port_extents_map);
    // Now check input error gradients
    composite_node.check_jacobian_input_backward(input_port_extents_map);
  }

  void test_AdderNode() {
    cout << "test_AdderNode()" << endl;
    const int dim1 = 3;
    const int dim2 = 2;
    const vector<int> input_extents = {dim1, dim2};
    AdderNode adder("Adder");
    MatrixF input_forward1(input_extents);
    set_value(input_forward1, 1.0f);
    cout << "Input 1: " << endl << input_forward1 << endl;
    MatrixF input_backward1(input_extents);
    adder.create_input_port(input_forward1, input_backward1, "in1");

    MatrixF input_forward2(input_extents);
    set_value(input_forward2, 2.0f);
    cout << "Input 2: " << endl << input_forward2 << endl;
    MatrixF input_backward2(input_extents);
    adder.create_input_port(input_forward2, input_backward2, "in2");

    //adder.reinitialize();
    adder.forward();

    cout << "Output: " << endl << adder.get_output_forward() << endl;
    
    // Check gradients:
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["in1"] = input_extents;
    input_port_extents_map["in2"] = input_extents;

    // Check weights gradients.
    adder.check_jacobian_weights(input_port_extents_map);
    // Now check bias gradients
    adder.check_jacobian_bias(input_port_extents_map);
    // Now check input error gradients
    adder.check_jacobian_input_backward(input_port_extents_map);
  }

  /*
  void test_MeanNode() {
    // Test MeanNode in usual model.
    cout << "test_MeanNode()" << endl;
    const int dim1 = 3;
    const int dim2 = 2;
    const vector<int> input_extents = {dim1, dim2};
    MeanNode meaner("MeanNode");
    MatrixF input_forward1(input_extents);
    set_value(input_forward1, 1.0f);
    cout << "Input 1: " << endl << input_forward1 << endl;
    MatrixF input_backward1(input_extents);
    meaner.create_input_port(input_forward1, input_backward1, "in1");

    MatrixF input_forward2(input_extents);
    set_value(input_forward2, 2.0f);
    cout << "Input 2: " << endl << input_forward2 << endl;
    MatrixF input_backward2(input_extents);
    meaner.create_input_port(input_forward2, input_backward2, "in2");

    meaner.forward();

    cout << "Output: " << endl << meaner.get_output_forward() << endl;
    
    // Check gradients:
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["in1"] = input_extents;
    input_port_extents_map["in2"] = input_extents;

    // Check weights gradients.
    meaner.check_jacobian_weights(input_port_extents_map);
    // Now check bias gradients
    meaner.check_jacobian_bias(input_port_extents_map);
    // Now check input error gradients
    meaner.check_jacobian_input_backward(input_port_extents_map);
  }
  */

  /*
  void test_MeanNode2() {
    // Test MeanNode in PFN mode
    cout << "test_MeanNode2()" << endl;
    const int dim1 = 3;
    const int dim2 = 2;
    const vector<int> input_extents = {dim1, dim2};
    MeanNode meaner("MeanNode", true);
    MatrixF input_forward1(input_extents);
    set_value(input_forward1, 1.0f);
    cout << "Input 1: " << endl << input_forward1 << endl;
    MatrixF input_backward1(input_extents);
    meaner.create_input_port(input_forward1, input_backward1, "in1");

    MatrixF input_forward2(input_extents);
    set_value(input_forward2, 2.0f);
    cout << "Input 2: " << endl << input_forward2 << endl;
    MatrixF input_backward2(input_extents);
    meaner.create_input_port(input_forward2, input_backward2, "in2");

    meaner.forward();
    cout << "Output: " << endl << meaner.get_output_forward() << endl;

    copy_matrix(meaner.get_output_backward(), meaner.get_output_forward());
    
    meaner.back_propagate();
    cout << "Input 1 backward: " << endl << input_backward1;
    cout << "Input 2 backward: " << endl << input_backward2;
    cout << "PASSED" << endl;
    // Note: can't check gradients in PFN mode.
  }
  */
  
  void test_ConcatNode() {
    cout << "test_ConcatNode()" << endl;
    const int dim1a = 3;
    const int dim1b = 4;
    const int minibatch_size = 2;
    ConcatNode concat("Concat");
    MatrixF input_forward1(dim1a, minibatch_size);
    set_value(input_forward1, 1.0f);
    cout << "Input 1: " << endl << input_forward1 << endl;
    MatrixF input_backward1(input_forward1.get_extents());
    concat.create_input_port(input_forward1, input_backward1, "in1");

    MatrixF input_forward2(dim1b, minibatch_size);
    set_value(input_forward2, 2.0f);
    cout << "Input 2: " << endl << input_forward2 << endl;
    MatrixF input_backward2(input_forward2.get_extents());
    concat.create_input_port(input_forward2, input_backward2, "in2");

    concat.forward();

    cout << "Output: " << endl << concat.get_output_forward() << endl;
    
    // Check gradients:
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["in1"] = input_forward1.get_extents();
    input_port_extents_map["in2"] = input_forward2.get_extents();

    // Check weights gradients.
    concat.check_jacobian_weights(input_port_extents_map);
    // Now check bias gradients
    concat.check_jacobian_bias(input_port_extents_map);
    // Now check input error gradients
    concat.check_jacobian_input_backward(input_port_extents_map);
  }


  void test_SubtractorNode() {
    cout << "test_SubtractorNode()" << endl;
    const int dim1 = 3;
    const int dim2 = 2;
    const vector<int> input_extents = {dim1, dim2};
    SubtractorNode suber("Suber");
    MatrixF input_forward1(input_extents);
    set_value(input_forward1, 1.0f);
    cout << "Input 1: " << endl << input_forward1 << endl;
    MatrixF input_backward1(input_extents);
    suber.create_input_port(input_forward1, input_backward1, "plus");

    MatrixF input_forward2(input_extents);
    set_value(input_forward2, 2.0f);
    cout << "Input 2: " << endl << input_forward2 << endl;
    MatrixF input_backward2(input_extents);
    suber.create_input_port(input_forward2, input_backward2, "minus");

    suber.forward();

    cout << "Output: " << endl << suber.get_output_forward() << endl;
    
    // Check gradients:
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["plus"] = input_extents;
    input_port_extents_map["minus"] = input_extents;

    // Check weights gradients.
    suber.check_jacobian_weights(input_port_extents_map);
    // Now check bias gradients
    suber.check_jacobian_bias(input_port_extents_map);
    // Now check input error gradients
    suber.check_jacobian_input_backward(input_port_extents_map);
  }

  void test_MultiplyerNode() {
    cout << "test_MultiplyerNode()" << endl;
    const int dim1 = 3;
    const int dim2 = 2;
    const vector<int> input_extents = {dim1, dim2};
    MultiplyerNode multiplyer("Multiplyer");
    MatrixF input_forward1(input_extents);
    set_value(input_forward1, 2.0f);
    cout << "Input 1: " << endl << input_forward1 << endl;
    MatrixF input_backward1(input_extents);
    multiplyer.create_input_port(input_forward1, input_backward1, "in1");

    MatrixF input_forward2(input_extents);
    set_value(input_forward2, 3.0f);
    cout << "Input 2: " << endl << input_forward2 << endl;
    MatrixF input_backward2(input_extents);
    multiplyer.create_input_port(input_forward2, input_backward2, "in2");

    MatrixF input_forward3(input_extents);
    set_value(input_forward3, 4.0f);
    cout << "Input 3: " << endl << input_forward3 << endl;
    MatrixF input_backward3(input_extents);
    multiplyer.create_input_port(input_forward3, input_backward2, "in3");

    multiplyer.forward();

    cout << "Output: " << endl << multiplyer.get_output_forward() << endl;
    
    // Check gradients:
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["in1"] = input_extents;
    input_port_extents_map["in2"] = input_extents;
    input_port_extents_map["in3"] = input_extents;

    // Check weights gradients.
    multiplyer.check_jacobian_weights(input_port_extents_map);
    // Now check bias gradients
    multiplyer.check_jacobian_bias(input_port_extents_map);
    // Now check input error gradients
    multiplyer.check_jacobian_input_backward(input_port_extents_map);
  }

  void test_SplitterNode() {
    cout << "test_SplitterNode()" << endl;
    const int dim1 = 3;
    const int dim2 = 2;
    const vector<int> input_extents = {dim1, dim2};
    const int output_port_count = 3;
    SplitterNode splitter(output_port_count, "Splitter");
    MatrixF input_forward(input_extents);
    randomize_uniform(input_forward, 0.0f, 1.0f);
    cout << "Input: " << endl << input_forward << endl;
    MatrixF input_backward(input_extents);
    splitter.create_input_port(input_forward, input_backward);

    splitter.forward();
    
    for (int i = 0; i < output_port_count; ++i) {
      cout << "Output port: " << i << " " << endl << splitter.get_output_forward(to_string(i)) << endl;
    }
    // Check gradients:
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map[DEFAULT_INPUT_PORT_NAME] = input_extents;

    // Now check input error gradients
    splitter.check_jacobian_input_backward(input_port_extents_map);
  }

  void test_SplitterNode2() {
    cout << "test_SplitterNode2()" << endl;
    // Test PFN mode (computes mean of outputs in backward pass).
    const int dim1 = 3;
    const int dim2 = 2;
    const vector<int> input_extents = {dim1, dim2};
    const int output_port_count = 3;
    SplitterNode splitter(output_port_count, "Splitter", true);
    MatrixF input_forward(input_extents);
    set_value(input_forward, 3.0f);
    cout << "Input forward: " << endl << input_forward << endl;
    MatrixF input_backward(input_extents);
    splitter.create_input_port(input_forward, input_backward);

    splitter.forward();
    
    for (int i = 0; i < output_port_count; ++i) {
      cout << "Output forward: " << i << " " << endl << splitter.get_output_forward(to_string(i)) << endl;
      set_value(splitter.get_output_backward(to_string(i)), static_cast<float>(i*i));
      cout << "Output backward (set manually): " << i << " " << endl << splitter.get_output_backward(to_string(i)) << endl;
    }
    splitter.back_propagate();
    cout << "Input backward: " << endl << input_backward << endl;

    cout << "PASSED" << endl;
  }

  
  void test_ExtractorNode() {
    cout << "test_ExtractorNode()" << endl;
    const int minibatch_size = 2;
    const vector<int> input_extents {9, minibatch_size};
    MatrixF input_forward(input_extents);
    const int output_port_count = 3;
    vector<int> partition_sizes {2, 4, 3};
    ExtractorNode extractor(partition_sizes, "Splitter");
    randomize_uniform(input_forward, -1.0f, 1.0f);
    cout << "Input: " << endl << input_forward << endl;
    MatrixF input_backward(input_extents);
    extractor.create_input_port(input_forward, input_backward);

    extractor.forward();
    
    for (int i = 0; i < output_port_count; ++i) {
      cout << "Output port: " << i << " " << endl << extractor.get_output_forward(to_string(i)) << endl;
    }
    // Check gradients:
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map[DEFAULT_INPUT_PORT_NAME] = input_extents;

    // Now check input error gradients
    extractor.check_jacobian_input_backward(input_port_extents_map);
  }


  void test_rnn_slice() {
    cout << "test_rnn_slice()" << endl;

    // Create 1 slice of a vanilla RNN as a local class and check its jacobians.

    // Create a local class that represents 1 time slice in the vanilla RNN:
    // 
    // Input ports:
    //               "x_t"
    //               "h_t_prev"
    //
    // Output ports:
    //               "h_t"
    //
    // Note: Since this will be a composite node (that is, a node that contains a subgraph of
    // other nodes), all we need to do is the following:
    //
    // 1. For each contained node, add a corresponding private member variable (see below).
    // 2. In the constructor initialization list, call the constructor of each contained node (most of
    // them will only need a name, but some will require configuration parameters.
    // 3. In the constructor body, do the following in order: 
    //     i. Connect each input port of this node to desired internal node input port.
    //     ii. Connect internal nodes together however you want, adding each of them via add_node() in
    //         the order that they should be called in the forward data propagation.
    //     iii. Connect outputs of internal nodes to output ports of this node.
    class RNNNode : public Node {

    public:

      RNNNode(int dim, std::string name) :
	Node(name), 
	m_linear_x_t {dim, "Linear x_t"},
	m_linear_h_t_prev {dim, "Linear h_t_prev"},
	m_adder {"Adder"},
	m_tanh {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh activation"}
      {
	// This node is a "composite node" since it will contain a network subgraph.
	// We now spcecify the subgraph:

	// Note: add_node() must add the contained nodes in the same order that they should
	// be called in the forward data pass.

	// Connect inputs of this node to internal nodes:
	connect_input_to_contained_node("x_t", m_linear_x_t);
	add_node(m_linear_x_t);
	connect_input_to_contained_node("h_t_prev", m_linear_h_t_prev);
	add_node(m_linear_h_t_prev);
	// Sum the outputs of the two linear nodes:
	m_adder.create_input_port_this_name(m_linear_x_t, "in1");
	m_adder.create_input_port_this_name(m_linear_h_t_prev, "in2");
	add_node(m_adder);

	// Connect the output of the adder to the input of a tanh activation:
	m_tanh.connect_parent(m_adder);
	add_node(m_tanh);

	// create output ports:
	create_output_port_this_name(m_tanh, "h_t");
      }

    private:

      LinearLayer m_linear_x_t;
      LinearLayer m_linear_h_t_prev;
      AdderNode m_adder;
      ColumnActivationFunction m_tanh;
    };

    // Internal size of RNN:
    const int rnn_dim = 5;
    RNNNode slice(rnn_dim, "Slice 0");

    const int char_dim = 7;
    const int minibatch_size = 2;
    const vector<int> x_t_extents = {char_dim, minibatch_size};
    const vector<int> h_t_extents = {rnn_dim, minibatch_size};

    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map["x_t"] = x_t_extents;
    input_port_extents_map["h_t_prev"] = h_t_extents;

    // Now check the Jacobians:
    slice.check_jacobian_weights(input_port_extents_map);
    slice.check_jacobian_bias(input_port_extents_map);
    slice.check_jacobian_input_backward(input_port_extents_map);
  }

  void test_simple_rnn() {
    cout << "test_simple_rnn()" << endl;
        // Create a local class that represents 1 time slice in the vanilla RNN.

    // Note: It is not actually necessary to create a subclass of Node explicity as below.
    // An alternative that also works is to simply get a new instance of Node, allocate
    // instances of internal Node instances (probably using unique_ptr) and calling the
    // various member functions of Node to connect everything together. The following is
    // simply intended to show one option that works.

    //
    // Input ports:
    //               "x_t"
    //               "h_t_prev"
    //
    // Output ports:
    //               "h_t": hidden output to next node (in next time slice)
    //               "y_t": output to next layer or final output (in same time slice)
    //
    // Note: Since this will be a composite node (that is, a node that contains a subgraph of
    // other nodes), all we need to do is the following:
    //
    // 1. For each contained node, add a corresponding private member variable (see below).
    // 2. In the constructor initialization list, call the constructor of each contained node (most of
    // them will only need a name, but some will require configuration parameters.
    // 3. In the constructor body, do the following in order:
    //     i. Connect each input port of this node to desired internal node input port.
    //     ii. Connect internal nodes together however you want, adding each of them via add_node() in
    //         the order that they should be called in the forward data propagation.
    //     iii. Connect outputs of internal nodes to output ports of this node.
    class RNNNode : public Node {

    public:

      // rnn_dim: dimension of internal RNN state.
      // output_dim: dimension of y_t, the output
      RNNNode(int rnn_dim, int output_dim, std::string name) :
        Node(name),
        // Note: Call the constructor of each contained node here, in the same order that
        // they appear as private member variables.
        m_linear_x_t {rnn_dim, "Linear x_t"},
        m_linear_h_t_prev {rnn_dim, "Linear h_t_prev"},
        m_adder {"Adder"},
        m_tanh {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh activation"},
        m_splitter{2, "Splitter"},
	m_linear_output {output_dim, "Linear output"}
      {
        // This node is a "composite node" since it will contain a network subgraph.
        // We now spcecify the subgraph:

        // Note: add_node() must add the contained nodes in the same order that they should
        // be called in the forward data pass.

        // Connect inputs of this node to internal nodes:
        connect_input_to_contained_node("x_t", m_linear_x_t);
        add_node(m_linear_x_t);
        connect_input_to_contained_node("h_t_prev", m_linear_h_t_prev);
        add_node(m_linear_h_t_prev);
        // Sum the outputs of the two linear nodes:
        m_adder.create_input_port_this_name(m_linear_x_t, "in1");
        m_adder.create_input_port_this_name(m_linear_h_t_prev, "in2");
        add_node(m_adder);

        // Connect the output of the adder to the input of a tanh activation:
        m_tanh.connect_parent(m_adder);
        add_node(m_tanh);

        // Connect the output of the tanh into the splitter node.
        m_splitter.connect_parent(m_tanh);
        add_node(m_splitter);

	// Connect output 1 of splitter to input of the "output" linear layer.
	m_linear_output.create_input_port_parent_name(m_splitter, "1");
	add_node(m_linear_output);

        // create output ports: 
        create_output_port(m_splitter, "0", "h_t"); // Internal state to send to next time slice.
	create_output_port_this_name(m_linear_output, "y_t"); // Output for this time slice.
      }

    private:
      // Each contain node should be a private member of this class:
      LinearLayer m_linear_x_t;
      LinearLayer m_linear_h_t_prev;
      AdderNode m_adder;
      ColumnActivationFunction m_tanh;
      SplitterNode m_splitter;
      LinearLayer m_linear_output;
    };

    // Internal size of RNN:
    const int rnn_dim = 5;
    const int output_dim = 3; // Number of unique characters
    RNNNode slice(rnn_dim, output_dim, "Slice 0");

    const int char_dim = 7;
    const int minibatch_size = 2;
    const vector<int> x_t_extents = {char_dim, minibatch_size};
    MatrixF x_t_foward(x_t_extents);
    MatrixF x_t_backward(x_t_extents);

    const vector<int> h_t_extents = {rnn_dim, minibatch_size};
    MatrixF h_t_prev_forward(h_t_extents);
    MatrixF h_t_prev_backward(h_t_extents);

    // Now create the input ports for a slice:
    slice.create_input_port(x_t_foward, x_t_backward, "x_t");
    slice.create_input_port(h_t_prev_forward, h_t_prev_backward, "h_t_prev");

    // Check gradients:
    // Only run this check on very small network size or it will take forever.
    const bool check_gradients_slice = true;
    if (check_gradients_slice) {
      std::map<std::string, std::vector<int>> input_port_extents_map1;
      input_port_extents_map1["x_t"] = x_t_extents;
      input_port_extents_map1["h_t_prev"] = h_t_extents;
      slice.check_jacobian_weights(input_port_extents_map1);
      slice.check_jacobian_bias(input_port_extents_map1);
      slice.check_jacobian_input_backward(input_port_extents_map1);
    }

    //
    // Input ports:
    //
    // "h_t_init": This initial h_t input.
    // The x_t inputs: "0", "1", ..., num_slices-1
    //
    // Output ports:
    //
    // h_t_final: the output of the last time slice.
    // The y_t outputs: "0", "1", ..., num_slices-1
    class VanillaRNN : public Node {

    public:

      VanillaRNN(int rnn_dim, int output_dim, int num_slices, std::string name) :
        Node(name)
      {
        for (int i = 0; i < num_slices; ++i) {
          string slice_name = "Slice " + std::to_string(i);
          cout << "Creating: " << slice_name << endl;
          m_slices.push_back(std::make_unique<RNNNode>(rnn_dim, output_dim, slice_name));
          Node& current_contained = *m_slices.back();
          if (i == 0) {
            cout << "Adding first rnn node." << endl;
            // Connect input port "h_t_init" of this container node to input port "h_t_prev" of the current contained node.
            connect_input_to_contained_node("h_t_init", current_contained, "h_t_prev");
          } else {
            // Make the contained node share paramters with the first of the contained nodes.
            cout << "Adding rnn node: " << i << endl;
            Node& contained_0 = *m_slices.at(0);
            current_contained.set_shared(contained_0); // Make all slices other than the first slice have shared parameters with the first slice.

            // Connect hidden output "h_t" of previous contained node to input port "h_t_prev" of the current contained node.
            Node& prev_contained = *m_slices.at(i-1);
            current_contained.create_input_port(prev_contained, "h_t", "h_t_prev");
          }
          // Connect input port "i" of the this container node to input port "x_t" of the current contained node.
          connect_input_to_contained_node(std::to_string(i), current_contained, "x_t");
          // Connect output port "y_t" of the current contained node to output port "i" of this container node.
          create_output_port(current_contained, "y_t", std::to_string(i));
          add_node(current_contained);
	  if (i == num_slices-1) {
	    // For the final slice, need to create an output port that sends the hidden state "h_t" as an output.
	    create_output_port(current_contained, "h_t", "h_t_final");
	  }
        }
      }

    private:
      vector<unique_ptr<RNNNode>> m_slices;
    };

    const int num_slices = 3;
    VanillaRNN rnn(rnn_dim, output_dim, num_slices, "Vanilla RNN");

    // Uncomment to check gradients of RNN.
    // Only run this check on very small network size or it will take forever.
    const bool check_gradients_rnn = true;
    if (check_gradients_rnn) {
      std::map<std::string, std::vector<int>> input_port_extents_map;
      input_port_extents_map.clear();
      input_port_extents_map["h_t_init"] = h_t_extents;
      for (int i = 0; i < num_slices; ++i) {
        input_port_extents_map[std::to_string(i)] = x_t_extents;
      }
      rnn.check_jacobian_weights(input_port_extents_map);
      rnn.check_jacobian_bias(input_port_extents_map);
      rnn.check_jacobian_input_backward(input_port_extents_map);
    }

  }

  void test_char_rnn_minibatch_getter() {
    cout << "test_char_rnn_minibatch_getter()" << endl;

    //string text = "abcde";
    //string text = "abcde abcde ";
    string text = "This is an example input string.";

    const int minibatch_size = 2;
    const int num_slices = 6;
    CharRNNMinibatchGetter getter(text, minibatch_size, num_slices);

    // Create the character->index map for the input text:
    map<char, int> char_to_idx = create_char_idx_map(text);

    getter.set_char_to_idx_map(char_to_idx);

    getter.next();

    //for (int n = 0; n < num_slices; ++n) {
    //const MatrixF& input_forward = getter.get_input_forward_batch(n);
    //cout << "Slice: " << n << " : input_foward" << endl << input_forward << endl;

    //const MatrixF& output_1_hot = getter.get_output_1_hot_batch(n);
    //cout << "Slice: " << n << " : output_1_hot" << endl << output_1_hot << endl;

    //const MatrixI& output_class_index = getter.get_output_class_index_batch(n);
    //cout << "Slice: " << n << " : output_class_index" << endl << output_class_index << endl;
    //cout << "--------------------" << endl;
    //}
    getter.print_current_minibatch();

    getter.next();
    cout << endl << "next " << endl << endl;
    getter.print_current_minibatch();

    getter.next();
    cout << endl << "next " << endl << endl;
    getter.print_current_minibatch();


    //for (int n = 0; n < num_slices; ++n) {
    //const MatrixF& input_forward = getter.get_input_forward_batch(n);
    //cout << "Slice: " << n << " : input_foward" << endl << input_forward << endl;

    //const MatrixF& output_1_hot = getter.get_output_1_hot_batch(n);
    //cout << "Slice: " << n << " : output_1_hot" << endl << output_1_hot << endl;

    //const MatrixI& output_class_index = getter.get_output_class_index_batch(n);
    //cout << "Slice: " << n << " : output_class_index" << endl << output_class_index << endl;
    //cout << "--------------------" << endl;
    //}


  }

  void run_all_tests() {
    test_mat_mult();
    test_mat_multiply_left_transpose();
    test_mat_multiply_left_transpose_accumulate();
    test_mat_multiply_right_transpose();
    test_mat_multiply_right_transpose_accumulate();
    test_MatrixF1D();
    test_MatrixF2D();
    test_MatrixF3D();
    test_MatrixF4D();
    test_MatrixF5D();
    test_MatrixF6D();
    test_MatrixResize();
    test_copy_to_from_submatrix();
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
    test_jacobian_LinearLayer_Node();
    test_jacobian_ConvLayer3D();
    test_select();
    test_jacobian_ConvLayer3D();
    test_SequentialLayer();
    test_SequentialLayer2();
    test_SequentialLayer3();
    test_SequentialLayer4();
    test_SequentialLayer_shared_parameters();
    test_jacobian_ImageToColumnLayer();
    test_jacobian_BoxActivationFunction();
    test_jacobian_ColumnActivationFunction();
    test_MSECostFunction();
    test_CrossEntropyCostFunction();
    test_BatchNormalization1D();
    test_Node_shared_parameters();
    test_Node_shared_parameters2();
    test_multi_port_node();
    test_AdderNode();
    //test_MeanNode();
    //test_MeanNode2();
    test_ConcatNode();
    test_SubtractorNode();
    test_MultiplyerNode();
    test_SplitterNode();
    test_SplitterNode2();
    test_ExtractorNode();
    test_rnn_slice();
    test_simple_rnn();
  }

}
