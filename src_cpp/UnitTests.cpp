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
#include "MatrixT.h"

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
#include "ConvLayer.h" // fixme: remove
#include "ConvLayer2D.h"
#include "ConvLayer3D.h"
#include "LinearLayer.h"

#include "Network2DConv3F1.h" 
#include "Network3DConv3F1.h" 
#include "Dropout1D.h"
#include "Dropout3D.h"


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
		Matrix A(rows_A, cols_A);
		
		Matrix B(rows_A, cols_B);
		B.randomize_uniform(-1.0f, 1.0f);
		Matrix C(cols_B, cols_A);
		C.randomize_uniform(-1.0f, 1.0f);

		// Make copies:
		Matrix Ac = A;
		Matrix Bc = B;
		Matrix Cc = C;

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
  /*
	void test_mat_mult_amp_2() {
		std::cout << "test_mat_mult_amp_2()..." << std::endl;
		const int rows_A = 5;
		const int cols_A = 4;
		const int cols_B = 3;
		// Compute A = B * C
		Matrix_AMP A(rows_A, cols_A);

		Matrix_AMP B(rows_A, cols_B);
		B.randomize_uniform(-1.0f, 1.0f);
		Matrix_AMP C(cols_B, cols_A);
		C.randomize_uniform(-1.0f, 1.0f);

		// Make copies:
		Matrix_AMP Ac = A;
		Matrix_AMP Bc = B;
		Matrix_AMP Cc = C;

		// Compute A = B * C using GPU:
		// Copy data in B from CPU to GPU:
		B.copy_cpu_to_gpu();
		// Copy data in C from CPU to GPU:
		C.copy_cpu_to_gpu();
		mat_multiply_amp(A, B, C);
		// Copy result in A from GPU to CPU:
		A.copy_gpu_to_cpu();
		std::cout << "A = " << std::endl << A << std::endl;
		//cout << "B = " << endl << B << endl;

		// Compute using CPU:
		matMultiply(Ac, Bc, Cc);
		std::cout << "Ac = " << std::endl << Ac << std::endl;
		//cout << "Bc = " << endl << Bc << endl;

		float tolerance = 1.0e-6;
		assert_almost_equal(A, Ac, tolerance);
		std::cout << "done" << std::endl;
	}
  */

  
	void benchmark_mat_mult()  {
		std::cout << "benchmark_mat_mult_amp_2()..." << std::endl;

		const int rows_A = 1024; // 511
		const int cols_A = 1024; // 1024
		const int cols_B = 1024; // 2048
		// Compute A = B * C
		Matrix A(rows_A, cols_A);

		Matrix B(rows_A, cols_B);
		B.randomize_uniform(-1.0f, 1.0f);
		Matrix C(cols_B, cols_A);
		C.randomize_uniform(-1.0f, 1.0f);

		// Make copies:
		Matrix Ac = A;
		Matrix Bc = B;
		Matrix Cc = C;

		// Start timer here. 
		using namespace std::chrono;
		auto t0 = high_resolution_clock::now();
		int loop_count = 400; // 1000
		for (int n = 0; n != loop_count; ++n) {
			mat_multiply_blas(A, B, C); // 
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


		
		//cout << "A = " << endl << A << endl;
		

		// Check result with slow version:
		mat_multiply_naive(Ac, Bc, Cc);
		
		//cout << "Ac = " << endl << Ac << endl;
		

		float tolerance = 1.0e-4;
		assert_almost_equal(A, Ac, tolerance);
		std::cout << "done" << std::endl;
	}
  

	void stress_test_forward_prop()   {
		std::cout << "benchmark_mat_mult_amp_2()..." << std::endl;

		const int rows_A = 1*1024; // 511
		const int cols_A = 1*1024; // 1024
		const int cols_B = 1024; // 2048
		// Compute A = B * C
		Matrix A(rows_A, cols_A);

		Matrix B(rows_A, cols_B);
		B.randomize_uniform(-1.0f, 1.0f);
		Matrix C(cols_B, cols_A);
		C.randomize_uniform(-1.0f, 1.0f);

		// Make copies:
		Matrix Ac = A;
		Matrix Bc = B;
		Matrix Cc = C;
		MatrixT<int> out_indices(A.extent(0), A.extent(1));
		Matrix out_values(A.extent(0), A.extent(1));

		// Start timer here. 
		using namespace std::chrono;
		auto t0 = high_resolution_clock::now();
		int loop_count = 400; // 1000
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
		Matrix A(rows_A, cols_A);

		Matrix B(cols_B, rows_A);
		B.randomize_uniform(-1.0f, 1.0f);
		Matrix C(cols_B, cols_A);
		C.randomize_uniform(-1.0f, 1.0f);

		// Make copies:
		Matrix Ac = A;
		Matrix Bc = B;
		Matrix Cc = C;

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
		Matrix A(rows_A, cols_A);

		Matrix B(rows_A, cols_B);
		B.randomize_uniform(-1.0f, 1.0f);
		Matrix C(cols_A, cols_B);
		C.randomize_uniform(-1.0f, 1.0f);

		// Make copies:
		Matrix Ac = A;
		Matrix Bc = B;
		Matrix Cc = C;

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
    Matrix A(rows_A, cols_A);
    
    Matrix B(rows_A, cols_B);
    B.randomize_uniform(-1.0f, 1.0f);
    Matrix C(cols_A, cols_B);
    C.randomize_uniform(-1.0f, 1.0f);

		// Make copies:
		Matrix Ac = A;
		Matrix Bc = B;
		Matrix Cc = C;

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

	void test_Matrix1D() {
		cout << "Testing test_Matrix1D...";
		const int dim0 = 5;
		Matrix mat_1d(dim0);
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

	void test_Matrix2D() {
		cout << "Testing test_Matrix2D...";
		const int dim0 = 3;
		const int dim1 = 4;
		Matrix mat_2d(dim0, dim1);
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

	void test_Matrix3D() {
		cout << "Testing test_Matrix3D...";
		const int dim0 = 2;
		const int dim1 = 3;
		const int dim2 = 4;
		Matrix mat_3d(dim0, dim1, dim2);
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

	void test_Matrix4D() {
		cout << "Testing test_Matrix4D...";
		const int dim0 = 2;
		const int dim1 = 3;
		const int dim2 = 4;
		const int dim3 = 5;
		Matrix mat_4d(dim0, dim1, dim2, dim3);
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

	void test_Matrix5D() {
		cout << "Testing test_Matrix5D...";
		const int dim0 = 2;
		const int dim1 = 3;
		const int dim2 = 4;
		const int dim3 = 5;
		const int dim4 = 6;
		Matrix mat_5d(dim0, dim1, dim2, dim3, dim4);
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

	void test_Matrix6D() {
		cout << "Testing test_Matrix6D...";
		const int dim0 = 2;
		const int dim1 = 3;
		const int dim2 = 4;
		const int dim3 = 5;
		const int dim4 = 6;
		const int dim5 = 7;
		Matrix mat_6d(dim0, dim1, dim2, dim3, dim4, dim5);
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







	void test_compute_kmax() {
		cout << "test_compute_kmax()..." << endl;
		const int M = 1024;
		const int N = 256;
		const int partition_count = 1;
		const int k = 512; // 294
		Matrix kmax_in(M, N);
		randomize_uniform(kmax_in, -1.0f, 1.0f);
		//cout << "kmax_in = " << endl << kmax_in << endl;
		Matrix kmax_out_values(M, N);
		MatrixT<int> kmax_out_indices(k*partition_count, N);
		// Compute forward-direction kmax:
		compute_forward_kmax(kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
		//cout << "kmax_out_values = " << endl << kmax_out_values << endl;
		//cout << "kmax_out_indices = " << endl << kmax_out_indices << endl;

		Matrix other_kmax_in(M, N);
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
		Matrix kmax_in(M, N);
		randomize_uniform(kmax_in, -1.0f, 1.0f);
		//cout << "kmax_in = " << endl << kmax_in << endl;
		Matrix kmax_out_values(M, N);
		MatrixT<int> kmax_out_indices(M, N);
		// Compute forward-direction kmax:
		compute_forward_kmax_v2(kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
		//cout << "kmax_out_values = " << endl << kmax_out_values << endl;
		//cout << "kmax_out_indices = " << endl << kmax_out_indices << endl;

		Matrix other_kmax_in(M, N);
		// Compute reverse-direction kmax:
		compute_reverse_kmax_v2(other_kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
		//cout << "Updated kmax_in = " << endl << other_kmax_in << endl;
		assert_almost_equal(kmax_out_values, other_kmax_in, 1e-3f);

		cout << "PASSED" << endl;
	}

	/*
	void test_gradients_2() {
		cout << "test_gradients_2()..." << endl;
		// Check gradient for the model:
		// This is convolutional layer for a conventional convolutional network.
		// Z2 = W (convolve with) A1.
		// where A1 is 2D image and W is stack of 2D filters.
		
		// Create X with random values.
		const int rows_A1 = 8;
		const int cols_A1 = 8;
		Matrix A1(rows_A1, cols_A1);
		A1.randomize_uniform(0.0f, 0.01f);
		
		Matrix deltas_A1(A1.extent(0), A1.extent(1));
		
		const int sub_rows = 4; // Convolution filter height
		const int sub_cols = 4; // Convolution filter width
		const int filter_count = 2; // 1 // Number of convolutional filters.
		//const bool force_nonnegative = true;

		// Create parameters matrix
		Matrix W(sub_rows, sub_cols, filter_count);
		W.randomize_uniform(0.0f, 0.01f);
		Matrix W_grad(sub_rows, sub_cols, filter_count);

		Matrix Z2(A1.extent(0), A1.extent(1), filter_count);
		Z2.randomize_uniform(0.0f, 0.01f);
		Matrix Z2_error(A1.extent(0), A1.extent(1), filter_count);
		Matrix Z2_approx(A1.extent(0), A1.extent(1), filter_count);


		// Compute Z2 = W (convolve with) A1:
		convolve_2d_filter(Z2_approx, W, A1);
		// Compute error matrix:
		// Z2_error = -(Z2 - Z2_approx)
		element_wise_difference(Z2_error, Z2_approx, Z2);
		
		// Compute gradient of W:
		compute_weight_grad_convolutive(W_grad, Z2_error, A1);

		// Update deltas_A1. (this is gradient of A1).
		compute_convolutive_deltas(deltas_A1, W, Z2_error); 

		/////////////////////////////////////////////////////////////////
		// Now check the gradients in W_grad 
		// the numerically-computed gradients:
		const float epsilon = 1e-4f;
		Matrix W_grad_numerical(sub_rows, sub_cols, filter_count);
		for (int n = 0; n != W.size(); ++n) {
			float orig = W[n]; // backup
			W[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1:
			convolve_2d_filter(Z2_approx, W, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			W[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1:
			convolve_2d_filter(Z2_approx, W, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			W[n] = orig;
			W_grad_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_W_score = compute_rmse(W_grad, W_grad_numerical);
		const float relative_error_grad_W_score = relative_error(W_grad, W_grad_numerical);
		//cout << "W gradient difference score RMSE = " << rmse_grad_W_score << endl;
		cout << "W gradient difference score norm closeness = " << relative_error_grad_W_score << endl;
		cout << "W_grad = " << endl << W_grad << endl;
		cout << "-----------------------------" << endl;
		cout << "W_grad_numerical = " << endl << W_grad_numerical << endl;
		//assert_almost_equal(rmse_grad_W_score, 0.0f, 1e-3f);
		assert_almost_equal(relative_error_grad_W_score, 0.0f, 1e-3f);



		/////////////////////////////////////////////////////////////////
		// Now check the gradients in deltas_A1 
		// the numerically-computed gradients:
		Matrix deltas_A1_numerical(A1.extent(0), A1.extent(1));
		for (int n = 0; n != A1.size(); ++n) {
			float orig = A1[n]; // backup
			A1[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1:
			convolve_2d_filter(Z2_approx, W, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			A1[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1:
			convolve_2d_filter(Z2_approx, W, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			A1[n] = orig;
			deltas_A1_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_A1_score = compute_rmse(deltas_A1, deltas_A1_numerical);
		//cout << "A1 gradient difference score RMSE = " << rmse_grad_A1_score << endl;
		const float grad_A1_score = relative_error(deltas_A1, deltas_A1_numerical);
		cout << "A1 gradient score = " << grad_A1_score << endl;
		cout << "deltas_A1 (A1 gradient) = " << endl << deltas_A1 << endl;
		cout << "-----------------------------" << endl;
		cout << "A1_grad_numerical = " << endl << deltas_A1_numerical << endl;
		//assert_almost_equal(rmse_grad_A1_score, 0.0f, 1e-3f);
		assert_almost_equal(grad_A1_score, 0.0f, 1e-2f);
		cout << "PASSED" << endl;
	}
	*/

	/*
	void test_gradients_3() {
		cout << "test_gradients_3()..." << endl;
		// Check gradient for the model:
		// This is convolutional layer for a conventional convolutional network.
		// Z2 = W (convolve with) A1 + bias.
		// where A1 is 2D image and W is stack of 2D filters.
		
		// Create X with random values.
		const int rows_A1 = 8;
		const int cols_A1 = 8;
		Matrix A1(rows_A1, cols_A1);
		A1.randomize_uniform(0.0f, 0.01f);
		
		Matrix deltas_A1(A1.extent(0), A1.extent(1));
		
		const int sub_rows = 4; // Convolution filter height
		const int sub_cols = 4; // Convolution filter width
		const int filter_count = 2; // 1 // Number of convolutional filters.
		//const bool force_nonnegative = true;

		// Create parameters matrix
		Matrix W(sub_rows, sub_cols, filter_count);
		W.randomize_uniform(0.0f, 0.01f);
		Matrix W_grad(sub_rows, sub_cols, filter_count);

		Matrix bias(filter_count);
		randomize_uniform(bias, -0.01f, 0.01f);
		Matrix grad_bias(filter_count);

		Matrix Z2(A1.extent(0), A1.extent(1), filter_count);
		Z2.randomize_uniform(0.0f, 0.01f);
		Matrix Z2_error(A1.extent(0), A1.extent(1), filter_count);
		Matrix Z2_approx(A1.extent(0), A1.extent(1), filter_count);


		// Compute Z2 = W (convolve with) A1 + bias:
		convolve_2d_filter_with_bias(Z2_approx, W, bias, A1);
		// Compute error matrix:
		// Z2_error = -(Z2 - Z2_approx)
		element_wise_difference(Z2_error, Z2_approx, Z2);
		
		// Compute gradient of W:
		compute_weight_grad_convolutive(W_grad, Z2_error, A1);

		// Compute gradient of bias.
		compute_bias_grad_convolutive(grad_bias, Z2_error);

		// Update deltas_A1. (this is gradient of A1).
		compute_convolutive_deltas(deltas_A1, W, Z2_error); 

		/////////////////////////////////////////////////////////////////
		// Now check the gradients in W_grad 
		// the numerically-computed gradients:
		const float epsilon = 1e-4f;
		Matrix W_grad_numerical(sub_rows, sub_cols, filter_count);
		for (int n = 0; n != W.size(); ++n) {
			float orig = W[n]; // backup
			W[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1 + bias:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			W[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1 + bias:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			W[n] = orig;
			W_grad_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_W_score = compute_rmse(W_grad, W_grad_numerical);
		//const float relative_error_grad_W_score = relative_error(W_grad.get_backing_vector(), W_grad_numerical.get_backing_vector());
		const float relative_error_grad_W_score = relative_error(W_grad, W_grad_numerical);
		//cout << "W gradient difference score RMSE = " << rmse_grad_W_score << endl;
		cout << "W gradient difference score norm closeness = " << relative_error_grad_W_score << endl;
		cout << "W_grad = " << endl << W_grad << endl;
		cout << "-----------------------------" << endl;
		cout << "W_grad_numerical = " << endl << W_grad_numerical << endl;
		//assert_almost_equal(rmse_grad_W_score, 0.0f, 1e-3f);
		assert_almost_equal(relative_error_grad_W_score, 0.0f, 1e-2f);



		/////////////////////////////////////////////////////////////////
		// Now check the gradients in deltas_A1 
		// the numerically-computed gradients:
		Matrix deltas_A1_numerical(A1.extent(0), A1.extent(1));
		for (int n = 0; n != A1.size(); ++n) {
			float orig = A1[n]; // backup
			A1[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			A1[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			A1[n] = orig;
			deltas_A1_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_A1_score = compute_rmse(deltas_A1, deltas_A1_numerical);
		//cout << "A1 gradient difference score RMSE = " << rmse_grad_A1_score << endl;
		//const float grad_A1_score = relative_error(deltas_A1.get_backing_vector(), deltas_A1_numerical.get_backing_vector());
		const float grad_A1_score = relative_error(deltas_A1, deltas_A1_numerical);
		cout << "A1 gradient score = " << grad_A1_score << endl;
		cout << "deltas_A1 (A1 gradient) = " << endl << deltas_A1 << endl;
		cout << "-----------------------------" << endl;
		cout << "A1_grad_numerical = " << endl << deltas_A1_numerical << endl;
		//assert_almost_equal(rmse_grad_A1_score, 0.0f, 1e-3f);
		assert_almost_equal(grad_A1_score, 0.0f, 1e-2f);


		/////////////////////////////////////////////////////////////////
		// Now check the gradients in grad_bias
		// the numerically-computed gradients:
		Matrix grad_bias_numerical(filter_count);
		for (int n = 0; n != bias.size(); ++n) {
			float orig = bias[n]; // backup
			bias[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			bias[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			bias[n] = orig;
			grad_bias_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_A1_score = compute_rmse(deltas_A1, deltas_A1_numerical);
		//cout << "A1 gradient difference score RMSE = " << rmse_grad_A1_score << endl;
		//const float grad_bias_score = relative_error(grad_bias.get_backing_vector(), grad_bias_numerical.get_backing_vector());
		const float grad_bias_score = relative_error(grad_bias, grad_bias_numerical);
		cout << "bias gradient score = " << grad_bias_score << endl;
		cout << "grad_bias = " << endl << grad_bias << endl;
		cout << "-----------------------------" << endl;
		cout << "grad_bias_numerical = " << endl << grad_bias_numerical << endl;
		//assert_almost_equal(rmse_grad_A1_score, 0.0f, 1e-3f);
		assert_almost_equal(grad_bias_score, 0.0f, 1e-2f);

		cout << "PASSED" << endl;
	}
	*/

	void test_gradients_convolutional_minibatch()  {
		cout << "test_gradients_convolutional_minibatch()..." << endl;
		// Check gradient for the model:
		// This is convolutional layer for a conventional convolutional network.
		// Z2 = W (convolve with) A1 + bias.
		// where A1 is 2D image and W is stack of 2D filters.
		// This is the mini-batch version.

		// Number of samples in a mini-batch.
		const int minibatch_size = 8;

		// Create X with random values.
		const int rows_A1 = 8;
		const int cols_A1 = 8;
		Matrix A1(minibatch_size, rows_A1, cols_A1);
		randomize_uniform(A1, 0.0f, 0.1f);
		Matrix deltas_A1(minibatch_size, rows_A1, cols_A1);
		
		const int sub_rows = 4; // Convolution filter height
		const int sub_cols = 4; // Convolution filter width
		const int filter_count = 2; // 1 // Number of convolutional filters.
		//const bool force_nonnegative = true;

		// Create parameters matrix
		//Matrix W(sub_rows, sub_cols, filter_count);
		Matrix W(filter_count, sub_rows, sub_cols);
		randomize_uniform(W, 0.0f, 0.1f);
		//Matrix W_grad(sub_rows, sub_cols, filter_count);
		Matrix W_grad(filter_count, sub_rows, sub_cols);

		Matrix bias(filter_count);
		randomize_uniform(bias, -0.1f, 0.1f);
		Matrix grad_bias(filter_count);

		//Matrix Z2(minibatch_size, rows_A1, cols_A1, filter_count);
		Matrix Z2(minibatch_size, filter_count, rows_A1, cols_A1);
		randomize_uniform(Z2, 0.0f, 0.1f);
		//Matrix Z2_error(minibatch_size, rows_A1, cols_A1, filter_count);
		Matrix Z2_error(minibatch_size, filter_count, rows_A1, cols_A1);
		//Matrix Z2_approx(minibatch_size, rows_A1, cols_A1, filter_count);
		Matrix Z2_approx(minibatch_size, filter_count, rows_A1, cols_A1);

		// Compute Z2 = W (convolve with) A1 + bias:
		convolve_2d_filter_with_bias_minibatch(Z2_approx, W, bias, A1);
		// Compute error matrix:
		// Z2_error = -(Z2 - Z2_approx)
		element_wise_difference(Z2_error, Z2_approx, Z2);
		
		// Compute gradient of W:
		compute_weight_grad_convolutive_minibatch(W_grad, Z2_error, A1);

		// Compute gradient of bias.
		compute_bias_grad_convolutive_minibatch(grad_bias, Z2_error);

		// Update deltas_A1. (this is gradient of A1).
		compute_convolutive_deltas_minibatch(deltas_A1, W, Z2_error); 

		/////////////////////////////////////////////////////////////////
		// Now check the gradients in W_grad 
		// the numerically-computed gradients:
		const float epsilon = 1e-3f;
		Matrix W_grad_numerical(sub_rows, sub_cols, filter_count);
		for (int n = 0; n != W.size(); ++n) {
			float orig = W[n]; // backup
			W[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1 + bias:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias_minibatch(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			W[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1 + bias:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias_minibatch(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			W[n] = orig;
			W_grad_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_W_score = compute_rmse(W_grad, W_grad_numerical);
		//const float relative_error_grad_W_score = relative_error(W_grad.get_backing_vector(), W_grad_numerical.get_backing_vector());
		const float relative_error_grad_W_score = relative_error(W_grad, W_grad_numerical);
		//cout << "W gradient difference score RMSE = " << rmse_grad_W_score << endl;
		cout << "W gradient difference score norm closeness = " << relative_error_grad_W_score << endl;
		cout << "W_grad = " << endl << W_grad << endl;
		cout << "-----------------------------" << endl;
		cout << "W_grad_numerical = " << endl << W_grad_numerical << endl;
		//assert_almost_equal(rmse_grad_W_score, 0.0f, 1e-3f);
		assert_almost_equal(relative_error_grad_W_score, 0.0f, 1e-2f);



		/////////////////////////////////////////////////////////////////
		// Now check the gradients in deltas_A1 
		// the numerically-computed gradients:
		Matrix deltas_A1_numerical(minibatch_size, rows_A1, cols_A1);
		for (int n = 0; n != A1.size(); ++n) {
			float orig = A1[n]; // backup
			A1[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias_minibatch(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			A1[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias_minibatch(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			A1[n] = orig;
			deltas_A1_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_A1_score = compute_rmse(deltas_A1, deltas_A1_numerical);
		//cout << "A1 gradient difference score RMSE = " << rmse_grad_A1_score << endl;
		//const float grad_A1_score = relative_error(deltas_A1.get_backing_vector(), deltas_A1_numerical.get_backing_vector());
		const float grad_A1_score = relative_error(deltas_A1, deltas_A1_numerical);
		cout << "A1 gradient score = " << grad_A1_score << endl;
		cout << "deltas_A1 (A1 gradient) = " << endl << deltas_A1 << endl;
		cout << "-----------------------------" << endl;
		cout << "A1_grad_numerical = " << endl << deltas_A1_numerical << endl;
		//assert_almost_equal(rmse_grad_A1_score, 0.0f, 1e-3f);
		assert_almost_equal(grad_A1_score, 0.0f, 1e-2f);


		/////////////////////////////////////////////////////////////////
		// Now check the gradients in grad_bias
		// the numerically-computed gradients:
		Matrix grad_bias_numerical(filter_count);
		for (int n = 0; n != bias.size(); ++n) {
			float orig = bias[n]; // backup
			bias[n] += epsilon;
			// Now compute J(theta_plus)
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias_minibatch(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_plus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_plus += Z2_error[i]*Z2_error[i];
			}
			J_plus *= 0.5;
			// Now compute J(theta_minus)
			bias[n] -= 2*epsilon;
			// Compute Z2_approx = W (convolve with) A1:
			//convolve_2d_filter(Z2_approx, W, A1);
			convolve_2d_filter_with_bias_minibatch(Z2_approx, W, bias, A1);
			// Compute error matrix:
			// Z2_error = -(Z2 - Z2_approx)
			element_wise_difference(Z2_error, Z2_approx, Z2);
			float J_minus = 0.0f;
			for (int i = 0; i != Z2_error.size(); ++i) {
				J_minus += Z2_error[i]*Z2_error[i];
			}
			J_minus *= 0.5;
			// Put back original value.
			bias[n] = orig;
			grad_bias_numerical[n] = (J_plus - J_minus)/(2*epsilon);
		}
		//const float rmse_grad_A1_score = compute_rmse(deltas_A1, deltas_A1_numerical);
		//cout << "A1 gradient difference score RMSE = " << rmse_grad_A1_score << endl;
		//const float grad_bias_score = relative_error(grad_bias.get_backing_vector(), grad_bias_numerical.get_backing_vector());
		const float grad_bias_score = relative_error(grad_bias, grad_bias_numerical);
		cout << "bias gradient score = " << grad_bias_score << endl;
		cout << "grad_bias = " << endl << grad_bias << endl;
		cout << "-----------------------------" << endl;
		cout << "grad_bias_numerical = " << endl << grad_bias_numerical << endl;
		//assert_almost_equal(rmse_grad_A1_score, 0.0f, 1e-3f);
		assert_almost_equal(grad_bias_score, 0.0f, 1e-2f);

		cout << "PASSED" << endl;
	}








	void test_relu() {
		cout << "test_relu()..." << endl;
		const int M = 50;
		const int N = 40;
		Matrix in_vals(M, N);
		randomize_uniform(in_vals, -1.0f, 1.0f);
		//cout << "in_vals = " << endl << in_vals << endl;
		Matrix out_values(M, N);
		MatrixT<int> out_indices(M, N);
		// Compute forward-direction ReLU:
		compute_forward_relu(in_vals, out_values, out_indices);
		//cout << "out_values = " << endl << out_values << endl;
		//cout << "out_indices = " << endl << out_indices << endl;

		Matrix other_in_vals(M, N);
		// Compute reverse-direction ReLU:
		compute_reverse_relu(other_in_vals, out_values, out_indices);
		//cout << "Updated in_vals = " << endl << other_in_vals << endl;
		assert_almost_equal(out_values, other_in_vals, 1e-3f);

		cout << "PASSED" << endl;
	}

	// deprecated
	void test_compute_maxout_3d() {
		cout << "test_compute_maxout_3d()..." << endl;

		const int image_height = 4;
		const int image_width = 10;
		const int filter_count = 12; // 4 // Number of convolutional filters.

		Matrix Z(image_height, image_width, filter_count);
		randomize_uniform(Z, -1.0f, 1.0f);
		cout << "Z = " << endl << Z << endl;
		//Matrix Z_error(image_height, image_width, filter_count);
		Matrix Z_new(image_height, image_width, filter_count);

		// Maxout parameters:
		const int maxout_factor_dim0 = 2;
		const int maxout_factor_dim1 = 5;
		const int maxout_factor_dim2 = 3;

		const int maxout_vals_dim0 = Z.extent(0) / maxout_factor_dim0;
		const int maxout_vals_dim1 = Z.extent(1) / maxout_factor_dim1;
		const int maxout_vals_dim2 = Z.extent(2) / maxout_factor_dim2;
		Matrix maxout_vals(maxout_vals_dim0, maxout_vals_dim1, maxout_vals_dim2);
		MatrixT<int> maxout_indices(maxout_vals_dim0, maxout_vals_dim1, maxout_vals_dim2, 3);

		compute_maxout_3d(Z, maxout_vals, maxout_indices); // updates maxout_values and maxout_indices
		cout << "maxout_vals = " << endl << maxout_vals << endl;
		cout << "maxout_indices = " << endl << maxout_indices << endl;

		compute_reverse_maxout_3d(Z_new, maxout_vals, maxout_indices); // updates _new
		cout << "Z_new = " << endl << Z_new << endl;

		for (int n = 0; n != Z.size(); ++n) {
			if (Z_new[n] > 0) {
				assert_almost_equal(Z[n], Z_new[n], 1e-3f);
			}
		}

		//assert_almost_equal(Z, Z_error, 1e-3f);
		cout << "PASSED" << endl;
	}

	void test_compute_maxout_3d_minibatch() {
		cout << "test_compute_maxout_3d_minibatch()..." << endl;

		const int minibatch_size = 2;
		const int filter_count = 4; // 4 // Number of convolutional filters.
		const int image_height = 6;
		const int image_width = 8;


		Matrix Z(minibatch_size, filter_count, image_height, image_width);
		randomize_uniform(Z, -1.0f, 1.0f);
		cout << "Z = " << endl << Z << endl;
		//Matrix Z_error(image_height, image_width, filter_count);
		Matrix Z_new(minibatch_size, filter_count, image_height, image_width);

		// Maxout parameters:
		const int pooling_size_depth = 3;
		const int pooling_size_height = 3;
		const int pooling_size_width = 3;

		const int depth_out = 2;
		const int height_out = 3;
		const int width_out = 4;
		Matrix maxout_vals(minibatch_size, depth_out, height_out, width_out);
		MatrixT<int> maxout_indices(minibatch_size, depth_out, height_out, width_out, 3);
		const vector<int> pooling_region_extents = {pooling_size_depth, pooling_size_height, pooling_size_width};

		//compute_maxout_3d_minibatch(Z, maxout_vals, maxout_indices); // updates maxout_values and maxout_indices
		forward_3d_max_pool(Z, maxout_vals, maxout_indices, pooling_region_extents); // updates maxout_values and maxout_indices
		cout << "maxout_vals = " << endl << maxout_vals << endl;
		cout << "maxout_indices = " << endl << maxout_indices << endl;

		//compute_reverse_maxout_3d_minibatch(Z_new, maxout_vals, maxout_indices); // updates _new
		reverse_3d_max_pool(Z_new, maxout_vals, maxout_indices); // updates _new
		cout << "Z_new = " << endl << Z_new << endl;

		for (int n = 0; n != Z.size(); ++n) {
			if (Z_new[n] > 0) {
				// fixme:
				//assert_almost_equal(Z[n], Z_new[n], 1e-3f);
			}
		}

		//assert_almost_equal(Z, Z_error, 1e-3f);
		cout << "PASSED" << endl;
	}

	/*
	void test_optimized_convolve_2d_minibatch_deprecated() {
		cout << " test_optimized_convolve_2d_minibatch_deprecated()..." << endl;
		const float pass_relative_error = 5e-3f; // Relative error must be below this to pass.

		// Number of samples in a mini-batch.
		const int minibatch_size = 128;

		// Create X with random values.
		const int image_height = 32;
		const int image_width = 32;
		const int conv_filter_height = 5; // Convolution filter height
		const int conv_filter_width = 5; // Convolution filter width
		const int filter_count = 64; // 1 // Number of convolutional filters.

		// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
		Matrix Z2_true(minibatch_size, filter_count, image_height, image_width);
		Matrix W(filter_count, conv_filter_height, conv_filter_width);
		randomize_uniform(W, -1.0f, 1.0f);
		Matrix bias(filter_count);
		randomize_uniform(bias, -1.0f, 1.0f);
		Matrix A1(minibatch_size, image_height, image_width);
		randomize_uniform(A1, -1.0f, 1.0f);

		cout << "Running naive convolution..." << endl;
		// Start timer here. 
		using namespace std::chrono;
		auto t0 = high_resolution_clock::now();
		const int loop_count = 10; // 1000
		for (int n = 0; n != loop_count; ++n) {
			//cout << "naive: n = " << n << endl;
			// Assume this version is correct.
			// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
			convolve_2d_filter_with_bias_minibatch(Z2_true, W, bias, A1); 
		}
		// Stop timer here. 
		auto t1 = high_resolution_clock::now();

		auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
		//double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
		std::cout << time_in_msec << " milliseconds for naive version." << std::endl;
		//std::cout << flops << " GFLOPS" << std::endl;


		// This will be the result of the optimized version.
		Matrix Z2_optimized(minibatch_size, filter_count, image_height, image_width);
		
		// Allocate temporary matrices that are needed by the optimized convolution function.
		Matrix temp_Z2(image_height*image_width, filter_count);
		Matrix temp_A1(image_height*image_width, conv_filter_height*conv_filter_width + 1);
		Matrix temp_W(conv_filter_height*conv_filter_width + 1, filter_count);
		cout << "Running optimized convolution..." << endl;
		auto t0_opt = high_resolution_clock::now();
		for (int n = 0; n != loop_count; ++n) {
			//cout << "opt: n = " << n << endl;
			//convolve_2d_filter_with_bias_minibatch(Z2_true, W, bias, A1); 
			// Compute Z2_optimized
			convolve_2d_filter_with_bias_minibatch_optimized_deprecated(Z2_optimized, W, bias, A1, temp_Z2, temp_A1, temp_W); 
		}
		auto t1_opt = high_resolution_clock::now();
		auto time_in_msec_opt = duration_cast<milliseconds>(t1_opt - t0_opt).count();
		std::cout << time_in_msec_opt << " milliseconds for optimized version." << std::endl;
		

		// Check Z2_optimized against the assumed correct value in Z2_true.
		const float rel_error = relative_error(Z2_true, Z2_optimized);
		cout << "relative error = " << rel_error << endl;
		assert_almost_equal(rel_error, 0.0f, pass_relative_error);

		cout << "PASSED" << endl;
	}
	*/

	void test_optimized_convolve_2d_minibatch() {
		cout << " test_optimized_convolve_2d_minibatch()..." << endl;
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

		// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
		Matrix Z2_true(minibatch_size, filter_count, image_height, image_width);
		Matrix W(filter_count, conv_filter_height, conv_filter_width);
		randomize_uniform(W, -1.0f, 1.0f);
		Matrix bias(filter_count);
		randomize_uniform(bias, -1.0f, 1.0f);
		Matrix A1(minibatch_size, image_height, image_width);
		randomize_uniform(A1, -1.0f, 1.0f);

		cout << "Running naive convolution..." << endl;
		// Start timer here. 
		using namespace std::chrono;
		auto t0 = high_resolution_clock::now();
		const int loop_count = 10; // 1000
		for (int n = 0; n != loop_count; ++n) {
			//cout << "naive: n = " << n << endl;
			// Assume this version is correct.
			// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
			convolve_2d_filter_with_bias_minibatch(Z2_true, W, bias, A1); 
		}
		// Stop timer here. 
		auto t1 = high_resolution_clock::now();

		auto time_in_msec = duration_cast<milliseconds>(t1 - t0).count();
		//double flops = 1e-6*static_cast<double>(loop_count)*(double)2 * (double)rows_A*(double)cols_A*(double)cols_B / (double)time_in_msec;
		std::cout << time_in_msec << " milliseconds for naive version." << std::endl;
		//std::cout << flops << " GFLOPS" << std::endl;


		// This will be the result of the optimized version.
		Matrix Z2_optimized(minibatch_size, filter_count, image_height, image_width);
		
		// Allocate temporary matrices that are needed by the optimized convolution function.
		Matrix temp_Z2(image_height*image_width*minibatch_size, filter_count);
		Matrix temp_A1(image_height*image_width*minibatch_size, conv_filter_height*conv_filter_width + 1);
		Matrix temp_W(conv_filter_height*conv_filter_width + 1, filter_count);
		cout << "Running optimized convolution..." << endl;
		auto t0_opt = high_resolution_clock::now();
		for (int n = 0; n != loop_count; ++n) {
			//cout << "opt: n = " << n << endl;
			//convolve_2d_filter_with_bias_minibatch(Z2_true, W, bias, A1); 
			// Compute Z2_optimized
			convolve_2d_filter_with_bias_minibatch_optimized(Z2_optimized, W, bias, A1, temp_Z2, temp_A1, temp_W); 
		}
		auto t1_opt = high_resolution_clock::now();
		auto time_in_msec_opt = duration_cast<milliseconds>(t1_opt - t0_opt).count();
		std::cout << time_in_msec_opt << " milliseconds for optimized version." << std::endl;
		

		// Check Z2_optimized against the assumed correct value in Z2_true.
		const float rel_error = relative_error(Z2_true, Z2_optimized);
		cout << "relative error = " << rel_error << endl;
		assert_almost_equal(rel_error, 0.0f, pass_relative_error);

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
		

		Matrix deltas_Z2(minibatch_size, filter_count, image_height, image_width);
		randomize_uniform(deltas_Z2, -1.0f, 1.0f);
		Matrix W(filter_count, conv_filter_height, conv_filter_width);
		randomize_uniform(W, -1.0f, 1.0f);
		Matrix bias(filter_count);
		randomize_uniform(bias, -1.0f, 1.0f);
		Matrix deltas_A1_true(minibatch_size, image_height, image_width);


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
		Matrix deltas_A1_optimized(minibatch_size, image_height, image_width);
		
		// Allocate temporary matrices that are needed by the optimized convolution function.
		Matrix temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
		randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
		Matrix temp_deltas_A1(image_height*image_width*minibatch_size, conv_filter_height*conv_filter_width + 1);
		randomize_uniform(temp_deltas_A1, -1.0f, 1.0f);
		Matrix temp_W(conv_filter_height*conv_filter_width + 1, filter_count);
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
		

		Matrix deltas_Z2(minibatch_size, filter_count, image_height, image_width);
		randomize_uniform(deltas_Z2, -1.0f, 1.0f);
		Matrix grad_W_true(filter_count, conv_filter_height, conv_filter_width);
		//Matrix bias(filter_count);
		//randomize_uniform(bias, -1.0f, 1.0f); //
		Matrix A1(minibatch_size, image_height, image_width);
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
		Matrix grad_W_optimized(filter_count, conv_filter_height, conv_filter_width);
		
		// Allocate temporary matrices that are needed by the optimized convolution function.
		Matrix temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
		randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
		Matrix temp_A1(image_height*image_width*minibatch_size, conv_filter_height*conv_filter_width + 1);
		randomize_uniform(temp_A1, -1.0f, 1.0f);
		Matrix temp_grad_W(conv_filter_height*conv_filter_width + 1, filter_count);
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
		Matrix Z2_true(minibatch_size, filter_count, image_height, image_width);
		Matrix W(filter_count, image_depth, conv_filter_height, conv_filter_width);
		randomize_uniform(W, -1.0f, 1.0f);
		Matrix bias(filter_count);
		randomize_uniform(bias, -1.0f, 1.0f);
		Matrix A1(minibatch_size, image_depth, image_height, image_width);
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
		Matrix Z2_optimized(minibatch_size, filter_count, image_height, image_width);
		
		// Allocate temporary matrices that are needed by the optimized convolution function.
		Matrix temp_Z2(image_height*image_width*minibatch_size, filter_count);
		Matrix temp_A1(image_height*image_width*minibatch_size, image_depth*conv_filter_height*conv_filter_width + 1);
		Matrix temp_W(image_depth*conv_filter_height*conv_filter_width + 1, filter_count);
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
		

		Matrix deltas_Z2(minibatch_size, filter_count, image_height, image_width);
		randomize_uniform(deltas_Z2, -1.0f, 1.0f);
		Matrix W(filter_count, image_depth, conv_filter_height, conv_filter_width);
		randomize_uniform(W, -1.0f, 1.0f);
		Matrix bias(filter_count);
		randomize_uniform(bias, -1.0f, 1.0f);
		Matrix deltas_A1_true(minibatch_size, image_depth, image_height, image_width);


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
		Matrix deltas_A1_optimized(minibatch_size, image_depth, image_height, image_width);
		
		// Allocate temporary matrices that are needed by the optimized convolution function.
		Matrix temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
		randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
		Matrix temp_deltas_A1(image_height*image_width*minibatch_size, image_depth*conv_filter_height*conv_filter_width + 1);
		randomize_uniform(temp_deltas_A1, -1.0f, 1.0f);
		Matrix temp_W(image_depth*conv_filter_height*conv_filter_width + 1, filter_count);
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
		

		Matrix deltas_Z2(minibatch_size, filter_count, image_height, image_width);
		randomize_uniform(deltas_Z2, -1.0f, 1.0f);
		Matrix grad_W_true(filter_count, image_depth, conv_filter_height, conv_filter_width);
		//Matrix bias(filter_count);
		//randomize_uniform(bias, -1.0f, 1.0f); //
		Matrix A1(minibatch_size, image_depth, image_height, image_width);
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
		Matrix grad_W_optimized(filter_count, image_depth, conv_filter_height, conv_filter_width);
		
		// Allocate temporary matrices that are needed by the optimized convolution function.
		Matrix temp_deltas_Z2(image_height*image_width*minibatch_size, filter_count);
		randomize_uniform(temp_deltas_Z2, -1.0f, 1.0f);
		Matrix temp_A1(image_height*image_width*minibatch_size, image_depth*conv_filter_height*conv_filter_width + 1);
		randomize_uniform(temp_A1, -1.0f, 1.0f);
		Matrix temp_grad_W(image_depth*conv_filter_height*conv_filter_width + 1, filter_count);
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

	/*
	void test_gradients_Network3DConvFull3L() {
		cout << "test_gradients_Network3DConvFull3L()..." << endl;

		const int minibatch_size = 2;
		const int image_height = 16;
		const int image_width = 16;
		const int image_depth = 3;
		const int dim_output = 5;
		const int conv_filter_height1 = 3;
		const int conv_filter_width1= 3;
		const int filter_count1 = 2;

		const int conv_filter_height2 = 3;
		const int conv_filter_width2= 3;
		const int filter_count2 = 2;
		const int maxpool_factor_height = 2;  // 2
		const int maxpool_factor_width = 2;  // 2
		const int maxpool_factor_filter_channels = 2; // 1
		const int dim_fully_connected_hidden = 7;

		Matrix input_activations(minibatch_size, image_depth, image_height, image_width);
		randomize_uniform(input_activations, 0.0f, 1.0f);
		Matrix output_activations(dim_output, minibatch_size);
		Matrix output_activations_train(dim_output, minibatch_size); // Training values
		randomize_uniform(output_activations_train, 0.0f, 1.0f);
		

		Network3DConvFull3L network(minibatch_size, image_height, image_width, image_depth, dim_output,
						  conv_filter_height1, conv_filter_width1, filter_count1,
									maxpool_factor_height, maxpool_factor_width, maxpool_factor_filter_channels,
									conv_filter_height2, conv_filter_width2, filter_count2,
									dim_fully_connected_hidden);
		
		bool result = network.check_gradients(input_activations, output_activations_train);
		if (result) {
			cout << "PASSED" << endl;
		} else {
			cerr << "Failed numerical gradient checks." << endl;
			exit(1);
		}
	}
	*/

	/*
	void test_gradients_Network3DConvFull3LMaxout() {
		cout << "test_gradients_Network3DConvFull3LMaxout()..." << endl;

		const int minibatch_size = 2;
		const int image_height = 16;
		const int image_width = 16;
		const int image_depth = 3;
		const int dim_output = 5;
		const int conv_filter_height1 = 3;
		const int conv_filter_width1= 3;
		const int filter_count1 = 4;

		const int conv_filter_height2 = 3;
		const int conv_filter_width2= 3;
		const int filter_count2 = 4;
		const int maxpool_factor_height = 2;  // 2
		const int maxpool_factor_width = 2;  // 2
		const int maxpool_factor_filter_channels = 2; // 1
		const int dim_fully_connected_hidden = 7;
		const int maxout_factor = 2;

		Matrix input_activations(minibatch_size, image_depth, image_height, image_width);
		randomize_uniform(input_activations, 0.0f, 1.0f);
		Matrix output_activations(dim_output, minibatch_size);
		Matrix output_activations_train(dim_output, minibatch_size); // Training values
		randomize_uniform(output_activations_train, 0.0f, 1.0f);
		

		Network3DConvFull3LMaxout network(minibatch_size, image_height, image_width, image_depth, dim_output,
						  conv_filter_height1, conv_filter_width1, filter_count1,
									maxpool_factor_height, maxpool_factor_width, maxpool_factor_filter_channels,
									conv_filter_height2, conv_filter_width2, filter_count2,
										  dim_fully_connected_hidden, maxout_factor);
		
		bool result = network.check_gradients(input_activations, output_activations_train);
		if (result) {
			cout << "PASSED" << endl;
		} else {
			cerr << "Failed numerical gradient checks." << endl;
			exit(1);
		}
	}
	*/

	/*
	void test_gradients_Network3DConvFul4LMaxout() {
		cout << "test_gradients_Network3DConvFull4LMaxout()..." << endl;

		const int minibatch_size = 2;
		const int image_height = 16;
		const int image_width = 16;
		const int image_depth = 3;
		const int dim_output = 5;
		const int conv_filter_height1 = 3;
		const int conv_filter_width1= 3;
		const int filter_count1 = 2;

		const int conv_filter_height2 = 3;
		const int conv_filter_width2= 3;
		const int filter_count2 = 2;

		const int conv_filter_height3 = 3;
		const int conv_filter_width3= 3;
		const int filter_count3 = 2;

		const int maxpool_factor_height = 2;  // 2
		const int maxpool_factor_width = 2;  // 2
		const int maxpool_factor_filter_channels = 2; // 1
		const int dim_fully_connected_hidden = 7;
		const int maxout_factor = 2;

		Matrix input_activations(minibatch_size, image_depth, image_height, image_width);
		randomize_uniform(input_activations, 0.0f, 1.0f);
		Matrix output_activations(dim_output, minibatch_size);
		Matrix output_activations_train(dim_output, minibatch_size); // Training values
		randomize_uniform(output_activations_train, 0.0f, 1.0f);
		

		Network3DConvFull4LMaxout network(minibatch_size, image_height, image_width, image_depth, dim_output,
						  conv_filter_height1, conv_filter_width1, filter_count1,
									maxpool_factor_height, maxpool_factor_width, maxpool_factor_filter_channels,
									conv_filter_height2, conv_filter_width2, filter_count2,
										  conv_filter_height3, conv_filter_width3, filter_count3,
										  dim_fully_connected_hidden, maxout_factor);
		
		bool result = network.check_gradients(input_activations, output_activations_train);
		if (result) {
			cout << "PASSED" << endl;
		} else {
			cerr << "Failed numerical gradient checks." << endl;
			exit(1);
		}
	}
	*/


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

		Matrix kmax_in(minibatch_size, depth, height, width);
		Matrix kmax_out(minibatch_size, depth, height, width);
		MatrixT<int> kmax_state(minibatch_size, depth, height, width);
		
		randomize_uniform(kmax_in, -1.0f, 1.0f);
		cout << "kmax_in = " << endl << kmax_in << endl;

		// Compute forward-direction kmax:
		compute_forward_3d_kmax(kmax_in, kmax_out, kmax_state, box_depth, box_height, box_width, k);
		cout << "kmax_out = " << endl << kmax_out << endl;
		cout << "kmax_state = " << endl << kmax_state << endl;

		Matrix other_kmax_in(minibatch_size, depth, height, width);
		// Compute reverse-direction kmax:
		//compute_reverse_kmax(other_kmax_in, kmax_out_values, kmax_out_indices, partition_count, k);
		compute_reverse_3d_kmax(other_kmax_in, kmax_out, kmax_state);
		cout << "Updated kmax_in = " << endl << other_kmax_in << endl;
		assert_almost_equal(kmax_out, other_kmax_in, 1e-3f);

		cout << "PASSED" << endl;
	}




	void test_Dropout1D() {
		cout << "test_Dropout1D()..." << endl;

		int unit_count = 10;
		int minibatch_size = 4;
		Matrix input_activations(unit_count, minibatch_size);
		randomize_uniform(input_activations, 0.0f, 1.0f);
		Matrix back_prop_input_activations(unit_count, minibatch_size);
		Matrix output_activations(unit_count, minibatch_size);
		float prob_keep = 0.7f;
		Dropout1D dropouter(input_activations.get_extents(), prob_keep);
		cout << "input_activations = " << input_activations << endl;
		dropouter.forward_dropout(input_activations);
		cout << "dropouter.m_output = " << dropouter.m_output << endl;
		copy_matrix(dropouter.m_output_deltas, dropouter.m_output);
		dropouter.reverse_dropout(back_prop_input_activations);
		cout << "back_prop_input_activations = " << back_prop_input_activations << endl;

	}

	void test_Dropout3D() {
		cout << "test_Dropout3D()..." << endl;

		const int minibatch_size = 2;
		const int depth = 2;
		const int height = 2;
		const int width = 2;
		Matrix input_activations(minibatch_size, depth, height, width);
		randomize_uniform(input_activations, 0.0f, 1.0f);
		Matrix back_prop_input_activations(minibatch_size, depth, height, width);
		Matrix output_activations(minibatch_size, depth, height, width);
		float prob_keep = 0.7f;
		Dropout3D dropouter(input_activations.get_extents(), prob_keep);
		cout << "input_activations = " << input_activations << endl;
		dropouter.forward_dropout(input_activations);
		cout << "dropouter.m_output = " << dropouter.m_output << endl;

		copy_matrix(dropouter.m_output_deltas, dropouter.m_output);
		dropouter.reverse_dropout(back_prop_input_activations);
		cout << "back_prop_input_activations = " << back_prop_input_activations << endl;

	}


	void test_gradients_Network2DConv3F1() {
		cout << "test_gradients_Network2DConv3F1()..." << endl;

		const int minibatch_size = 8;
		const int image_height = 16;
		const int image_width = 16;
		const int dim_output = 5;

		const int conv_filter_height1 = 3;
		const int conv_filter_width1= 3;
		const int filter_count1 = 2;
		// For pooling layer 1
		const vector<int> pooling_region_extents1 = {1, 2, 2};
		// (depth, height, width)
		const vector<int> pooling_output_extents1 = {filter_count1, 8, 8};

		const int conv_filter_height2 = 3;
		const int conv_filter_width2= 3;
		const int filter_count2 = 4;
		// For pooling layer 2
		const vector<int> pooling_region_extents2 = {1, 2, 2};
		// (depth, height, width)
		const vector<int> pooling_output_extents2 = {filter_count2, 4, 4};

		const int conv_filter_height3 = 3;
		const int conv_filter_width3= 3;
		const int filter_count3 = 8;
		// For pooling layer 3
		const vector<int> pooling_region_extents3 = {1, 2, 2};
		// (depth, height, width)
		const vector<int> pooling_output_extents3 = {filter_count3, 2, 2};

		const int dim_fully_connected_hidden = 7;
		const int maxout_factor = 1;

		// Amount of dropout for each layer. Expresssed as probability of keeping an activation.
		const vector<float> dropout_keep_probabilities = {1.0f, 1.0f, 1.0f, 1.0f};

		BoxActivationFunction::ACTIVATION_TYPE box_activation_type = BoxActivationFunction::ACTIVATION_TYPE::leakyReLU;
		ColumnActivationFunction::ACTIVATION_TYPE col_activation_type = ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU;

		Matrix input_activations(minibatch_size, image_height, image_width);
		randomize_uniform(input_activations, 0.0f, 1.0f);
		Matrix output_activations(dim_output, minibatch_size);
		Matrix output_activations_train(dim_output, minibatch_size); // Training values
		randomize_uniform(output_activations_train, 0.0f, 1.0f);

		Network2DConv3F1 network(input_activations.get_extents(), 
										filter_count1, conv_filter_height1, conv_filter_width1,
										pooling_region_extents1, pooling_output_extents1,
										filter_count2, conv_filter_height2, conv_filter_width2,
										pooling_region_extents2, pooling_output_extents2,
										filter_count3, conv_filter_height3, conv_filter_width3,
										pooling_region_extents3, pooling_output_extents3,
										dim_output,
										dim_fully_connected_hidden, maxout_factor,
										box_activation_type, col_activation_type,
										dropout_keep_probabilities);

		cout << "OK so far..." << endl;
		
		bool result = network.check_gradients(input_activations, output_activations_train);
		if (result) {
			cout << "PASSED" << endl;
		} else {
			cerr << "Failed numerical gradient checks." << endl;
			exit(1);
		}
	}


	void test_gradients_Network3DConv3F1() {
		cout << "test_gradients_Network3DConv3F1()..." << endl;

		const int minibatch_size = 8;
		const int image_height = 16;
		const int image_width = 16;
		const int image_depth = 3;
		const int dim_output = 5;

		const int conv_filter_height1 = 3;
		const int conv_filter_width1= 3;
		const int filter_count1 = 2;
		// For pooling layer 1
		const vector<int> pooling_region_extents1 = {1, 2, 2};
		// (depth, height, width)
		const vector<int> pooling_output_extents1 = {filter_count1, 8, 8};

		const int conv_filter_height2 = 3;
		const int conv_filter_width2= 3;
		const int filter_count2 = 4;
		// For pooling layer 2
		const vector<int> pooling_region_extents2 = {1, 2, 2};
		// (depth, height, width)
		const vector<int> pooling_output_extents2 = {filter_count2, 4, 4};

		const int conv_filter_height3 = 3;
		const int conv_filter_width3= 3;
		const int filter_count3 = 8;
		// For pooling layer 3
		const vector<int> pooling_region_extents3 = {1, 2, 2};
		// (depth, height, width)
		const vector<int> pooling_output_extents3 = {filter_count3, 2, 2};

		const int dim_fully_connected_hidden = 7;
		const int maxout_factor = 1;

		// Amount of dropout for each layer. Expresssed as probability of keeping an activation.
		//const vector<float> dropout_keep_probabilities = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
		const vector<float> dropout_keep_probabilities = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

		BoxActivationFunction::ACTIVATION_TYPE box_activation_type = BoxActivationFunction::ACTIVATION_TYPE::leakyReLU;
		ColumnActivationFunction::ACTIVATION_TYPE col_activation_type = ColumnActivationFunction::ACTIVATION_TYPE::leakyReLU;

		Matrix input_activations(minibatch_size, image_depth, image_height, image_width);
		randomize_uniform(input_activations, 0.0f, 1.0f);
		Matrix output_activations(dim_output, minibatch_size);
		Matrix output_activations_train(dim_output, minibatch_size); // Training values
		randomize_uniform(output_activations_train, 0.0f, 1.0f);

		Network3DConv3F1 network(input_activations.get_extents(), 
										filter_count1, conv_filter_height1, conv_filter_width1,
										pooling_region_extents1, pooling_output_extents1,
										filter_count2, conv_filter_height2, conv_filter_width2,
										pooling_region_extents2, pooling_output_extents2,
										filter_count3, conv_filter_height3, conv_filter_width3,
										pooling_region_extents3, pooling_output_extents3,
										dim_output,
										dim_fully_connected_hidden, maxout_factor,
										box_activation_type, col_activation_type,
										dropout_keep_probabilities);

		
		
		bool result = network.check_gradients(input_activations, output_activations_train);
		if (result) {
			cout << "PASSED" << endl;
		} else {
			cerr << "Failed numerical gradient checks." << endl;
			exit(1);
		}
	}


	void run_all_tests() {
		test_mat_mult();
		test_mat_multiply_left_transpose();
		test_mat_multiply_right_transpose();
		test_Matrix1D();
		test_Matrix2D();
		test_Matrix3D();
		test_Matrix4D();
		test_Matrix5D();
		test_Matrix6D();
		//

		
		//test_gradients_2();
		test_compute_kmax();
		test_compute_kmax_v2();
		test_relu();
		//test_compute_maxout_3d();
		test_compute_maxout_3d_minibatch();
		//test_gradients_3();
		test_gradients_convolutional_minibatch();

		//test_gradients_Network2DConvFull();
		//test_gradients_Network2DConvFull2L();
		
		test_optimized_convolve_2d_minibatch();
		test_optimized_convolutive_deltas();
		test_optimized_weight_grad_convolutive();

		test_optimized_convolve_3d_minibatch();
		test_optimized_3d_convolutive_deltas();
		test_optimized_3d_weight_grad_convolutive();
		//test_gradients_Network2DConvFull3L();
		//test_gradients_Network3DConvFull3L();
		//test_gradients_Network3DConvFull3LMaxout();
		//test_gradients_Network3DConvFul4LMaxout();
		//test_gradients_Network3DConvFul4LExp1();

		test_compute_3d_kmax();


		test_Dropout1D();
		test_Dropout3D();
		test_gradients_Network2DConv3F1();
		test_gradients_Network3DConv3F1();
	}

}
