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
#include "Utilities.h"
#include "Constants.h"
#include <string>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>

// Use OpenBLAS for optimized matrix multiplication:
#include <cblas.h>


// Uncomment following line to disable assertion checking.
// #define NDEBUG
#include <assert.h>
using namespace std;

namespace kumozu {


  void mat_multiply_naive(Matrix& A, const Matrix &B, const Matrix &C) {
    int rowsOut = B.extent(0);
		int innerDim = B.extent(1);
		int colsOut = C.extent(1);
		if ((A.extent(0) != rowsOut) || (A.extent(1) != C.extent(1)) || (B.extent(1) != C.extent(0))) {
			std::cerr << "Error: Inconsistent matrix dimensions! Exiting." << std::endl;
			exit(1);
		}
		float sum;
		// For each row of B
		for (int i = 0; i < rowsOut; i++) {
			// For each column of C
			for (int j = 0; j < colsOut; j++) {
				// Compute dot product of row i of B with column j of C.
				sum = 0;
				for (int k = 0; k < innerDim; k++) {
					sum += B.get(i, k) * C.get(k, j);
				}
				A.set(i, j, sum);
			}
		}
  }


	void mat_multiply(Matrix& A, const Matrix &B, const Matrix &C) {
	  mat_multiply_blas(A, B, C); // Optimized BLAS version.
	  //mat_multiply_naive(A, B, C); // Super slow naive version.
	}

	void mat_multiply(Matrix& A, const Matrix& B, const Matrix& C, float alpha, float beta) {
		mat_multiply_blas(A, B, C, alpha, beta);
	}


  // Use this if you have an optimized BLAS implementation (requires you include cblas.h)
	void mat_multiply_blas(Matrix& A, const Matrix &B, const Matrix &C) {
		mat_multiply_blas(A, B, C, 1.0f, 0.0f);
	}

	void mat_multiply_blas(Matrix& A, const Matrix &B, const Matrix &C, float alpha, float beta) {
		// Compute A = alpha*B*C + beta*A
		if ((A.extent(0) != B.extent(0)) || (A.extent(1) != C.extent(1)) || (B.extent(1) != C.extent(0))) {
			std::cerr << "Error: Inconsistent matrix dimensions! Exiting." << std::endl;
			exit(1);
		}

		float* backingArrayA = A.get_backing_data();
		const float* backingArrayB = B.get_backing_data();
		const float* backingArrayC = C.get_backing_data();

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.extent(0), A.extent(1), B.extent(1), alpha,
			    backingArrayB, B.extent(1), backingArrayC, C.extent(1), beta, backingArrayA, A.extent(1));
	
	}


	void element_wise_multiply(Matrix& A, const Matrix &B, const Matrix &C) {
		check_dimensions(A, B);
		check_dimensions(A, C);
		for (int i = 0; i < A.size(); i++) {
			A[i] = B[i] * C[i];
		}
	}

	
	void randomize_uniform(Matrix& A, float min, float max) {
		static std::mt19937 mersenne_twister_engine;
		mersenne_twister_engine.seed(static_cast<unsigned long>(time(NULL)));
		std::uniform_real_distribution<float> uni(min, max);
		for (int i = 0; i < A.size(); i++) {
			A[i] = uni(mersenne_twister_engine);
		}
	}

	void randomize_normal(Matrix& A, float mean, float std_deviation) {
		static std::mt19937 mersenne_twister_engine;
		mersenne_twister_engine.seed(static_cast<unsigned long>(time(NULL)));
		std::normal_distribution<float> normal_dist(mean, std_deviation);
		for (int i = 0; i < A.size(); i++) {
			A[i] = normal_dist(mersenne_twister_engine);
		}
	}



	void element_wise_divide(Matrix& A, const Matrix &B, const Matrix &C, const float epsilon) {
		check_dimensions(A, B);
		check_dimensions(A, C);
		for (int i = 0; i < A.size(); i++) {
			A[i] = (B[i] + epsilon) / (C[i] + epsilon);
		}
	}


	void element_wise_difference(Matrix& A, const Matrix &B, const Matrix &C) {
		check_dimensions(A, B);
		check_dimensions(A, C);
		#pragma omp parallel for
		for (int i = 0; i < A.size(); ++i) {
			A[i] = B[i] - C[i];
		}
	}


	void element_wise_square(Matrix& A, const Matrix& B) {
		check_dimensions(A, B);
		for (int i = 0; i < A.size(); i++) {
			A[i] = B[i] * B[i];
		}
	}

	float sum(Matrix& A) {
		float sum = 0;
		for (int i = 0; i < A.size(); i++) {
			sum += A[i];
		}
		return sum;
	}

	void transpose(Matrix& A, const Matrix& B) {
		int rowsB = B.extent(0);
		int colsB = B.extent(1);
		int rowsA = A.extent(0);
		int colsA = A.extent(1);
		if ((rowsA != colsB) || (colsA != rowsB)) {
			std::cerr << "Inconsistent matrix dimensions in transpose()! Exiting.";
			exit(1);
		}
		// For each row of B
		for (int i = 0; i < rowsB; i++) {
			// For each column of C
			for (int j = 0; j < colsB; j++) {
				A.set(j, i, B.get(i, j));
			}
		}
	}

	

	void square_root(Matrix& A) {
		for (int i = 0; i < A.size(); i++) {
			A[i] = sqrtf(A[i]);
		}
	}

	

	void add_scalar(Matrix& A, float b) {
		for (int i = 0; i < A.size(); i++) {
			A[i] = A[i] + b;
		}

	}

	void scale(Matrix& A, const Matrix& B, float scaleFactor) {
		check_dimensions(A, B);
		#pragma omp parallel for
		for (int i = 0; i < A.size(); ++i) {
			A[i] = B[i]*scaleFactor;
		}
	}



	void element_wise_sum(Matrix& A, const Matrix& B, const Matrix& C) {
		check_dimensions(A, B);
		check_dimensions(A, C);
		for (int i = 0; i < A.size(); i++) {
			A[i] = B[i] + C[i];
		}
	}

	void copy_matrix(Matrix& A, const Matrix& B) {
		check_dimensions(A, B);
		// Copy contents of B into A.
		#pragma omp parallel for
		for (int i = 0; i < A.size(); i++) {
			A[i] = B[i];
		}
	}

		void check_matrix_factorization_dimensions(const Matrix& X, const Matrix& W, const Matrix& H) {
		if (X.extent(0) != W.extent(0)) {
			std::cerr << "Error: X and W don't have the same number of rows!" << std::endl;
			exit(1);
		}
		else if (X.extent(1) != H.extent(1)) {
			std::cerr << "Error: X and H don't have the same number of columns!" << std::endl;
			std::cerr << "Columns in X = " << X.extent(1) << std::endl;
			std::cerr << "Columns in H = " << H.extent(1) << std::endl;
			exit(1);
		}
		else if (W.extent(1) != H.extent(0)) {
			std::cerr << "Error: Number of columns in W does not equal number of rows in H!" << std::endl;
			std::cerr << "Columns in W = " << W.extent(1) << std::endl;
			std::cerr << "Rows in H = " << H.extent(0) << std::endl;
			exit(1);
		}
	}





	bool check_dimensions_a_eq_b_tran_times_c(const Matrix& A, const Matrix& B, const Matrix& C) {
		// Check that dimensions are consistant with A = B^T * C
		if (A.extent(0) != B.extent(1)) {
			return false;
		}
		else if (A.extent(1) != C.extent(1)) {
			return false;
		}
		else if (B.extent(0) != C.extent(0)) {
			return false;
		}
		return true;
	}


	bool check_dimensions_a_eq_b_times_c_tran(const Matrix& A, const Matrix& B, const Matrix& C) {
		// Check that dimensions are consistant with A = B^T * C
		if (A.extent(0) != B.extent(1)) {
			return false;
		}
		else if (A.extent(1) != C.extent(0)) {
			return false;
		}
		else if (B.extent(0) != C.extent(1)) {
			return false;
		}
		return true;
	}

	void mat_multiply_left_transpose(Matrix& A, const Matrix& B, const Matrix& C) {
	  // Compute A = B^T * C
	  //cout << "A rows = " << A.extent(0) << endl;
	  //cout << "A cols = " << A.extent(1) << endl;
	  //cout << "B rows = " << B.extent(0) << endl;
	  //cout << "B cols = " << B.extent(1) << endl;
	  //cout << "C rows = " << C.extent(0) << endl;
	  //cout << "C cols = " << C.extent(1) << endl;
	  //exit(1);
	  check_dimensions_a_eq_b_tran_times_c(A, B, C);
	  float* backingArrayA = A.get_backing_data();
	  const float* backingArrayB = B.get_backing_data();
	  const float* backingArrayC = C.get_backing_data();

	  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A.extent(0), A.extent(1), B.extent(0), 1.0f,
	  	      backingArrayB, B.extent(1), backingArrayC, C.extent(1), 0.0f, backingArrayA, A.extent(1));

	}

	void mat_multiply_left_transpose_naive(Matrix& A, const Matrix& B, const Matrix& C) {
		// Compute A = B^T * C
		check_dimensions_a_eq_b_tran_times_c(A, B, C);
		float new_val_A;
		int row, col, cur_feature;
		int rows_A = A.extent(0);
		int cols_A = A.extent(1);
		#pragma omp parallel for private(row, col, cur_feature, new_val_A)
		for (col = 0; col < cols_A; col++) {
			for (row = 0; row < rows_A; row++) {
				new_val_A = 0.0f;
				for (cur_feature = 0; cur_feature < static_cast<int>(B.extent(0)); cur_feature++) {
					new_val_A += (B(cur_feature, row)) * (C(cur_feature, col));
				}
				A(row, col) = new_val_A;
			}
		}
	}


	void mat_multiply_right_transpose(Matrix& A, const Matrix& B, const Matrix& C) {
	  // Compute A = B * C^T
	  check_dimensions_a_eq_b_times_c_tran(A, B, C);
	  //cout << "A rows = " << A.extent(0) << endl;
	  //cout << "A cols = " << A.extent(1) << endl;
	  //cout << "B rows = " << B.extent(0) << endl;
	  //cout << "B cols = " << B.extent(1) << endl;
	  //cout << "C rows = " << C.extent(0) << endl;
	  //cout << "C cols = " << C.extent(1) << endl;
	  //exit(1);

	  float* backingArrayA = A.get_backing_data();
	  const float* backingArrayB = B.get_backing_data();
	  const float* backingArrayC = C.get_backing_data();
	  
	  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B.extent(0), C.extent(0), B.extent(1), 1.0f,
	  	      backingArrayB, B.extent(1), backingArrayC, C.extent(1), 0.0f, backingArrayA, A.extent(1));

	}

	void mat_multiply_right_transpose_naive(Matrix& A, const Matrix& B, const Matrix& C)  {
		// Compute A = B * C^T
		check_dimensions_a_eq_b_times_c_tran(A, B, C);
		float new_val_A;
		int row, col, cur_feature;
		int rows_A = A.extent(0);
		int cols_A = A.extent(1);
#pragma omp parallel for private(row, col, cur_feature, new_val_A)
		for (col = 0; col < cols_A; col++) {
			for (row = 0; row < rows_A; row++) {
				new_val_A = 0.0f;
				for (cur_feature = 0; cur_feature < static_cast<int>(B.extent(1)); cur_feature++) {
					new_val_A += (B(row, cur_feature)) * (C(col, cur_feature));
				}
				A(row, col) = new_val_A;
			}
		}
	}

	float compute_test_error(const Matrix& network_output, const vector<float>& true_output) {
		float number_correct = 0.0f;
		int num_classes = network_output.extent(0);
		int test_data_count = network_output.extent(1);
		if (test_data_count != static_cast<int>(true_output.size())) {
			cerr << "compute_test_error(): Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		for (int c = 0; c < test_data_count; ++c) {
			// Get max value for each column of network_output.
			int max_row = 0;
			float max_val = network_output(0, c);
			for (int r = 1; r < num_classes; ++r) {
				if (network_output(r, c) > max_val) {
					max_val = network_output(r, c);
					max_row = r;
				}
			}
			if (max_row == true_output[c]) {
				// Correct!
				++number_correct;
			}
		}
		return 1.0f - number_correct / test_data_count;
	}


	void compute_forward_maxout(const Matrix& in_mat, Matrix& maxout_values_mat, MatrixT<int>& maxout_indices_mat)  {
		int cols = in_mat.extent(1);
		if (cols != maxout_values_mat.extent(1)) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		if (maxout_values_mat.extent(0) != maxout_indices_mat.extent(0)) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		int in_rows = in_mat.extent(0);
		int out_rows = maxout_indices_mat.extent(0);
		if (in_rows < out_rows) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		if (in_rows % out_rows != 0) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		int dim_reduce_factor = in_rows / out_rows;
		int row_in, out_ind, start, stop, c, row_out, temp_row;
		float out_val;
#pragma omp parallel for private(row_in, out_ind, start, stop, c, row_out, temp_row, out_val)
		for (c = 0; c < maxout_values_mat.extent(1); ++c) {
			for (row_out = 0; row_out < maxout_values_mat.extent(0); ++row_out) {
				// Number of parallel independent threads = size of maxout_values_mat.
				row_in = row_out*dim_reduce_factor;
				out_val = in_mat(row_in, c);
				out_ind = row_in;
				start = row_in + 1;
				stop = row_in + dim_reduce_factor;
				for (temp_row = start; temp_row < stop; ++temp_row) {
				  if (in_mat(temp_row, c) > out_val) {
						out_val = in_mat(temp_row, c);
						out_ind = temp_row;
					}
				}
				maxout_values_mat(row_out, c) = out_val;
				maxout_indices_mat(row_out, c) = out_ind;
			}
		}
		
	}


	void compute_reverse_maxout_with_zeros(Matrix& in_mat, const Matrix& maxout_values_mat, const MatrixT<int>& maxout_indices_mat) {
		int cols = in_mat.extent(1);
		if (cols != maxout_values_mat.extent(1)) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		if (maxout_values_mat.extent(0) != maxout_indices_mat.extent(0)) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		int in_rows = in_mat.extent(0);
		int out_rows = maxout_indices_mat.extent(0);
		if (in_rows < out_rows) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		if (in_rows % out_rows != 0) {
			cerr << "Inconsistent dimensions. Exiting." << endl;
			exit(1);
		}
		float val;
		int r, in_row;
		for (r = 0; r != in_rows; ++r) {
		  for (int c = 0; c != cols; ++c) {
			
				// Zero out all values in in_mat first.
				in_mat(r, c) = 0.0f;
			}
		}
		for (r = 0; r != out_rows; ++r) {
		  for (int c = 0; c != cols; ++c) {
			
				in_row = maxout_indices_mat(r, c);
				val = maxout_values_mat(r, c);
				in_mat(in_row, c) = val;
			}
		}
	}

	
	void compute_forward_kmax(const Matrix& kmax_in, Matrix& kmax_out_values, MatrixT<int>& kmax_out_indices,
							  int partition_count, int k) {
		// Implementation note: Each sub-partition of a column can be executed in parallel.

		// Check dimensions.
		check_dimensions(kmax_in, kmax_out_values);
		int M = kmax_in.extent(0);
		int N = kmax_in.extent(1);
		if ((M % partition_count) != 0) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			cerr << "kmax_in.extent(0) = " << kmax_in.extent(0) << endl;
			cerr << "partition_count = " << partition_count << endl;
			exit(1);
		}
		int partition_size = M/partition_count;
		if (k > partition_size) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			cerr << "k = " << k << endl;
			cerr << "partition_size = " << partition_size << endl;
			exit(1);
		}
		if (N != kmax_out_indices.extent(1)) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			cerr << "kmax_in.extent(1) = " << kmax_in.extent(1) << endl;
			cerr << "kmax_out_indices.extent(1) = " << kmax_out_indices.extent(1) << endl;
			exit(1);
		}
		if (kmax_out_indices.extent(0) != k*partition_count) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			cerr << "kmax_out_indices.extent(0) = " << kmax_out_indices.extent(0) << endl;
			cerr << "k*partition_count = " << k*partition_count << endl;
			exit(1);
		}
		set_value(kmax_out_values, 0.0f);
		set_value(kmax_out_indices, 0); // Might not be necessary.
		//vector<float> sorted_vals(partition_size);
		//vector<int> sorted_indices(partition_size);
		//#pragma omp parallel for private(sorted_vals, sorted_indices)
		const int reuse_count = 16;
		#pragma omp parallel for
		for (int col_counter = 0; col_counter < N; col_counter += reuse_count) {
			vector<float> sorted_vals(partition_size);
			vector<int> sorted_indices(partition_size);
			// Reuse each vector for reuse_count iterations.
			for (int inner_counter = 0; inner_counter < reuse_count; ++inner_counter) {
				int col = col_counter + inner_counter;
				for (int p = 0; p < partition_count; ++p) {
				
					for (int q = 0; q < partition_size; ++q) {
						// Copy data to be sorted into the vectors.
						sorted_vals[q] = kmax_in(q + p*partition_size, col);
						sorted_indices[q] = q;
					}
				
					// Now do the sorting in descending order.
					// Indices are sorted but sorted_vals remains unmodified.
					// Sorted values are then: sorted_vals[sorted_indices[0]], sorted_vals[sorted_indices[1]], sorted_vals[sorted_indices[2]], ...
					
					sort(sorted_indices.begin(), sorted_indices.end(), [&sorted_vals](size_t i0, size_t i1) {
							return sorted_vals[i0] > sorted_vals[i1];
						});
					
					// Can also try sorting by magnitude (does not work as well).
					/*
					sort(sorted_indices.begin(), sorted_indices.end(), [&sorted_vals](size_t i0, size_t i1) {
							return abs(sorted_vals[i0]) > abs(sorted_vals[i1]);
						});
					*/
					// Now update the output matrices with these sorted results.
				
					for (int n = 0; n < k; ++n) {
						kmax_out_values(sorted_indices[n] + p*partition_size, col) = sorted_vals[sorted_indices[n]];
						kmax_out_indices(k*p + n, col) = sorted_indices[n] + p*partition_size;
					}
				}
			}
		}
	}

	void compute_forward_kmax_v2(const Matrix& kmax_in, Matrix& kmax_out_values, MatrixT<int>& kmax_out_indices,
							  int partition_count, int k) {
		// Implementation note: Each sub-partition of a column can be executed in parallel.

		// Check dimensions.
		check_dimensions(kmax_in, kmax_out_values);
		check_dimensions(kmax_in, kmax_out_indices);

		int M = kmax_in.extent(0);
		int N = kmax_in.extent(1);
		if ((M % partition_count) != 0) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			cerr << "kmax_in.extent(0) = " << kmax_in.extent(0) << endl;
			cerr << "partition_count = " << partition_count << endl;
			exit(1);
		}
		int partition_size = M/partition_count;
		if (k > partition_size) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			cerr << "k = " << k << endl;
			cerr << "partition_size = " << partition_size << endl;
			exit(1);
		}
		
		set_value(kmax_out_values, 0.0f);
		set_value(kmax_out_indices, 0); // Might not be necessary.
		//vector<float> sorted_vals(partition_size);
		//vector<int> sorted_indices(partition_size);
		//#pragma omp parallel for private(sorted_vals, sorted_indices)
		const int reuse_count = 16;
		#pragma omp parallel for
		for (int col_counter = 0; col_counter < N; col_counter += reuse_count) {
			vector<float> sorted_vals(partition_size);
			vector<int> sorted_indices(partition_size);
			// Reuse each vector for reuse_count iterations.
			for (int inner_counter = 0; inner_counter < reuse_count; ++inner_counter) {
				int col = col_counter + inner_counter;
				for (int p = 0; p < partition_count; ++p) {
				
					for (int q = 0; q < partition_size; ++q) {
						// Copy data to be sorted into the vectors.
						sorted_vals[q] = kmax_in(q + p*partition_size, col);
						sorted_indices[q] = q;
					}
				
					// Now do the sorting in descending order.
					// Indices are sorted but sorted_vals remains unmodified.
					// Sorted values are then: sorted_vals[sorted_indices[0]], sorted_vals[sorted_indices[1]], sorted_vals[sorted_indices[2]], ...
					
					sort(sorted_indices.begin(), sorted_indices.end(), [&sorted_vals](size_t i0, size_t i1) {
							return sorted_vals[i0] > sorted_vals[i1];
						});
					
					// Can also try sorting by magnitude (does not work as well).
					/*
					sort(sorted_indices.begin(), sorted_indices.end(), [&sorted_vals](size_t i0, size_t i1) {
							return abs(sorted_vals[i0]) > abs(sorted_vals[i1]);
						});
					*/
					// Now update the output matrices with these sorted results.
				
					for (int n = 0; n < k; ++n) {
						kmax_out_values(sorted_indices[n] + p*partition_size, col) = sorted_vals[sorted_indices[n]];
						kmax_out_indices(sorted_indices[n] + p*partition_size, col) = 1;
					}
				}
			}
		}
	}
	
	void compute_reverse_kmax(Matrix& kmax_in, const Matrix& kmax_out_values, const MatrixT<int>& kmax_out_indices,
							  int partition_count, int k) {
		// Implementation note: Each sub-partition of a column can be executed in parallel.
		// Check dimensions.
		check_dimensions(kmax_in, kmax_out_values);
		int M = kmax_in.extent(0);
		int N = kmax_in.extent(1);
		if ((M % partition_count) != 0) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			exit(1);
		}
		int partition_size = M/partition_count;
		if (k > partition_size) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			exit(1);
		}
		if (N != kmax_out_indices.extent(1)) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			exit(1);
		}
		if (kmax_out_indices.extent(0) != k*partition_count) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			exit(1);
		}
		set_value(kmax_in, 0.0f);
		for (int row = 0; row != kmax_out_indices.extent(0); ++row) {
			for (int col = 0; col != N; ++col) {
				int in_row = kmax_out_indices(row, col);
				kmax_in(in_row, col) = kmax_out_values(in_row, col);
			}
		}
	}

	void compute_reverse_kmax_v2(Matrix& kmax_in, const Matrix& kmax_out_values, const MatrixT<int>& kmax_out_indices,
							  int partition_count, int k) {
		// Implementation note: Each sub-partition of a column can be executed in parallel.
		// Check dimensions.
		check_dimensions(kmax_in, kmax_out_values);
		check_dimensions(kmax_in, kmax_out_indices);
		int M = kmax_in.extent(0);
		int N = kmax_in.extent(1);
		if ((M % partition_count) != 0) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			exit(1);
		}
		int partition_size = M/partition_count;
		if (k > partition_size) {
			cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
			exit(1);
		}
		
		set_value(kmax_in, 0.0f);
		for (int row = 0; row != kmax_out_indices.extent(0); ++row) {
			for (int col = 0; col != N; ++col) {
				if (kmax_out_indices(row, col) == 1) {
					kmax_in(row, col) = kmax_out_values(row, col);
				}
			}
		}
	}

	void compute_forward_3d_kmax(const Matrix& kmax_in, Matrix& kmax_out, MatrixT<int>& kmax_state,
			int box_depth, int box_height, int box_width, int k) {
		// kmax_in: (minibatch_size x depth x height x width) matrix containing the input values.
		const int minibatch_size = kmax_in.extent(0);
		const int depth = kmax_in.extent(1);
		const int height = kmax_in.extent(2);
		const int width = kmax_in.extent(3);
		bool bad_parameters = false;
		check_dimensions(kmax_in, kmax_out);
		check_dimensions(kmax_in, kmax_state);
		if (bad_parameters) {
			cerr << "compute_forward_3d_kmax(): bad parameters." << endl;
			exit(1);
		}
		set_value(kmax_out, 0.0f);
		set_value(kmax_state, 0);
		const int partition_size = box_depth*box_height*box_width;
		

		#pragma omp parallel for
		for (int minibatch_index = 0; minibatch_index < minibatch_size; ++minibatch_index) {
			vector<float> sorted_vals(partition_size);
			vector<int> sorted_indices(partition_size);
			for (int ind_depth=0; ind_depth < depth; ind_depth += box_depth) {
				for (int ind_height=0; ind_height < height; ind_height += box_height) {
					for (int ind_width = 0; ind_width < width; ind_width += box_width) {
						// (i,j,k) is the corner of the box.
						// Iterate over all elements in the box, copying them in an array to be sorted.
						int box_1d_index = 0;
						for (int l = 0; l < box_depth; ++l) {
							for (int m = 0; m < box_height; ++m) {
								for (int n = 0; n < box_width; ++n) {
									float temp = -1e9f; // fixme: replace with most negative possible value.
									if (((l+ind_depth) < depth) && ((m+ind_height) < height) && ((n+ind_width) < width)) {
										temp = kmax_in(minibatch_index, l+ind_depth, m+ind_height, n+ind_width);
									} 
									//cout << "temp = " << temp << endl;
									sorted_vals[box_1d_index] = temp;
									sorted_indices[box_1d_index] = box_1d_index;
									++box_1d_index;
								}
							}
						}
						// Now do the sorting in descending order.
						// Indices are sorted but sorted_vals remains unmodified.
						// Sorted values are then: sorted_vals[sorted_indices[0]], sorted_vals[sorted_indices[1]], sorted_vals[sorted_indices[2]], ...
						sort(sorted_indices.begin(), sorted_indices.end(), [&sorted_vals](size_t i0, size_t i1) {
								return sorted_vals[i0] > sorted_vals[i1];
							});
						// Iterate through sorted values, keeping the k-max values and setting all others to 0.
						for (size_t q = k; q < sorted_indices.size(); ++q) {
							// todo: try leaky version of this. instead of 0, set to 0.01*neg_val.
							sorted_vals[sorted_indices[q]] = 0.0f;
						}

						// Now iterate through the box again, keeping only the largest k of the sorted values.
						box_1d_index = 0;
						for (int l = 0; l < box_depth; ++l) {
							for (int m = 0; m < box_height; ++m) {
								for (int n = 0; n < box_width; ++n) {
									//float temp = -1e9f; // fixme: replace with most negative possible value.

									float temp = sorted_vals[box_1d_index];
									//cout << "temp = " << temp << endl;
									if (temp > 0.0f) {
										kmax_out(minibatch_index, l+ind_depth, m+ind_height, n+ind_width) = temp;
										kmax_state(minibatch_index, l+ind_depth, m+ind_height, n+ind_width) = 1;
									}
									//if (((l+i) < depth) && ((m+j) < height) && ((n+k) < width)) {
											
									//}  


									++box_1d_index;
								}
							}
						}

					}
				}
			}
		}


	}

	void compute_reverse_3d_kmax(Matrix& kmax_in, const Matrix& kmax_out, const MatrixT<int>& kmax_state) {
		// kmax_in: (minibatch_size x depth x height x width) matrix containing the input values.
		check_dimensions(kmax_in, kmax_out);
		check_dimensions(kmax_in, kmax_state);
		set_value(kmax_in, 0.0f);
		#pragma omp parallel for
		for (int n = 0; n < kmax_in.size(); ++n) {
			int temp = kmax_state[n];
			if (temp == 1) {
				kmax_in[n] = kmax_out[n];
			} 
		}

	}

	void compute_forward_relu(const Matrix& in_vals, Matrix& out_vals, MatrixT<int>& out_indices) {
		check_dimensions(in_vals, out_vals);
		check_dimensions(in_vals, out_indices);
		set_value(out_vals, 0.0f);
		set_value(out_indices, 0);
		#pragma omp parallel for
		for (int n = 0; n < in_vals.size(); ++n) {
			double temp = in_vals[n];
			/*
			if (temp > 0.0f) {
				out_vals[n] = temp;
				out_indices[n] = 1;
			}
			*/
			if (temp > 0.0f) {
				out_vals[n] = temp;
				out_indices[n] = 1;
			} 
		}
	}

	void compute_forward_leaky_relu(const Matrix& in_vals, Matrix& out_vals, MatrixT<int>& out_indices) {
		check_dimensions(in_vals, out_vals);
		check_dimensions(in_vals, out_indices);
		set_value(out_vals, 0.0f);
		set_value(out_indices, 0);
		//const float leakiness = 0.01f; // Normal leaky
		const float leakiness = 0.33f; // Very leaky
		#pragma omp parallel for
		for (int n = 0; n < in_vals.size(); ++n) {
			double temp = in_vals[n];
			/*
			if (temp > 0.0f) {
				out_vals[n] = temp;
				out_indices[n] = 1;
			}
			*/
			if (temp > 0.0f) {
				out_vals[n] = temp;
				out_indices[n] = 1;
			} else {
				out_vals[n] = leakiness*temp; // leaky
				out_indices[n] = -1;
			}
		}
	}

	// For debug.
	void compute_forward_identity_activation(const Matrix& in_vals, Matrix& out_vals, MatrixT<int>& out_indices) {
		check_dimensions(in_vals, out_vals);
		check_dimensions(in_vals, out_indices);
		set_value(out_vals, 0.0f);
		set_value(out_indices, 0);
		#pragma omp parallel for
		for (int n = 0; n < in_vals.size(); ++n) {
			double temp = in_vals[n];
			/*
			if (temp > 0.0f) {
				out_vals[n] = temp;
				out_indices[n] = 1;
			}
			*/
			//if (temp > 0.0f) {
			if (true) {
				out_vals[n] = temp;
				out_indices[n] = 1;
			} 
		}
	}

	// For debug.
	void compute_reverse_identity_activation(Matrix& in_vals, const Matrix& out_vals, const MatrixT<int>& out_indices) {
		check_dimensions(in_vals, out_vals);
		check_dimensions(in_vals, out_indices);
		set_value(in_vals, 0.0f);
		#pragma omp parallel for
		for (int n = 0; n < in_vals.size(); ++n) {
			//int temp = out_indices[n];
			/*
			if (temp == 1) {
				in_vals[n] = out_vals[n];
			}
			*/
			//if (temp == 1) {
			if (true) {
				in_vals[n] = out_vals[n];
			} 
		}
	}

	void compute_reverse_relu(Matrix& in_vals, const Matrix& out_vals, const MatrixT<int>& out_indices) {
		check_dimensions(in_vals, out_vals);
		check_dimensions(in_vals, out_indices);
		set_value(in_vals, 0.0f);
		#pragma omp parallel for
		for (int n = 0; n < in_vals.size(); ++n) {
			int temp = out_indices[n];
			/*
			if (temp == 1) {
				in_vals[n] = out_vals[n];
			}
			*/
			if (temp == 1) {
				in_vals[n] = out_vals[n];
			} 
		}
	}

	void compute_reverse_leaky_relu(Matrix& in_vals, const Matrix& out_vals, const MatrixT<int>& out_indices) {
		check_dimensions(in_vals, out_vals);
		check_dimensions(in_vals, out_indices);
		set_value(in_vals, 0.0f);
		//const float leakiness = 0.01f; // Normal leaky
		const float leakiness = 0.33f; // Very leaky
		#pragma omp parallel for
		for (int n = 0; n < in_vals.size(); ++n) {
			int temp = out_indices[n];
			/*
			if (temp == 1) {
				in_vals[n] = out_vals[n];
			}
			*/
			if (temp == 1) {
				in_vals[n] = out_vals[n];
			} else if (temp == -1) {
				in_vals[n] = leakiness*out_vals[n];
			}
		}
	}
	






	//This computes grad_W = X_error*H^T.
	// To get mean gradient, you should do element-wise divide by the mini-batch size.
	void compute_weight_grad_sgd_minibatch(const Matrix& X_error, Matrix& W_grad, const Matrix& H) {
		check_matrix_factorization_dimensions(X_error, W_grad, H);
		mat_multiply_right_transpose(W_grad, X_error, H);
		//int minibatch_size = X_error.extent(1);
		//float scale_factor = 1.0f / static_cast<float>(minibatch_size);
		//scale(W_grad, W_grad, scale_factor);
	}



	void update_weights_from_gradient(Matrix& W, const Matrix& W_grad, float alpha) {
		check_dimensions(W, W_grad);
		#pragma omp parallel for
		for (int backing_index = 0; backing_index < W.size(); ++backing_index) {
			W[backing_index] -= alpha*W_grad[backing_index];
			// Optional: enable clipping to keep in a reasonable range. // fixme: maybe disable later.
			if (W[backing_index] > 1.0f) {
				W[backing_index] = 1.0f;
			} else if (W[backing_index] < -1.0f) {
				W[backing_index] = -1.0f;
			}
		}
		

	}


	void update_weights_from_gradient(Matrix& W, const Matrix& W_grad, float alpha, float lambda,
									  float sparsity_param, bool force_nonnegative) {
		check_dimensions(W, W_grad);

		#pragma omp parallel for
		for (int backing_index = 0; backing_index < W.size(); ++backing_index) {
			float w_i = W[backing_index];
			w_i = w_i - alpha*W_grad[backing_index] - alpha*lambda*w_i;
			// I add the L1 penalty using SGD-L1 (Clipping) method from:
			// "Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty"
			// by Yoshimasa Tsuruoka et al.
			// This is really slow so disable for now.
			
			// Disable for now because slow. (only for gcc)
			if (w_i > 0) {
				w_i = max(EPSILON_DENORMAL, w_i - alpha*sparsity_param);
			}
			else if (w_i < 0) {
				w_i = min(-EPSILON_DENORMAL, w_i + alpha*sparsity_param);
			}
			////// 
			if (force_nonnegative) {
				if (w_i < EPSILON_DENORMAL) {
					// Prevent denormalized values.
					w_i = EPSILON_DENORMAL;
				}
			}
			// Optional: enable clipping to keep in a reasonable range.
			if (w_i > 1.0f) {
				w_i = 1.0f;
			} else if (w_i < -1.0f) {
				w_i = -1.0f;
			}

			W[backing_index] = w_i;
		}
	}







	void update_weights_from_gradient_rmsprop_v2(Matrix& W, const Matrix& W_grad, Matrix& W_grad_sum_square, 
												 float alpha, float counter) {
		check_dimensions(W, W_grad);
		check_dimensions(W, W_grad_sum_square);
		for (int i = 0; i < W.size(); ++i) {
			// Update sum of squares of gradients.
			W_grad_sum_square[i] += W_grad[i]*W_grad[i];
			float rms_grad = sqrt(W_grad_sum_square[i]/counter);
			if (rms_grad > 0) {
				float w = W[i];
				w -= alpha*W_grad[i]/rms_grad;
				W[i] = w;
			}

		}

	}

	void update_weights_from_gradient_rmsprop_v3(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
												 float alpha, float rho) {
		check_dimensions(W, W_grad);
		check_dimensions(W, W_grad_mean_square);
		#pragma omp parallel for
		for (int i = 0; i < W.size(); ++i) {
			// Update sum of squares of gradients.
			W_grad_mean_square[i] = rho*W_grad_mean_square[i] + (1 -rho)*W_grad[i]*W_grad[i];
			float rms_grad = sqrt(W_grad_mean_square[i]);
			if (rms_grad > 0) {
				float w = W[i];
				w -= alpha*W_grad[i]/rms_grad;
				W[i] = w;
			}

		}

	}

	void update_weights_from_gradient_rmsprop_kernel_ball_1(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
												 float alpha, float rho) {
		check_dimensions(W, W_grad);
		check_dimensions(W, W_grad_mean_square);
		if (W.order() != 4) {
			cout << "oops. W is wrong order." << endl;
			cout << "Actual order is " << W.order();
			exit(1);
		}
		static std::mt19937 mersenne_twister_engine;
		//mersenne_twister_engine.seed(static_cast<unsigned long>(time(NULL)));
		float mean = 0.0;
		float std_deviation = 1.0; // 1-2
		std::normal_distribution<float> normal_dist(mean, std_deviation);
		vector<float> x1(4);
		vector<float> x2(4);

		// A[i] = normal_dist(mersenne_twister_engine);
		const int filter_count = W.extent(0);
		const int image_depth = W.extent(1);
		const int filter_height = W.extent(2);
		const int filter_width = W.extent(3);
		//#pragma omp parallel for collapse(4)
		for (int fc = 0; fc < filter_count; ++fc) {
			for (int id = 0; id < image_depth; ++id) {
				for (int fh = 0; fh < filter_height; ++fh) {
					for (int fw = 0; fw < filter_width; ++fw) {
						// Update sum of squares of gradients.
						W_grad_mean_square(fc,id,fh,fw) = rho*W_grad_mean_square(fc,id,fh,fw) + (1 -rho)*W_grad(fc,id,fh,fw)*W_grad(fc,id,fh,fw);
						float rms_grad = sqrt(W_grad_mean_square(fc,id,fh,fw));
						if (rms_grad > 0) {
							//float w = W(fc,id,fh,fw);
							//w -= alpha*W_grad(fc,id,fh,fw)/rms_grad;
							float update = alpha*W_grad(fc,id,fh,fw)/rms_grad;

							const int num_updates = 1;
							for (int n=0; n < num_updates; ++n) {
								// Now update a random nearby element.
								int rand_channel = static_cast<int>(normal_dist(mersenne_twister_engine));
								int new_fc = fc + rand_channel;
								if ((new_fc < 0) || (new_fc >= filter_count)) {
									new_fc = fc;
								} 
								//
								int rand_height = static_cast<int>(normal_dist(mersenne_twister_engine));
								int new_fh = fh + rand_height;
								if ((new_fh < 0) || (new_fh >= filter_height)) {
									new_fh = fh;
								} 
								//
								int rand_depth = static_cast<int>(normal_dist(mersenne_twister_engine));
								int new_id = id + rand_depth;
								if ((new_id < 0) || (new_id >= image_depth)) {
									new_id = id;
								} 
								//
								int rand_width = static_cast<int>(normal_dist(mersenne_twister_engine));
								int new_fw = fw + rand_width;
								if ((new_fw < 0) || (new_fw >= filter_width)) {
									new_fw = fw;
								} 
								
								x1[0] = fc;
								x1[1] = id;
								x1[2] = fh;
								x1[3] = fw;

								x2[0] = new_fc;
								x2[1] = new_id;
								x2[2] = new_fh;
								x2[3] = new_fw;
								float kernel_val = gaussian_kernel(x1, x2, std_deviation);
								//cout << "kernel_val = " << kernel_val << endl;
								
								//W(new_fc,id,new_fh,new_fw) = w - update;
								//W(fc,id,fh,fw) = w - update;
								update *= kernel_val;
								//W(new_fc,new_id,new_fh,new_fw) -= update;
								//W(new_fc,id,fh,fw) -= update; // works well
								//W(fc,new_id,fh,fw) -= update; // works well
								W(new_fc,new_id,fh,fw) -= update; // works well
								
							}


						}
					}
				}
			}
		}

	}

	void update_weights_from_gradient_rmsprop_momentum(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
													   Matrix W_momentum, float alpha, float rho, float momentum) {
		check_dimensions(W, W_grad);
		check_dimensions(W, W_grad_mean_square);
		#pragma omp parallel for
		for (int i = 0; i < W.size(); ++i) {
			// Update sum of squares of gradients.
			W_grad_mean_square[i] = rho*W_grad_mean_square[i] + (1 -rho)*W_grad[i]*W_grad[i];
			float rms_grad = sqrt(W_grad_mean_square[i]);
			if (rms_grad > 0) {
				float w = W[i];
				//float update_val = momentum*W_momentum[i] - (1-momentum)*alpha*W_grad[i]/rms_grad;
				float update_val = momentum*W_momentum[i] - alpha*W_grad[i]/rms_grad;
				W_momentum[i] = update_val;
				//w -= alpha*W_grad[i]/rms_grad;
				w += update_val;
				W[i] = w;
			}
		}
	}

	void update_weights_from_gradient_rmsprop_momentum_1d_kernel_ball(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
													   Matrix W_momentum, float alpha, float rho, float momentum) {
		check_dimensions(W, W_grad);
		check_dimensions(W, W_grad_mean_square);
		const int radius_width = 2;
		const int radius_height = 2;
		vector<float> width_kernel(1 + radius_width);
		vector<float> height_kernel(1 + radius_width);
		width_kernel[0] = 1.0f;
		width_kernel[1] = 0.8f;
		width_kernel[2] = 0.5f;

		height_kernel[0] = 1.0f;
		height_kernel[1] = 0.8f;
		height_kernel[2] = 0.5f;
		//#pragma omp parallel for
		//for (int i = 0; i < W.size(); ++i) {
		for (int r = 0; r < W.extent(0); ++r) {
			for (int c = 0; c < W.extent(1); ++c) {
				// Update sum of squares of gradients.
				W_grad_mean_square(r,c) = rho*W_grad_mean_square(r,c) + (1 -rho)*W_grad(r,c)*W_grad(r,c);
				float rms_grad = sqrt(W_grad_mean_square(r,c));
				if (rms_grad > 0) {
					float w = W(r,c);
					//float update_val = momentum*W_momentum[i] - (1-momentum)*alpha*W_grad[i]/rms_grad;
					float update_val = momentum*W_momentum(r,c) - alpha*W_grad(r,c)/rms_grad;
					W_momentum(r,c) = update_val;
					//w -= alpha*W_grad[i]/rms_grad;
					//w += update_val;
					// Now update a box/ball region about the current element .
					for (int m = -radius_width; m < radius_width; ++m) {
						for (int n = -radius_height; n < radius_height; ++n) {
						    int r_ball = r + m;
							int c_ball = c + n;
							if ((r_ball >= 0) && (r_ball < W.extent(0))) {
								if ((c_ball >= 0) && (c_ball < W.extent(1))) {
									// Current ball element is inside W.
									W(r_ball,c_ball) += update_val*width_kernel[abs(m)]*height_kernel[abs(n)];
								}
							}
						}
					}

					//W(r,c) = w;
				}
			}
		}
	}

	void update_weights_from_gradient_adagrad(Matrix& W, const Matrix& W_grad, Matrix& W_grad_sum_square, 
												 float alpha) {
		check_dimensions(W, W_grad);
		check_dimensions(W, W_grad_sum_square);
		for (int i = 0; i < W.size(); ++i) {
			// Update sum of squares of gradients.
			W_grad_sum_square[i] += W_grad[i]*W_grad[i];
			float rms_grad = sqrt(W_grad_sum_square[i]);
			if (rms_grad > 0) {
				float w = W[i];
				w -= alpha*W_grad[i]/rms_grad;
				W[i] = w;
			}

		}

	}


	float gaussian_kernel(const std::vector<float>& x1, const std::vector<float>& x2, float sigma) {
		float length_sq = 0.0f;
		for (size_t i = 0; i < x1.size(); ++i) {
			length_sq += (x1[i] - x2[i])*(x1[i] - x2[i]);
		}
		return exp(-length_sq/(2.0*sigma*sigma));
	}



	void do_backprop_update_sgd_minibatch(const Matrix& X_error, const Matrix& W, Matrix& H_error) {
		check_matrix_factorization_dimensions(X_error, W, H_error);
		// We compute H_error = W^T * X_error
		mat_multiply_left_transpose(H_error, W, X_error);
		//mat_multiply_left_transpose_naive(H_error, W, X_error);

	}

	void do_bias_update_sgd_minibatch(const Matrix& X_error, const Matrix& W, const Matrix& H, std::vector<float>& b,
		float alpha, float lambda, float sparsity_param, bool force_nonnegative)  {
		check_matrix_factorization_dimensions(X_error, W, H);
		int minibatch_size = X_error.extent(1);

		int row_W, col_X_error;
		float avg_grad_w, cur_error, b_i;
		float minibatch_size_f = static_cast<float>(minibatch_size);
		//#pragma omp parallel for private(row_W, col_W, avg_grad_w, w_i, col_X_error, cur_error)
		for (row_W = 0; row_W < W.extent(0); ++row_W) {
			// Note: maximum number of threads = b.size() = number of rows in W.

			// For each element b[row_W], loop over all columns in the mini-batch region of X_error, H to update.
			avg_grad_w = 0.0f; // Will contain the average gradient for W(row_W, col_W).
			b_i = b[row_W]; // Old value of W.
			for (col_X_error = 0; col_X_error < minibatch_size; ++col_X_error) {
				cur_error = X_error(row_W, col_X_error);
				avg_grad_w += cur_error;
			}
			avg_grad_w /= minibatch_size_f;
			b_i += alpha*avg_grad_w;
			if (b_i > 0) {
				b_i = max(EPSILON_DENORMAL, b_i);
			}
			else if (b_i < 0) {
				b_i = min(-EPSILON_DENORMAL, b_i);
			}
			if (force_nonnegative) {
				if (b_i < EPSILON_DENORMAL) {
					// Prevent denormalized values.
					b_i = EPSILON_DENORMAL;
				}
			}
			b[row_W] = b_i;
		}
	}


	// To get average gradient, scale by 1/minibatch_size.
	void compute_bias_grad_sgd_minibatch(const Matrix& X_error, Matrix& b_grad) {
		int minibatch_size = X_error.extent(1);
		float avg_grad_w, cur_error;
		//float minibatch_size_f = static_cast<float>(minibatch_size);
		for (int row_b = 0; row_b != b_grad.size(); ++row_b) {
			// Note: maximum number of threads = b.size() = number of rows in W.

			// For each element b[row_W], loop over all columns in the mini-batch region of X_error, H to update.
			avg_grad_w = 0.0f; // Will contain the gradient for W(row_W, col_W).
			
			for (int col_X_error = 0; col_X_error < minibatch_size; ++col_X_error) {
				cur_error = X_error(row_b, col_X_error);
				avg_grad_w += cur_error;
			}
			//avg_grad_w /= minibatch_size_f;
			b_grad[row_b] = avg_grad_w;
		}

	}












	void do_product_update_naive(Matrix& X, const Matrix& W, const Matrix& H, float update_weight) {
		check_matrix_factorization_dimensions(X, W, H);
		float new_val_X, val_X, approx_val_X;
		int row, col, cur_feature;
		int rows_X = X.extent(0);
		int cols_X = X.extent(1);
#pragma omp parallel for private(row, col, val_X, approx_val_X, cur_feature, new_val_X)
		for (col = 0; col < cols_X; col++) {
			for (row = 0; row < rows_X; row++) {
				val_X = X(row, col); // The old value of X
				// Compute estimate for val_X and the approximation error.
				approx_val_X = 0;
				for (cur_feature = 0; cur_feature < static_cast<int>(W.extent(1)); cur_feature++) {
					approx_val_X += (W(row, cur_feature)) * (H(cur_feature, col));
				}
				// Update X as convex combinatino of the old value + estimated value.
				new_val_X = (1 - update_weight)*val_X + update_weight*approx_val_X;
				X(row, col) = new_val_X;
			}
		}
	}

	void do_product_update(Matrix& X, const Matrix& W, const Matrix& H, float update_weight) {
		mat_multiply_blas(X, W, H, update_weight, 1.0f-update_weight);
	}


  void do_product_update(Matrix& X, const Matrix& W, const Matrix& H, const std::vector<float>& b) {
    check_matrix_factorization_dimensions(X, W, H);
    if (static_cast<int>(b.size()) != X.extent(0)) {
      cerr << "Supplied bias vector is wrong size." << endl;
      exit(1);
    }
    // X = W * H
    mat_multiply(X, W, H);
    //mat_multiply_naive(X, W, H);
    // Now add the bias component:
	
    int col, row;
    int rows_X = X.extent(0);
    int cols_X = X.extent(1);
    //#pragma omp parallel for private(row, col)
    for (col = 0; col < cols_X; col++) {
		for (row = 0; row < rows_X; row++) {
			X(row, col) += b[row];
		}
    }
    
  }

	void do_product_update_naive(Matrix& X, const Matrix& W, const Matrix& H, const std::vector<float>& b) {
		check_matrix_factorization_dimensions(X, W, H);
		if (static_cast<int>(b.size()) != X.extent(0)) {
			cerr << "Supplied bias vector is wrong size." << endl;
			exit(1);
		}
		float new_val_X;
		int row, col, cur_feature;
		int rows_X = X.extent(0);
		int cols_X = X.extent(1);
		//#pragma omp parallel for private(row, col, cur_feature, new_val_X)
		for (col = 0; col < cols_X; col++) {
			for (row = 0; row < rows_X; row++) {
				new_val_X = b[row];
				for (cur_feature = 0; cur_feature < static_cast<int>(W.extent(1)); cur_feature++) {
					new_val_X += (W(row, cur_feature)) * (H(cur_feature, col));
				}
				X(row, col) = new_val_X;
			}
		}
	}


	float compute_reconstruction_error(const Matrix& X, const Matrix& W, const Matrix& H) {
		check_matrix_factorization_dimensions(X, W, H);
		float sum_errors = 0.0f;
		int rows = X.extent(0);
		int cols = X.extent(1);
		int cur_feature;
		float approx_val_X, cur_error;
		for (int col = 0; col < cols; col++) {
			for (int row = 0; row < rows; row++) {
				// Compute estimate for val_X and the approximation error.
				approx_val_X = 0.0f;
				for (cur_feature = 0; cur_feature < static_cast<int>(W.extent(1)); cur_feature++) {
					approx_val_X += (W(row, cur_feature)) * (H(cur_feature, col));
				}
				cur_error = X(row, col) - approx_val_X;
				sum_errors += cur_error*cur_error;
			}
		}
		float rmse = sqrt(sum_errors / (rows*cols));
		return rmse;
	}

	float compute_rmse(const Matrix& X) {
		float sum_errors = 0.0f;
		int rows = X.extent(0);
		int cols = X.extent(1);
		float cur_error;
		for (int col = 0; col < cols; col++) {
			for (int row = 0; row < rows; row++) {
				cur_error = X(row, col);
				sum_errors += cur_error*cur_error;
			}
		}
		float rmse = sqrt(sum_errors / (rows*cols));
		return rmse;
	
	}

	/*
	* Given two matrices, compute and return the RMSE of their element-wise differences.
	*/
	float compute_rmse(const Matrix& A, const Matrix& B)  {
		check_dimensions(B, B);
		float sum_errors = 0.0f;
		float cur_error;
		for (int ind = 0;  ind < A.size(); ++ind) {
				cur_error = A[ind] - B[ind];
				sum_errors += cur_error*cur_error;
		}
		float rmse = sqrt(sum_errors / A.size());
		return rmse;
	}

	



	void get_sub_matrix(Matrix& A, const Matrix &B, int start_col) {
		const int rowsA = A.extent(0);
		const int rowsB = B.extent(0);
		if (rowsA != rowsB) {
			std::cerr << "Matrix A and B do not have the same number of rows! Exiting.";
			exit(1);
		}
		const int stop_col = start_col + A.extent(1);
		if (start_col < 0) {
			std::cerr << "Invalid start_col < 0.";
			exit(1);
		}
		else if (stop_col > B.extent(1)) {
			std::cerr << "Invalid parameters in getSubMatrix()." << std::endl;
			std::cerr << "stop_col = " << stop_col << std::endl;
			std::cerr << "B.extent(1) = " << B.extent(1) << std::endl;
			exit(1);
		}
		for (int r = 0; r != A.extent(0); ++r) {
		  for (int c = start_col; c != stop_col; ++c) {
			
				A(r, c - start_col) = B(r, c);
			}
		}
	}

	// A is 3D.
	// B is 4D
	void get_sample(Matrix& A, const Matrix& B, int sample_index) {
		const int N = B.extent(0);
		bool bad_parameters = false;
		if (sample_index >= N) {
			bad_parameters = true;
		}
		if (A.extent(0) != B.extent(1)) {
			bad_parameters = true;
		}
		if (A.extent(1) != B.extent(2)) {
			bad_parameters = true;
		}
		if (A.extent(2) != B.extent(3)) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			std::cerr << "Invalid parameters in get_sample().";
			exit(1);
		}
		for (int i = 0; i != A.extent(0); ++i) {
			for (int j = 0; j != A.extent(1); ++j) {
				for (int k = 0; k != A.extent(2); ++k) {
					A(i,j,k) = B(sample_index, i, j, k);
				}
			}
		}
	}

	// A is 3D.
	// B is 4D.
	void return_sample(const Matrix& A, Matrix& B, int sample_index) {
		const int N = B.extent(0);
		bool bad_parameters = false;
		if (sample_index >= N) {
			bad_parameters = true;
		}
		if (A.extent(0) != B.extent(1)) {
			bad_parameters = true;
		}
		if (A.extent(1) != B.extent(2)) {
			bad_parameters = true;
		}
		if (A.extent(2) != B.extent(3)) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			std::cerr << "Invalid parameters in get_sample().";
			exit(1);
		}
		for (int i = 0; i != A.extent(0); ++i) {
			for (int j = 0; j != A.extent(1); ++j) {
				for (int k = 0; k != A.extent(2); ++k) {
					B(sample_index, i, j, k) = A(i,j,k);
				}
			}
		}
	}

	void return_sub_matrix(const Matrix& A, Matrix &B, int start_col) {
		const int rowsA = A.extent(0);
		const int rowsB = B.extent(0);
		if (rowsA != rowsB) {
			std::cerr << "Matrix A and B do not have the same number of rows! Exiting." << endl;
			exit(1);
		}
		const int stop_col = start_col + A.extent(1);
		if (start_col < 0) {
			std::cerr << "Invalid start_col < 0.";
			exit(1);
		}
		else if (stop_col > B.extent(1)) {
			std::cerr << "Invalid parameters in returnSubMatrix()." << endl;
			exit(1);
		}
		for (int r = 0; r != A.extent(0); ++r) {
		  for (int c = start_col; c != stop_col; ++c) {
			
				B(r, c) = A(r, c - start_col);
			}
		}
	}

	void assert_almost_equal(float a, float b, float tolerance) {
		float diff = abs(a - b);
		/*
		float numer = abs(a - b);
		float denom = abs(a + b);
		float score = 0.0f;
		if (denom > 0.0f) {
			score = numer/denom;
		}
		cout << "numer = " << numer << endl;
		cout << "denom = " << denom << endl;
		*/
		// Note: score is close to 0 if the difference in magnitude between a and b is small compared
		// to their individual magnitudes.
		if (diff > tolerance) {
			cerr << "Tolerance exceeded!:" << endl;
			cerr << "a = " << a << endl;
			cerr << "b = " << b << endl;
			cerr << "difference magnitude = " << diff << endl;
			exit(1);
		}
	}

	void assert_almost_equal(const Matrix& A, const Matrix& B, float tolerance) {
		check_dimensions(A, B);
		float score = relative_error(A, B);
		if (score > tolerance) {
			cerr << "Tolerance exceeded!:" << endl;
			exit(1);
		}
		/*
		for (int r = 0; r != A.extent(0); ++r) {
			for (int c = 0; c != A.extent(1); ++c) {
				if (abs(A(r, c) - B(r, c)) > tolerance) {
					cerr << "Tolerance exceeded!:" << endl;
					cerr << "a = " << A(r, c) << endl;
					cerr << "b = " << B(r, c) << endl;
					cerr << "Difference magnitude = " << abs(A(r, c) - B(r, c)) << endl;
					cerr << "at row = " << r << endl;
					cerr << "at col = " << c << endl;
					exit(1);
				}
			}
		}
		*/
	}



	void print_stats(const Matrix& A, std::string name) {
		// Compute mean:
		float N = static_cast<float>(A.size());
		float mean = 0.0f;
		//float abs_mean = 0.0f;
		for (int i = 0; i != A.size(); ++i) {
			mean += A[i];
			//abs_mean += abs(A.m_values[i]);
		}
		mean /= N;
		//abs_mean /= N;
		// Compute sample standard deviation:
		float std_dev = 0.0f;
		for (int i = 0; i != A.size(); ++i) {
			std_dev += (A[i] - mean)*(A[i] - mean);
		}
		std_dev /= N;
		std_dev = sqrt(std_dev);
		
		cout << "-------------------------------------------" << endl;
		cout << "Stats for: " << name << endl;
		cout << "mean = " << mean << endl;
		//cout << "mean of magnitude of elements = " << abs_mean << endl;
		cout << "standard deviation = " << std_dev << endl;
		cout << "-------------------------------------------" << endl;
	}









	void convolve_2d_filter_with_bias_minibatch(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1) {
		// for i in [0,...,minibatch_size).
		// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
		const int P = W.extent(1);
		const int Q = W.extent(2);
		const int R = W.extent(0);
		bool bad_parameters = false;
		if (Z2.extent(0) != A1.extent(0)) {
			cerr << "Z2.extent(0) != A1.extent(0)" << endl;
			cerr << "Z2.extent(0) = " << Z2.extent(0) << endl;
			cerr << "A1.extent(0) = " << A1.extent(0) << endl;
			
			bad_parameters = true;
		}
		if (Z2.extent(2) != A1.extent(1)) {
			cerr << "Z2.extent(2) != A1.extent(1)" << endl;
			cerr << "Z2.extent(2) = " << Z2.extent(2) << endl;
			cerr << "A1.extent(1) = " << A1.extent(1) << endl;
			bad_parameters = true;
		}
		if (Z2.extent(3) != A1.extent(2)) {
			bad_parameters = true;
		}
		if (Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (bias.order() != 1) {
			bad_parameters = true;
		}
		if (bias.size() != R) {
			bad_parameters = true;
		}

		if (bad_parameters) {
			cerr << "convolve_2d_filter_with_bias_minibatch(): Bad parameter sizes." << endl;
			exit(1);
		}

		const int M = A1.extent(1);
		const int N = A1.extent(2);
		const int minibatch_size = A1.extent(0);
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					// Note: for each work item (cur_batch, r,c), the work below can be performed in parallel. todo: parallelize.
					for (int k = 0; k < R; ++k) {
						float sum = 0.0f;
						// Now compute convultion of r'th filter for the pixel X(r,c).
						for (int i = 0; i < P; ++i) {
		                    for (int j = 0; j < Q; ++j) {
								if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
									sum += A1(cur_batch, r - i, c - j)*W(k, i,j);
								}
							}
						}
						Z2(cur_batch, k, r,c) = sum + bias[k];
					}
				}
			}
		}
	}		


	void convolve_3d_filter_with_bias_minibatch(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1) {
		// for i in [0,...,minibatch_size).
		// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.

		const int R = W.extent(0);
		const int D = W.extent(1);
		const int P = W.extent(2);
		const int Q = W.extent(3);

		bool bad_parameters = false;
		// minibatch_size
		if (Z2.extent(0) != A1.extent(0)) {
			cerr << "Z2.extent(0) != A1.extent(0)" << endl;
			cerr << "Z2.extent(0) = " << Z2.extent(0) << endl;
			cerr << "A1.extent(0) = " << A1.extent(0) << endl;
			
			bad_parameters = true;
		}
		if (Z2.extent(2) != A1.extent(2)) {
			cerr << "Z2.extent(2) != A1.extent(2)" << endl;
			cerr << "Z2.extent(2) = " << Z2.extent(2) << endl;
			cerr << "A1.extent(2) = " << A1.extent(2) << endl;
			bad_parameters = true;
		}
		if (Z2.extent(3) != A1.extent(3)) {
			bad_parameters = true;
		}
		if (Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (bias.order() != 1) {
			bad_parameters = true;
		}
		if (bias.size() != R) {
			bad_parameters = true;
		}
		// depth
		if (D != A1.extent(1) ) {
			cerr << "Inconsistent depth parameters." << endl;
			bad_parameters = true;
		}

		if (bad_parameters) {
			cerr << "convolve_2d_filter_with_bias_minibatch(): Bad parameter sizes." << endl;
			exit(1);
		}
		
		const int M = A1.extent(2);
		const int N = A1.extent(3);
		const int minibatch_size = A1.extent(0);
		//#pragma omp parallel for
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					// Note: for each work item (cur_batch, r,c), the work below can be performed in parallel. todo: parallelize.
					for (int k = 0; k < R; ++k) {
						float sum = 0.0f;
						// Now compute convultion of r'th filter for the pixel X(r,c).
						for (int d = 0; d < D; ++d) {
							for (int i = 0; i < P; ++i) {
								for (int j = 0; j < Q; ++j) {
									if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
										sum += A1(cur_batch, d, r - i, c - j)*W(k, d, i,j);
									}
								}
							}
						}
						Z2(cur_batch, k, r,c) = sum + bias[k];
					}
				}
			}
		}
	}		


	void convolve_3d_filter_with_bias_minibatch_optimized(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1, Matrix& temp_Z2, Matrix& temp_A1, Matrix& temp_W) {
		// for i in [0,...,minibatch_size).
		// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.

		const int R = W.extent(0);
		const int D = W.extent(1);
		const int P = W.extent(2);
		const int Q = W.extent(3);

		bool bad_parameters = false;
		// minibatch_size
		if (Z2.extent(0) != A1.extent(0)) {
			cerr << "Z2.extent(0) != A1.extent(0)" << endl;
			cerr << "Z2.extent(0) = " << Z2.extent(0) << endl;
			cerr << "A1.extent(0) = " << A1.extent(0) << endl;
			
			bad_parameters = true;
		}
		if (Z2.extent(2) != A1.extent(2)) {
			cerr << "Z2.extent(2) != A1.extent(2)" << endl;
			cerr << "Z2.extent(2) = " << Z2.extent(2) << endl;
			cerr << "A1.extent(2) = " << A1.extent(2) << endl;
			bad_parameters = true;
		}
		if (Z2.extent(3) != A1.extent(3)) {
			bad_parameters = true;
		}
		if (Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (bias.order() != 1) {
			bad_parameters = true;
		}
		if (bias.size() != R) {
			bad_parameters = true;
		}
		// depth
		if (D != A1.extent(1) ) {
			cerr << "Inconsistent depth parameters:" << endl;
			cerr << "D from W.extent(1) = " << W.extent(1) << endl;
			cerr << "but D from A1.extent(1) = " << A1.extent(1) << endl;
			bad_parameters = true;
		}

		
		const int M = A1.extent(2);
		const int N = A1.extent(3);
		const int minibatch_size = A1.extent(0);

		if (temp_Z2.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_Z2.extent(1) != R ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(1) != (D*P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(0) != (D*P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(1) != R) {
			bad_parameters = true;
		}
		

		if (bad_parameters) {
			cerr << "convolve_3d_filter_with_bias_minibatch_optimized: Bad parameter sizes." << endl;
			exit(1);
		}

		// Copy data from W and bias into temp_W. 
		//
		// Format for temp_W:
		// Each column of temp_W will contain the unrolled version of one of the R convolutional filters, plus the
		// corresponding bias value at the bottom (i.e., last row).
		//#pragma omp parallel for
		#pragma omp parallel for collapse(3)
		for (int k = 0; k < R; ++k) {
			// For each of the R convolutional filters:
			for (int d = 0; d < D; ++d) {
				for (int i = 0; i < P; ++i) {
					for (int j = 0; j < Q; ++j) {
						temp_W(d*P*Q + i*Q + j, k) = W(k, d, i,j);
					}
				}
			}
		}

		//cout << "D = " << D << endl;

		// Write the bias value in the last row;
		for (int k = 0; k < R; ++k) {
			//cout << "D*P*Q = " << D*P*Q << endl;
			//cout << "k = " << k << endl;
			//cout << "bias(k) = " << bias(k) << endl;
			temp_W(D*P*Q, k) = bias(k);
		}

		//cout << "temp_W:" << endl << temp_W << endl;
		//cout << "W:" << endl << W << endl;		

		//int row_ind = 0; // Each row corresponds to one pixel in the input image over all output images in mini-batch.
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			// Copy data from A1 into temp_A1 for current minibatch.
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					
						int row_ind = c + r*N + cur_batch*M*N;
						//int row_ind = cur_batch*M*N*D + r*N*D + c*D + d;
						int col_ind = 0; 
						for (int d = 0; d < D; ++d) {
							for (int i = 0; i < P; ++i) {
								for (int j = 0; j < Q; ++j) {
									//if (row_ind >= temp_A1.extent(0)) {
									//	cerr << "oops row" << endl;
									//	exit(1);
									//}
									if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
										temp_A1(row_ind, col_ind) = A1(cur_batch, d, r - i, c - j);
									} else {
										// Write 0.
										temp_A1(row_ind, col_ind) = 0.0f;
									}
									++col_ind;
								}
							}
						}
						// Write a 1 for the bias term.
						temp_A1(row_ind, col_ind) = 1.0f;
						//if (col_ind >= temp_A1.extent(1)) {
						//	cerr << "oops col" << endl;
						//	exit(1);
						//}
					
				}
			}
		}
		//cout << "temp_A1: " << endl << temp_A1 << endl;
		//cout << "A1: " << endl << A1 << endl;
		//cout << "temp_A1 ready" << endl;

		// Now compute temp_Z2 = temp_A1 * temp_W using BLAS sgemm:
		mat_multiply(temp_Z2, temp_A1, temp_W);

		//cout << "Did mat multiply." << endl;

		// Copy result from temp_Z2 into Z2.
		// Z2 has dimensions minibatch_size x R x M x N.
		//set_value(Z2, 0.0f);
		//#pragma omp parallel for
		#pragma omp parallel for collapse(2)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int k = 0; k < R; ++k) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						//for (int d = 0; d < D; ++d) {
							Z2(cur_batch, k, r, c) = temp_Z2(c + r*N + cur_batch*M*N, k);
							//}
					}
				}
			}	
		}


		/*
		//#pragma omp parallel for
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					// Note: for each work item (cur_batch, r,c), the work below can be performed in parallel. todo: parallelize.
					for (int k = 0; k < R; ++k) {
						float sum = 0.0f;
						// Now compute convultion of r'th filter for the pixel X(r,c).
						for (int d = 0; d < D; ++d) {
							for (int i = 0; i < P; ++i) {
								for (int j = 0; j < Q; ++j) {
									if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
										sum += A1(cur_batch, d, r - i, c - j)*W(k, d, i,j);
									}
								}
							}
						}
						Z2(cur_batch, k, r,c) = sum + bias[k];
					}
				}
			}
		}
		*/
	}		


	void convolve_2d_filter_with_bias_minibatch_optimized(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1, Matrix& temp_Z2, Matrix& temp_A1, Matrix& temp_W)  {
		// todo: consider performing all convolutions for the mini-batch in a single big matrix multiplication.

		// for i in [0,...,minibatch_size).
		// Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
		const int P = W.extent(1);
		const int Q = W.extent(2);
		const int R = W.extent(0);
		bool bad_parameters = false;
		if (Z2.extent(0) != A1.extent(0)) {
			cerr << "Z2.extent(0) != A1.extent(0)" << endl;
			cerr << "Z2.extent(0) = " << Z2.extent(0) << endl;
			cerr << "A1.extent(0) = " << A1.extent(0) << endl;
			
			bad_parameters = true;
		}
		if (Z2.extent(2) != A1.extent(1)) {
			cerr << "Z2.extent(2) != A1.extent(1)" << endl;
			cerr << "Z2.extent(2) = " << Z2.extent(2) << endl;
			cerr << "A1.extent(1) = " << A1.extent(1) << endl;
			bad_parameters = true;
		}
		if (Z2.extent(3) != A1.extent(2)) {
			bad_parameters = true;
		}
		if (Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (bias.order() != 1) {
			bad_parameters = true;
		}
		if (bias.size() != R) {
			bad_parameters = true;
		}
		const int M = A1.extent(1);
		const int N = A1.extent(2);
		const int minibatch_size = A1.extent(0);

		if (temp_Z2.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_Z2.extent(1) != R ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(1) != (P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(0) != (P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(1) != R) {
			bad_parameters = true;
		}
		

		if (bad_parameters) {
			cerr << "convolve_2d_filter_with_bias_minibatch_v2(): Bad parameter sizes." << endl;
			exit(1);
		}

		// Copy data from W and bias into temp_W. 
		//
		// Format for temp_W:
		// Each column of temp_W will contain the unrolled version of one of the R convolutional filters, plus the
		// corresponding bias value at the bottom (i.e., last row).
		//#pragma omp parallel for
		#pragma omp parallel for collapse(3)
		for (int k = 0; k < R; ++k) {
			// For each of the R convolutional filters:
			for (int i = 0; i < P; ++i) {
				for (int j = 0; j < Q; ++j) {
					temp_W(j + i*Q, k) = W(k, i,j);
				}
			}
		}

		// Write the bias value in the last row;
		for (int k = 0; k < R; ++k) {
			temp_W(P*Q, k) = bias(k);
		}

		//int row_ind = 0; // Each row corresponds to one pixel in the input image over all output images in mini-batch.
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			// Copy data from A1 into temp_A1 for current minibatch.
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					int row_ind = c + r*N + cur_batch*M*N;
					int col_ind = 0; 
					for (int i = 0; i < P; ++i) {
						for (int j = 0; j < Q; ++j) {
							if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
								temp_A1(row_ind, col_ind) = A1(cur_batch, r - i, c - j);
							} else {
								// Write 0.
								temp_A1(row_ind, col_ind) = 0.0f;
							}
							++col_ind;
						}
					}
					// Write a 1 for the bias term.
					temp_A1(row_ind, col_ind) = 1.0f;
				}
			}
		}
		//cout << "temp_A1 ready" << endl;

		// Now compute temp_Z2 = temp_A1 * temp_W using BLAS sgemm:
		mat_multiply(temp_Z2, temp_A1, temp_W);

		//cout << "Did mat multiply." << endl;

		// Copy result from temp_Z2 into Z2.
		// Z2 has dimensions minibatch_size x R x M x N.
		
		//#pragma omp parallel for
		#pragma omp parallel for collapse(2)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int k = 0; k < R; ++k) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						Z2(cur_batch, k, r, c) = temp_Z2(c + r*N + cur_batch*M*N, k);
					}
				}
			}	
		}
		
		

		/*
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					// Note: for each work item (cur_batch, r,c), the work below can be performed in parallel. todo: parallelize.
					for (int k = 0; k < R; ++k) {
						float sum = 0.0f;
						// Now compute convultion of r'th filter for the pixel X(r,c).
						for (int i = 0; i < P; ++i) {
		                    for (int j = 0; j < Q; ++j) {
								if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
									sum += A1(cur_batch, r - i, c - j)*W(k, i,j);
								}
							}
						}
						Z2(cur_batch, k, r,c) = sum + bias[k];
					}
				}
			}
		}
		*/
	}		



	void compute_convolutive_deltas_minibatch(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2) {
		// model: Z2(i'th minibatch) = W (convolve with) A1(i'th minibatch)
		const int P = W.extent(1);
		const int Q = W.extent(2);
		const int R = W.extent(0);
		bool bad_parameters = false;
		if (deltas_A1.extent(0) != deltas_Z2.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(1) != deltas_Z2.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(2) != deltas_Z2.extent(3)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "compute_convolutive_deltas_minibatch(): wrong dimensions." << endl;
			exit(1);
		}
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		const int minibatch_size = deltas_A1.extent(0);

#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					float sum = 0.0f;
					// Note: for each work item (r,c), the work below can be performed in parallel. todo: parallelize.
					for (int k = 0; k < R; ++k) {
						// Now compute convultion of r'th filter for the pixel deltas_Z2(r,c).
						for (int i = 0; i < P; ++i) {
							for (int j = 0; j < Q; ++j) {
								if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
									sum += deltas_Z2(cur_batch, k, r + i, c + j)*W(k, i,j);
								}
							}
						}
					}
					deltas_A1(cur_batch, r,c) = sum;
				}
			}
		}
	}

	void compute_3d_convolutive_deltas_minibatch(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2) {
		// model: Z2(i'th minibatch) = W (convolve with) A1(i'th minibatch)
		const int R = W.extent(0);
		const int D = W.extent(1);
		const int P = W.extent(2);
		const int Q = W.extent(3);
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		const int minibatch_size = deltas_A1.extent(0);

		bool bad_parameters = false;
		if (minibatch_size != deltas_Z2.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(2) != deltas_Z2.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(3) != deltas_Z2.extent(3)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (W.extent(1) != deltas_A1.extent(1)) {
			bad_parameters = true;
		}

		if (bad_parameters) {
			cerr << "compute_convolutive_deltas_minibatch(): wrong dimensions." << endl;
			exit(1);
		}
		
#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int d = 0; d < D; ++d) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						float sum = 0.0f;
						// Note: for each work item (r,c), the work below can be performed in parallel. todo: parallelize.
						for (int k = 0; k < R; ++k) {
							for (int i = 0; i < P; ++i) {
								for (int j = 0; j < Q; ++j) {
									if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
										sum += deltas_Z2(cur_batch, k, r + i, c + j)*W(k, d, i,j);
									}
								}
							}
						}
						deltas_A1(cur_batch, d, r,c) = sum;
					}
				}
			}
		}
	}


	void compute_3d_convolutive_deltas_minibatch_optimized(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2, 
														   Matrix& temp_deltas_Z2, Matrix& temp_deltas_A1, Matrix& temp_W)  {
		// model: Z2(i'th minibatch) = W (convolve with) A1(i'th minibatch)
		const int R = W.extent(0);
		const int D = W.extent(1);
		const int P = W.extent(2);
		const int Q = W.extent(3);
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		const int minibatch_size = deltas_A1.extent(0);

		bool bad_parameters = false;
		if (minibatch_size != deltas_Z2.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(2) != deltas_Z2.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(3) != deltas_Z2.extent(3)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (W.extent(1) != deltas_A1.extent(1)) {
			bad_parameters = true;
		}


		if (temp_deltas_Z2.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_deltas_Z2.extent(1) != R ) {
			bad_parameters = true;
		}
		if (temp_deltas_A1.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_deltas_A1.extent(1) != (D*P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(0) != (D*P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(1) != R) {
			bad_parameters = true;
		}

		if (bad_parameters) {
			cerr << "compute_convolutive_deltas_minibatch(): wrong dimensions." << endl;
			exit(1);
		}

		#pragma omp parallel for collapse(3)
		for (int k = 0; k < R; ++k) {
			// For each of the R convolutional filters:
			for (int d = 0; d < D; ++d) {
				for (int i = 0; i < P; ++i) {
					for (int j = 0; j < Q; ++j) {
						temp_W(d*P*Q + i*Q + j, k) = W(k, d, i,j);
					}
				}
			}
		}

		// Note: bias is not needed for computing deltas, so we can ignore the bottom row of W.
		// and can also ignore the right-most column of temp_deltas_A1.

		#pragma omp parallel for collapse(2)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int k = 0; k < R; ++k) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						temp_deltas_Z2(c + r*N + cur_batch*M*N, k) = deltas_Z2(cur_batch, k, r, c);
					}
				}
			}	
		}

		// Compute temp_deltas_A1 = temp_deltas_Z2 * temp_W^T.
		mat_multiply_right_transpose(temp_deltas_A1, temp_deltas_Z2, temp_W);

		set_value(deltas_A1, 0.0f);

		// Update deltas_A1 from temp_deltas_A1.
		#pragma omp parallel for
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					int row_ind = c + r*N + cur_batch*M*N;
					int col_ind = 0; 
					for (int d = 0; d < D; ++d) {
						for (int i = 0; i < P; ++i) {
							for (int j = 0; j < Q; ++j) {
								if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
									deltas_A1(cur_batch, d, r - i, c - j) += temp_deltas_A1(row_ind, col_ind);
								} 
								++col_ind;
							}
						}
					}
				}
			}
		}



		/*
#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int d = 0; d < D; ++d) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						float sum = 0.0f;
						// Note: for each work item (r,c), the work below can be performed in parallel. todo: parallelize.
						for (int k = 0; k < R; ++k) {
							for (int i = 0; i < P; ++i) {
								for (int j = 0; j < Q; ++j) {
									if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
										sum += deltas_Z2(cur_batch, k, r + i, c + j)*W(k, d, i,j);
									}
								}
							}
						}
						deltas_A1(cur_batch, d, r,c) = sum;
					}
				}
			}
		}
		*/
	}

	void compute_convolutive_deltas_minibatch_optimized(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2,
														   Matrix& temp_deltas_Z2, Matrix& temp_deltas_A1, Matrix& temp_W) {
		// model: Z2(i'th minibatch) = W (convolve with) A1(i'th minibatch)
		const int P = W.extent(1);
		const int Q = W.extent(2);
		const int R = W.extent(0);
		bool bad_parameters = false;
		if (deltas_A1.extent(0) != deltas_Z2.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(1) != deltas_Z2.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_A1.extent(2) != deltas_Z2.extent(3)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		const int minibatch_size = deltas_A1.extent(0);

		if (temp_deltas_Z2.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_deltas_Z2.extent(1) != R ) {
			bad_parameters = true;
		}
		if (temp_deltas_A1.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_deltas_A1.extent(1) != (P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(0) != (P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_W.extent(1) != R) {
			bad_parameters = true;
		}

		if (bad_parameters) {
			cerr << "compute_convolutive_deltas_minibatch_optimized(): wrong dimensions." << endl;
			exit(1);
		}

		// Copy data from W and bias into temp_W. 
		//
		// Format for temp_W:
		// Each column of temp_W will contain the unrolled version of one of the R convolutional filters, plus the
		// corresponding bias value at the bottom (i.e., last row).

		#pragma omp parallel for collapse(3)
		for (int k = 0; k < R; ++k) {
			// For each of the R convolutional filters:
			for (int i = 0; i < P; ++i) {
				for (int j = 0; j < Q; ++j) {
					temp_W(j + i*Q, k) = W(k, i,j);
				}
			}
		}

		// Note: bias is not needed for computing deltas, so we can ignore the bottom row of W.
		// and can also ignore the right-most column of temp_deltas_A1.

		#pragma omp parallel for collapse(2)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int k = 0; k < R; ++k) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						temp_deltas_Z2(c + r*N + cur_batch*M*N, k) = deltas_Z2(cur_batch, k, r, c);
					}
				}
			}	
		}

		// Compute temp_deltas_A1 = temp_deltas_Z2 * temp_W^T.
		mat_multiply_right_transpose(temp_deltas_A1, temp_deltas_Z2, temp_W);

		set_value(deltas_A1, 0.0f);

		// Update deltas_A1 from temp_deltas_A1.
		#pragma omp parallel for
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					int row_ind = c + r*N + cur_batch*M*N;
					int col_ind = 0; 
					for (int i = 0; i < P; ++i) {
						for (int j = 0; j < Q; ++j) {
							if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
								deltas_A1(cur_batch, r - i, c - j) += temp_deltas_A1(row_ind, col_ind);
							} 
							++col_ind;
						}
					}
				}
			}
		}



		/*
#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					float sum = 0.0f;
					// Note: for each work item (r,c), the work below can be performed in parallel. todo: parallelize.
					for (int k = 0; k < R; ++k) {
						// Now compute convultion of r'th filter for the pixel deltas_Z2(r,c).
						for (int i = 0; i < P; ++i) {
							for (int j = 0; j < Q; ++j) {
								if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
									//sum += deltas_Z2(cur_batch, r + i, c + j, k)*W(i,j,k);
									//sum += deltas_Z2(cur_batch, r + i, c + j, k)*W(k, i,j);
									sum += deltas_Z2(cur_batch, k, r + i, c + j)*W(k, i,j);
								}
							}
						}
					}
					deltas_A1(cur_batch, r,c) = sum;
				}
			}
		}
		*/
	}


	
	void compute_weight_grad_convolutive_minibatch(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1) {
		// convolutive model:  Z2 = W (convolve with) A1
		const int P = grad_W.extent(1);
		const int Q = grad_W.extent(2);
		const int R = grad_W.extent(0);
		bool bad_parameters = false;
		if (deltas_Z2.extent(0) != A1.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(2) != A1.extent(1)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(3) != A1.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "compute_weight_grad_convolutive_minibatch(): wrong dimensions." << endl;
			exit(1);
		}
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		//#pragma omp parallel for private(r,c,k,i,j,sum)
		const int minibatch_size = A1.extent(0);
#pragma omp parallel for collapse(3)
		for (int r = 0; r < P; ++r) {
			for (int c = 0; c < Q; ++c) {
				// Note: for each work item (r,c), the work below can be performed in parallel. todo: parallelize.
				for (int k = 0; k < R; ++k) {
					float sum = 0.0f;
					for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
						for (int i = 0; i < M; ++i) {
							for (int j = 0; j < N; ++j) {
								if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
									//sum += deltas_Z2(cur_batch, r + i, c + j, k)*A1(cur_batch, i,j);
									sum += deltas_Z2(cur_batch, k, r + i, c + j)*A1(cur_batch, i,j);
								}
							}
						}
					}
					//grad_W(r,c,k) = sum;
					grad_W(k, r,c) = sum;
				}
			}
		}
	}

	void compute_3d_weight_grad_convolutive_minibatch(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1)  {
		// convolutive model:  Z2 = W (convolve with) A1
		const int R = grad_W.extent(0);
		const int D = grad_W.extent(1);
		const int P = grad_W.extent(2);
		const int Q = grad_W.extent(3);

		bool bad_parameters = false;
		if (deltas_Z2.extent(0) != A1.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(2) != A1.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(3) != A1.extent(3)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (D != A1.extent(1)) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "compute_weight_grad_convolutive_minibatch(): wrong dimensions." << endl;
			exit(1);
		}
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		//#pragma omp parallel for private(r,c,k,i,j,sum)
		const int minibatch_size = A1.extent(0);
#pragma omp parallel for collapse(3)
		for (int r = 0; r < P; ++r) {
			for (int c = 0; c < Q; ++c) {
				for (int k = 0; k < R; ++k) {
					for (int d = 0; d < D; ++d) {
						float sum = 0.0f;
						for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
							for (int i = 0; i < M; ++i) {
								for (int j = 0; j < N; ++j) {
									if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
										sum += deltas_Z2(cur_batch, k, r + i, c + j)*A1(cur_batch, d, i,j);
									}
								}
							}
						}
						grad_W(k, d, r,c) = sum;
					}
				}
			}
		}
	}


	void compute_weight_grad_convolutive_minibatch_optimized(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1,
															 Matrix& temp_deltas_Z2, Matrix& temp_A1, 
															 Matrix& temp_grad_W) {

		// convolutive model:  Z2 = W (convolve with) A1
		const int P = grad_W.extent(1);
		const int Q = grad_W.extent(2);
		const int R = grad_W.extent(0);
		bool bad_parameters = false;
		if (deltas_Z2.extent(0) != A1.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(2) != A1.extent(1)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(3) != A1.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		const int minibatch_size = A1.extent(0);

		if (temp_deltas_Z2.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_deltas_Z2.extent(1) != R ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(1) != (P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_grad_W.extent(0) != (P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_grad_W.extent(1) != R) {
			bad_parameters = true;
		}

		if (bad_parameters) {
			cerr << "compute_weight_grad_convolutive_minibatch(): wrong dimensions." << endl;
			exit(1);
		}

		// Copy date into temp_A1:
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			// Copy data from A1 into temp_A1 for current minibatch.
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					int row_ind = c + r*N + cur_batch*M*N;
					int col_ind = 0; 
					for (int i = 0; i < P; ++i) {
						for (int j = 0; j < Q; ++j) {
							if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
								temp_A1(row_ind, col_ind) = A1(cur_batch, r - i, c - j);
							} else {
								// Write 0.
								temp_A1(row_ind, col_ind) = 0.0f;
							}
							++col_ind;
						}
					}
					// Write a 1 for the bias term.
					temp_A1(row_ind, col_ind) = 1.0f;
				}
			}
		}

		// Copy data into temp_deltas_Z2:
		#pragma omp parallel for collapse(2)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int k = 0; k < R; ++k) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						temp_deltas_Z2(c + r*N + cur_batch*M*N, k) = deltas_Z2(cur_batch, k, r, c);
					}
				}
			}	
		}
		
		// Compute temp_grad_W = temp_A1^T * temp_deltas_Z2.
		mat_multiply_left_transpose(temp_grad_W, temp_A1, temp_deltas_Z2);

		// Copy data from W and bias into temp_grad_W. 
		//
		// Format for temp_W:
		// Each column of temp_W will contain the unrolled version of one of the R convolutional filters, plus the
		// corresponding bias value at the bottom (i.e., last row).

		#pragma omp parallel for collapse(3)
		for (int k = 0; k < R; ++k) {
			// For each of the R convolutional filters:
			for (int i = 0; i < P; ++i) {
				for (int j = 0; j < Q; ++j) {
					grad_W(k, i,j) = temp_grad_W(j + i*Q, k);
				}
			}
		}
		
		/*
#pragma omp parallel for collapse(3)
		for (int r = 0; r < P; ++r) {
			for (int c = 0; c < Q; ++c) {
				// Note: for each work item (r,c), the work below can be performed in parallel. todo: parallelize.
				for (int k = 0; k < R; ++k) {
					float sum = 0.0f;
					for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
						for (int i = 0; i < M; ++i) {
							for (int j = 0; j < N; ++j) {
								if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
									sum += deltas_Z2(cur_batch, k, r + i, c + j)*A1(cur_batch, i,j);
								}
							}
						}
					}
					grad_W(k, r,c) = sum;
				}
			}
		}
		*/


	}


	void compute_3d_weight_grad_convolutive_minibatch_optimized(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1,
																Matrix& temp_deltas_Z2, Matrix& temp_A1, 
															 Matrix& temp_grad_W)   {
		// convolutive model:  Z2 = W (convolve with) A1
		const int R = grad_W.extent(0);
		const int D = grad_W.extent(1);
		const int P = grad_W.extent(2);
		const int Q = grad_W.extent(3);
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		//#pragma omp parallel for private(r,c,k,i,j,sum)
		const int minibatch_size = A1.extent(0);

		bool bad_parameters = false;
		if (deltas_Z2.extent(0) != A1.extent(0)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(2) != A1.extent(2)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(3) != A1.extent(3)) {
			bad_parameters = true;
		}
		if (deltas_Z2.extent(1) != R) {
			bad_parameters = true;
		}
		if (D != A1.extent(1)) {
			bad_parameters = true;
		}

		if (temp_deltas_Z2.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_deltas_Z2.extent(1) != R ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(0) != (M*N*minibatch_size) ) {
			bad_parameters = true;
		}
		if (temp_A1.extent(1) != (D*P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_grad_W.extent(0) != (D*P*Q + 1) ) {
			bad_parameters = true;
		}
		if (temp_grad_W.extent(1) != R) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "compute_weight_grad_convolutive_minibatch(): wrong dimensions." << endl;
			exit(1);
		}

		// Copy date into temp_A1:
		#pragma omp parallel for collapse(3)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			// Copy data from A1 into temp_A1 for current minibatch.
			for (int r = 0; r < M; ++r) {
				for (int c = 0; c < N; ++c) {
					int row_ind = c + r*N + cur_batch*M*N;
					int col_ind = 0; 
					for (int d = 0; d < D; ++d) {
						for (int i = 0; i < P; ++i) {
							for (int j = 0; j < Q; ++j) {
								if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
									temp_A1(row_ind, col_ind) = A1(cur_batch, d, r - i, c - j); /// fixme
								} else {
									// Write 0.
									temp_A1(row_ind, col_ind) = 0.0f;
								}
								++col_ind;
							}
						}
						// Write a 1 for the bias term.
						temp_A1(row_ind, col_ind) = 1.0f;
					}
				}
			}
		}

		// Copy data into temp_deltas_Z2:
		#pragma omp parallel for collapse(2)
		for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
			for (int k = 0; k < R; ++k) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						temp_deltas_Z2(c + r*N + cur_batch*M*N, k) = deltas_Z2(cur_batch, k, r, c);
					}
				}
			}	
		}
		
		// Compute temp_grad_W = temp_A1^T * temp_deltas_Z2.
		mat_multiply_left_transpose(temp_grad_W, temp_A1, temp_deltas_Z2);

		// Copy data from W and bias into temp_grad_W. 
		//
		// Format for temp_W:
		// Each column of temp_W will contain the unrolled version of one of the R convolutional filters, plus the
		// corresponding bias value at the bottom (i.e., last row).

	
		#pragma omp parallel for collapse(3)
		for (int k = 0; k < R; ++k) {
			// For each of the R convolutional filters:
			for (int d = 0; d < D; ++d) {
				for (int i = 0; i < P; ++i) {
					for (int j = 0; j < Q; ++j) {
						grad_W(k, d, i,j) = temp_grad_W(d*P*Q + i*Q + j, k);
					}
				}
			}
		}


		/*
#pragma omp parallel for collapse(3)
		for (int r = 0; r < P; ++r) {
			for (int c = 0; c < Q; ++c) {
				for (int k = 0; k < R; ++k) {
					for (int d = 0; d < D; ++d) {
						float sum = 0.0f;
						for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
							for (int i = 0; i < M; ++i) {
								for (int j = 0; j < N; ++j) {
									if (((r + i) < M) && ((c + j) < N)) { // Don't allow out-of-bounds elements.
										sum += deltas_Z2(cur_batch, k, r + i, c + j)*A1(cur_batch, d, i,j);
									}
								}
							}
						}
						grad_W(k, d, r,c) = sum;
					}
				}
			}
		}
		*/
	}


	void compute_bias_grad_convolutive_minibatch(Matrix& grad_bias, const Matrix& deltas_Z2) {
	
		// bias is R x 1 (same as 1-dimensional vector<float> of size R)
		if (grad_bias.order() != 1) {
			cerr << "convolve_2d_filter_with_bias(): grad_bias vector has wrong order." << endl;
			exit(1);
		}
		if (grad_bias.size() != deltas_Z2.extent(1)) {
			cerr << "compute_weight_grad_convolutive(): wrong dimensions." << endl;
			exit(1);
		}
		const int M = deltas_Z2.extent(2);
		const int N = deltas_Z2.extent(3);
		const int R = grad_bias.size();
		const int minibatch_size = deltas_Z2.extent(0);
		#pragma omp parallel for
		for (int k = 0; k < R; ++k) {
			float sum = 0.0f;
			for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
				for (int r = 0; r < M; ++r) {
					for (int c = 0; c < N; ++c) {
						//sum += deltas_Z2(cur_batch, r, c, k);
						sum += deltas_Z2(cur_batch, k, r, c);
					}
				}
			}
			grad_bias[k] = sum;
		}

	}



	float relative_error(const Matrix& a, const Matrix& b) {
		if (a.size() != b.size()) {
			cerr << "relative_error(): Must be same sizes." << endl;
			exit(1);
		}
		float numer = 0.0f; // L2_norm(a - b)
		float a_norm = 0.0f;
		float b_norm = 0.0f;
		for (int i = 0; i != a.size(); ++i) {
			numer += (a[i] - b[i])*(a[i] - b[i]);
			a_norm += a[i]*a[i];
			b_norm += b[i]*b[i];
		}
		numer = sqrt(numer);
		a_norm = sqrt(a_norm);
		b_norm = sqrt(b_norm);
		float denom = 0.0f; // L2_norm(a) + L2_norm( b)
		denom = a_norm + b_norm;
		if (denom > 0.0f) {
			return numer/denom;
		} else {
			return 0.0f;
		}
	}

	


	float max_value(const std::vector<float>& a) {
		float max_val = a[0];
		for (size_t i = 0; i != a.size(); ++i) {
			max_val = max(max_val, a[i]);
		}
		return max_val;
	}

	float min_value(const std::vector<float>& a) {
		float min_val = a[0];
		for (size_t i = 0; i != a.size(); ++i) {
			min_val = min(min_val, a[i]);
		}
		return min_val;
	}


	// H is 3D
	void max_out_3d(Matrix& H, int dim0_box, int dim1_box, int dim2_box) {
		const float min_possible_value = -1e9f; // Set this smaller than any element in H.
		for (int r = 0; r < H.extent(0); r+=dim0_box) {
			for (int c = 0; c < H.extent(1); c+=dim1_box) {
				for (int k = 0; k < H.extent(2); k+=dim2_box) {		
					// (r,c,k) corresponds to corner of the box. This can be work item on GPU.
					// Now find the maximum element in this box.
					float max_val = min_possible_value;
					for (int i = 0; i < dim0_box; ++i) {
						for (int j = 0; j < dim1_box; ++j) {
							for (int l = 0; l < dim2_box; ++l) {
								// Don't allow out-of-bounds elements.
								if (((r + i) < H.extent(0)) && ((c + j) < H.extent(1)) && ((k + l) < H.extent(2))) { 
									float temp = H(r+i, c+j, k+l);
									if (temp > max_val) {
										max_val = temp;
									} 
								}
							}
						}
					}
					// max_val now contains the maximum value of current box.
					// Now set all other elements to close to 0.
					for (int i = 0; i < dim0_box; ++i) {
						for (int j = 0; j < dim1_box; ++j) {
							for (int l = 0; l < dim2_box; ++l) {
								// Don't allow out-of-bounds elements.
								if (((r + i) < H.extent(0)) && ((c + j) < H.extent(1)) && ((k + l) < H.extent(2))) { 
									float temp = H(r+i, c+j, k+l);
									if (temp != max_val) {
										H(r+i, c+j, k+l) = EPSILON_DENORMAL;
									} 
								}
							}
						}
					}

				}
			}
		}
	}

	// H_sub_sampled, H are 3D.
	void max_out_sub_sampling_3d(Matrix& H_sub_sampled, Matrix& H, int dim0_box, int dim1_box, int dim2_box)  {
		const float min_possible_value = -1e9f; // Set this smaller than any element in H.
		bool bad_parameters = false;
		if ((H.extent(0) % H_sub_sampled.extent(0)) != 0) {
			bad_parameters = true;
		}
		if ((H.extent(1) % H_sub_sampled.extent(1)) != 0) {
			bad_parameters = true;
		}
		if (H.extent(2) != H_sub_sampled.extent(2)) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "max_out_sub_sampling_3d(): Inconsistent parameter values." << endl;
			cerr << "H.extent(0) = " << H.extent(0) << endl;
			cerr << "H_sub_sampled.extent(0) = " << H_sub_sampled.extent(0) << endl;
			cerr << "H.extent(1) = " << H.extent(1) << endl;
			cerr << "H_sub_sampled.extent(1) = " << H_sub_sampled.extent(1) << endl;
			cerr << "H.extent(2) = " << H.extent(2) << endl;;
			cerr << "H_sub_sampled.extent(2) = " << H_sub_sampled.extent(2) << endl;
			exit(1);
		}
		const int sub_sample_factor_0 = H.extent(0)/H_sub_sampled.extent(0);
		const int sub_sample_factor_1 = H.extent(1)/H_sub_sampled.extent(1);

		for (int r = 0; r < H_sub_sampled.extent(0); ++r) {
			for (int c = 0; c < H_sub_sampled.extent(1); ++c) {
				for (int k = 0; k < H_sub_sampled.extent(2); ++k) {	
					H_sub_sampled(r,c,k) = 0.0f;
					//H_sub_sampled(r,c,k) = 1.2f;
				}
			}
		}

		for (int r = 0; r < H.extent(0); r+=dim0_box) {
			for (int c = 0; c < H.extent(1); c+=dim1_box) {
				for (int k = 0; k < H.extent(2); k+=dim2_box) {		
					// (r,c,k) corresponds to corner of the box. This can be work item on GPU.
					// Now find the maximum element in this box.
					float max_val = min_possible_value;
					int max_row = 0;
					int max_col = 0;
					int max_channel = 0;
					for (int i = 0; i < dim0_box; ++i) {
						for (int j = 0; j < dim1_box; ++j) {
							for (int l = 0; l < dim2_box; ++l) {
								// Don't allow out-of-bounds elements.
								if (((r + i) < H.extent(0)) && ((c + j) < H.extent(1)) && ((k + l) < H.extent(2))) { 
									float temp = H(r+i, c+j, k+l);
									if (temp > max_val) {
										max_val = temp;
										max_row = r+i;
										max_col = c+j;
										max_channel = k+l;
									} 
								}
							}
						}
					}
					// Set max value in the sub-sampled matrix.
					max_row /= sub_sample_factor_0;
					max_col /= sub_sample_factor_1;
					H_sub_sampled(max_row, max_col, max_channel) = max_val;
					
					//float h_sub_max_s = max_value(H_sub_sampled.get_backing_vector());
					//cout << "h_sub_max_s = " << h_sub_max_s << endl;
					//cout << "max_val = " << max_val << endl;
					//cout << "max_row = " << max_row << endl;
					//cout << "max_col = " << max_col << endl;
					//cout << "max_channel = " << max_channel << endl;
					// max_val now contains the maximum value of current box.
					// Now set all other elements to close to 0.
					for (int i = 0; i < dim0_box; ++i) {
						for (int j = 0; j < dim1_box; ++j) {
							for (int l = 0; l < dim2_box; ++l) {
								// Don't allow out-of-bounds elements.
								if (((r + i) < H.extent(0)) && ((c + j) < H.extent(1)) && ((k + l) < H.extent(2))) { 
									float temp = H(r+i, c+j, k+l);
									if (temp != max_val) {
										H(r+i, c+j, k+l) = EPSILON_DENORMAL;
									} 
								}
							}
						}
					}
					//
				}
			}
		}
		//float h_sub_max_s = max_value(H_sub_sampled.get_backing_vector());
		//cout << "h_sub_max_s = " << h_sub_max_s << endl;
	}

	// B is 4D.
	void reshape_3d_features_to_1d_features(Matrix& A, const Matrix& B) {
		// Check dimensions. Column vector of A should be same size as the 3D box for the i'th
		// sample in B.
		if (A.extent(0) != B.extent(1)*B.extent(2)*B.extent(3)) {
			cerr << "reshape_3d_features_to_1d_features() bad parameters. = " << endl;
			exit(1);
		}
		if (A.extent(1) != B.extent(0)) {
			cerr << "reshape_3d_features_to_1d_features() bad parameters. = " << endl;
			exit(1);
		}
		for (int i = 0; i < B.extent(0); ++i) {
			int row_A = 0;
			for (int j = 0; j < B.extent(1); ++ j) {
				for (int k = 0; k < B.extent(2); ++k) {
					for (int l = 0; l < B.extent(3); ++l) {
						A(row_A, i) = B(i,j,k,l);
						++row_A;
					}
				}
			}
		}
	}

	// A, maxout_vals are 3D.
	// maxout_indices is 4D.
	void compute_maxout_3d(const Matrix& A, Matrix& maxout_vals, MatrixT<int>& maxout_indices) {
		const int dim0_A = A.extent(0);
		const int dim1_A = A.extent(1);
		const int dim2_A = A.extent(2);
		const int dim0_out = maxout_vals.extent(0);
		const int dim1_out = maxout_vals.extent(1);
		const int dim2_out = maxout_vals.extent(2);
		const int maxout_factor_dim0 = dim0_A/dim0_out;
		const int maxout_factor_dim1 = dim1_A/dim1_out;
		const int maxout_factor_dim2 = dim2_A/dim2_out;
		bool bad_parameters = false;
		if ((dim0_A % dim0_out) != 0) {
			bad_parameters = true;
		}
		if ((dim1_A % dim1_out) != 0) {
			bad_parameters = true;
		}
		if ((dim2_A % dim2_out) != 0) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "compute_maxout_3d(): bad parameters." << endl;
			exit(1);
		}
		// Now we perform the maxout operations. The maxout can be computed independently and in parallel for
		// each element of maxout_vals.
		for (int i = 0; i < dim0_out; ++i) {
			for (int j = 0; j < dim1_out; ++j) {
				for (int k = 0; k < dim2_out; ++k) {
					// (i,j,k) can be work item on GPU.
					// Iteratoe over the cube inside A to find the max value.
					int dim0_A_corner = i*maxout_factor_dim0;
					int dim1_A_corner = j*maxout_factor_dim1;
					int dim2_A_corner = k*maxout_factor_dim2;
					int max_ind0 = dim0_A_corner;
					int max_ind1 = dim1_A_corner;
					int max_ind2 = dim2_A_corner;
					float max_val = A(max_ind0, max_ind1, max_ind2);
					for (int l = 0; l < maxout_factor_dim0; ++l) {
						for (int m = 0; m < maxout_factor_dim1; ++m) {
							for (int n = 0; n < maxout_factor_dim2; ++n) {
								float temp = A(l + dim0_A_corner, m + dim1_A_corner, n + dim2_A_corner);
								if (temp > max_val) {
									max_val = temp;
									max_ind0 = l + dim0_A_corner;
									max_ind1 = m + dim1_A_corner;
									max_ind2 = n + dim2_A_corner;
								}
							}
						}
					}
					// We now have the max val for the current cube.
					maxout_vals(i,j,k) = max_val;
					maxout_indices(i,j,k,0) = max_ind0;
					maxout_indices(i,j,k,1) = max_ind1;
					maxout_indices(i,j,k,2) = max_ind2;
				}
			}
		}

	}


	void compute_reverse_maxout_3d(Matrix& A, const Matrix& maxout_vals, const MatrixT<int>& maxout_indices) {
		// fixme: no error checking on maxout_indices.
		const int dim0_A = A.extent(0);
		const int dim1_A = A.extent(1);
		const int dim2_A = A.extent(2);
		const int dim0_out = maxout_vals.extent(0);
		const int dim1_out = maxout_vals.extent(1);
		const int dim2_out = maxout_vals.extent(2);
		//const int maxout_factor_dim0 = dim0_A/dim0_out;
		//const int maxout_factor_dim1 = dim1_A/dim1_out;
		//const int maxout_factor_dim2 = dim2_A/dim2_out;
		bool bad_parameters = false;
		if ((dim0_A % dim0_out) != 0) {
			bad_parameters = true;
		}
		if ((dim1_A % dim1_out) != 0) {
			bad_parameters = true;
		}
		if ((dim2_A % dim2_out) != 0) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "compute_maxout_3d(): bad parameters." << endl;
			exit(1);
		}
		// Zero out A.
		for (int p = 0; p != A.size(); ++p) {
			A[p] = 0.0f;
		}

		// The reverse maxout can be computed independently and in parallel for
		// each element of maxout_vals.
		for (int i = 0; i < dim0_out; ++i) {
			for (int j = 0; j < dim1_out; ++j) {
				for (int k = 0; k < dim2_out; ++k) {
					// (i,j,k) can be work item on GPU.
					float max_val = maxout_vals(i,j,k);
					int max_ind0 = maxout_indices(i,j,k,0);
					int max_ind1 = maxout_indices(i,j,k,1);
					int max_ind2 = maxout_indices(i,j,k,2);
					A(max_ind0, max_ind1, max_ind2) = max_val;
				}
			}
		}

	}



	void forward_3d_max_pool(const Matrix& in_activations, Matrix& out_activations, MatrixT<int>& state, 
						 const std::vector<int>& pooling_region_extents) {
	
		const int minibatch_size = in_activations.extent(0);
		const int in_depth = in_activations.extent(1);
		const int in_height = in_activations.extent(2);
		const int in_width = in_activations.extent(3);
		const int out_depth = out_activations.extent(1);
		const int out_height = out_activations.extent(2);
		const int out_width = out_activations.extent(3);

		// Todo: make this float to support non-int stride + randomized pooling regions.
		const int stride_depth = in_depth/out_depth;
		const int stride_height = in_height/out_height;
		const int stride_width = in_width/out_width;

		bool bad_parameters = false;
		// For now, require integer stride.
		if ((in_depth % out_depth) != 0) {
			cerr << "Inconsitent depth." << endl;
			bad_parameters = true;
		}
		if ((in_height % out_height) != 0) {
			cerr << "Inconsitent height." << endl;
			bad_parameters = true;
		}
		if ((in_width % out_width) != 0) {
			cerr << "Inconsitent width." << endl;
			bad_parameters = true;
		}
		if (minibatch_size != out_activations.extent(0)) {
			cerr << "Inconsitent mini-batch in out_activations." << endl;
			bad_parameters = true;
		}
		if (minibatch_size != state.extent(0)) {
			cerr << "Inconsitent mini-batch in state." << endl;
			bad_parameters = true;
		}
		if (state.order() != 5) {
			cerr << "state has wrong order. Should be 5." << endl;
			bad_parameters = true;
		}
		if (state.extent(4) != 3) {
			cerr << "state.extent(4) != 3" << endl;
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "forward_3d_max_pool(): bad parameters." << endl;
			exit(1);
		}
		// Now we perform the maxout operations. The maxout can be computed independently and in parallel for
		// each element of maxout_vals.
		#pragma omp parallel for collapse(4)
		for (int minibatch_index = 0; minibatch_index < minibatch_size; ++minibatch_index) {
			for (int i = 0; i < out_depth; ++i) {
				for (int j = 0; j < out_height; ++j) {
					for (int k = 0; k < out_width; ++k) {
						// (minibatch_index,i,j,k) can be work item on GPU.
						// Iteratoe over the cube inside A to find the max value.
						int depth_in_corner = i*stride_depth;
						int height_in_corner = j*stride_height;
						int width_in_corner = k*stride_width;
						int max_ind_depth_in = depth_in_corner;
						int max_ind_height_in = height_in_corner;
						int max_ind_width_in = width_in_corner;
						float max_val = in_activations(minibatch_index, max_ind_depth_in, max_ind_height_in, max_ind_width_in);
						for (int l = 0; l < pooling_region_extents[0]; ++l) {
							for (int m = 0; m < pooling_region_extents[1]; ++m) {
								for (int n = 0; n < pooling_region_extents[2]; ++n) {
									if (((l + depth_in_corner) < in_depth) && ((m + height_in_corner) < in_height) && ((n + width_in_corner) < in_width)) {
										float temp = in_activations(minibatch_index, l + depth_in_corner, m + height_in_corner, n + width_in_corner);
										if (temp > max_val) {
											max_val = temp;
											max_ind_depth_in = l + depth_in_corner;
											max_ind_height_in = m + height_in_corner;
											max_ind_width_in = n + width_in_corner;
										}
									}
								}
							}
						}
						// We now have the max val for the current cube.
						out_activations(minibatch_index, i,j,k) = max_val;
						state(minibatch_index, i,j,k,0) = max_ind_depth_in;
						state(minibatch_index, i,j,k,1) = max_ind_height_in;
						state(minibatch_index, i,j,k,2) = max_ind_width_in;
					}
				}
			}
		}
	}

	void reverse_3d_max_pool(Matrix& in_activations, const Matrix& out_activations, const MatrixT<int>& state) {
		const int minibatch_size = in_activations.extent(0);
		const int in_depth = in_activations.extent(1);
		const int in_height = in_activations.extent(2);
		const int in_width = in_activations.extent(3);
		const int out_depth = out_activations.extent(1);
		const int out_height = out_activations.extent(2);
		const int out_width = out_activations.extent(3);

		// Todo: make this float to support non-int stride + randomized pooling regions.
		//const int stride_depth = in_depth/out_depth;
		//const int stride_height = in_height/out_height;
		//const int stride_width = in_width/out_width;

		bool bad_parameters = false;
		// For now, require integer stride.
		if ((in_depth % out_depth) != 0) {
			bad_parameters = true;
		}
		if ((in_height % out_height) != 0) {
			bad_parameters = true;
		}
		if ((in_width % out_width) != 0) {
			bad_parameters = true;
		}
		if (minibatch_size != out_activations.extent(0)) {
			bad_parameters = true;
		}
		if (minibatch_size != state.extent(0)) {
			bad_parameters = true;
		}
		if (state.order() != 5) {
			bad_parameters = true;
		}
		if (state.extent(4) != 3) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "reverse_3d_max_pool(): bad parameters." << endl;
			exit(1);
		}

		// Zero out in_activations
		for (int p = 0; p != in_activations.size(); ++p) {
			in_activations[p] = 0.0f;
		}
		// The reverse maxout can be computed independently and in parallel for
		// each element of maxout_vals.
		// Warning: Note safe to enable paralle for due to "+=" below. If you wish to enable, must refactor first, which 
		// is straightforward to do.
		//#pragma omp parallel for collapse(4)
		for (int minibatch_index = 0; minibatch_index < minibatch_size; ++minibatch_index) {
			for (int i = 0; i < out_depth; ++i) {
				for (int j = 0; j < out_height; ++j) {
					for (int k = 0; k < out_width; ++k) {
						// (minibatch_index,i,j,k) can be work item on GPU.
						float max_val = out_activations(minibatch_index, i,j,k);
						int max_ind_depth = state(minibatch_index, i,j,k,0);
						int max_ind_height = state(minibatch_index, i,j,k,1);
						int max_ind_width = state(minibatch_index, i,j,k,2);
						// Note that if overlapping pooling regions are used, a particular input activation index can
						// correspond to multiple locations in out_activations. In this case, we should sum all values
						// corresponding to the same input activation.
						// Caution: this might cause undefined and/or undeterministic behavior when implemented on
						// GPU or CPU with OpenMP.
						in_activations(minibatch_index, max_ind_depth, max_ind_height, max_ind_width) += max_val;
					}
				}
			}
		}

	}

	float max_value(const Matrix& A) {
		return max_value(A.get_backing_vector());
	}

	float min_value(const Matrix& A) {
		return min_value(A.get_backing_vector());
	}

	void extract_3d_minibatch(Matrix& A, const Matrix& B, int start_col, int minibatch_size) {
		bool bad_parameters = false;
		if (A.order() != B.order()) {
			cout << "A.order() = " << A.order() << endl;
			cout << "B.order() = " << B.order() << endl;
			bad_parameters = true;
		}
		if ((A.order() != 3) && (A.order() != 4)) {
			cout << "(A.order() != 3) || (A.order() != 4)" << endl;
			cout << "A.order() = " << A.order() << endl;
			bad_parameters = true;
		}
		if (A.extent(0) != minibatch_size) {
			cout << "A.extent(0) != minibatch_size" << endl;
			bad_parameters = true;
		}
		int D = -1;
		int M = -1;
		int N = -1;
		if (A.order() == 3) {
			M = A.extent(1);
			N = A.extent(2);
			if (B.extent(1) != M) {
				cout << "B.extent(1) != M" << endl;
				bad_parameters = true;
			}
			if (B.extent(2) != N) {
				cout << "B.extent(2) != N" << endl;
				bad_parameters = true;
			}
		} else if (A.order() == 4) {
			D = A.extent(1);
			M = A.extent(2);
			N = A.extent(3);
			if (B.extent(1) != D) {
				cout << "B.extent(1) != D" << endl;
				bad_parameters = true;
			}
			if (B.extent(2) != M) {
				cout << "B.extent(2) != M" << endl;
				bad_parameters = true;
			}
			if (B.extent(3) != N) {
				cout << "B.extent(3) != N" << endl;
				bad_parameters = true;
			}
		}
		if ((start_col + minibatch_size) > B.extent(0)) {
			cout << "(start_col + minibatch_size) > B.extent(0)" << endl;
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "extract_3d_minibatch(): bad parameters." << endl;
			exit(1);
		}

		if (A.order() == 3) {
			for (int i = 0; i < minibatch_size; ++i) {
				for (int j = 0; j < M; ++j) {
					for (int k = 0; k < N; ++k) {
						A(i,j,k) = B(i+start_col, j, k);
					}
				}
			}
		} else if (A.order() == 4) {
			for (int i = 0; i < minibatch_size; ++i) {
				for (int d = 0; d < D; ++d) {
					for (int j = 0; j < M; ++j) {
						for (int k = 0; k < N; ++k) {
							A(i,d,j,k) = B(i+start_col, d, j, k);
						}
					}
				}
			}
		}

	}


	void multi_dim_minibatch_to_column_minibatch(Matrix& A, const Matrix&B) {
		bool bad_parameters = false;
		if (A.size() != B.size()) {
			bad_parameters = true;
		}
		const int minibatch_size = A.extent(1);
		if (minibatch_size != B.extent(0)) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "multi_dim_minibatch_to_column_minibatch(): bad parameters." << endl;
			exit(1);
		}
		// Number of dimensions in B that get flattened into a column of data in A.
		const int combine_dims = B.order() -1;
		if (combine_dims == 2) {
			#pragma omp parallel for
			for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
				int col = 0;
				for (int i = 0; i < B.extent(1); ++i) {
					for (int j = 0; j < B.extent(2); ++j) {
							A(col, minibatch_index) = B(minibatch_index, i, j);
							++col;
					}
				}
			}
		} else if (combine_dims == 3) {
			#pragma omp parallel for
			for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
				int col = 0;
				for (int i = 0; i < B.extent(1); ++i) {
					for (int j = 0; j < B.extent(2); ++j) {
						for (int k = 0; k < B.extent(3); ++k) {
							A(col, minibatch_index) = B(minibatch_index, i, j, k);
							++col;
						}
					}
				}
			}
		} else {
			cerr << "multi_dim_minibatch_to_column_minibatch(): This size is not supported yet. Sorry." << endl;
			exit(1);
		}
	}

	void column_minibatch_to_multi_dim_minibatch(const Matrix& A, Matrix&B) {
		bool bad_parameters = false;
		if (A.size() != B.size()) {
			bad_parameters = true;
		}
		const int minibatch_size = A.extent(1);
		if (minibatch_size != B.extent(0)) {
			bad_parameters = true;
		}
		if (bad_parameters) {
			cerr << "multi_dim_minibatch_to_column_minibatch(): bad parameters." << endl;
			exit(1);
		}
		// Number of dimensions in B that get flattened into a column of data in A.
		const int combine_dims = B.order() -1;
        if (combine_dims == 2) {
			#pragma omp parallel for
			for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
				int col = 0;
				for (int i = 0; i < B.extent(1); ++i) {
					for (int j = 0; j < B.extent(2); ++j) {
							 B(minibatch_index, i, j) = A(col, minibatch_index);
							++col;
					}
				}
			}
		} else if (combine_dims == 3) {
			#pragma omp parallel for
			for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
				int col = 0;
				for (int i = 0; i < B.extent(1); ++i) {
					for (int j = 0; j < B.extent(2); ++j) {
						for (int k = 0; k < B.extent(3); ++k) {
							 B(minibatch_index, i, j, k) = A(col, minibatch_index);
							++col;
						}
					}
				}
			}
		} else {
			cerr << "multi_dim_minibatch_to_column_minibatch(): This size is not supported yet. Sorry." << endl;
			exit(1);
		}
	}


	void threshold_lower(Matrix& X, float min_val) {
		#pragma omp parallel for
		for (int i = 0; i < X.size(); ++i) {
		X[i] = max(X[i], min_val);
		}
	}

	void threshold_upper(Matrix& X, float max_val)  {
		#pragma omp parallel for
		for (int i = 0; i < X.size(); ++i) {
		X[i] = min(X[i], max_val);
		}
	}

	Matrix labels_to_mat(const std::vector<float>& labels) {
		// Find max value in labels.
		float max_val_f = *max_element(labels.begin(), labels.end());
		int max_val = static_cast<int>(max_val_f);
		int class_label_count = max_val + 1;
		Matrix A(class_label_count, labels.size());
		for (size_t i = 0; i != labels.size(); ++i) {
			A(labels.at(i), i) = 1.0f;
		}
		return A;
	}

}
