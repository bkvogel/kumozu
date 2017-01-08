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
//#include <cblas.h>
// Use MKL for optimized matrix multiplication:
#include "mkl.h"

using namespace std;

namespace kumozu {




////////////////////////////////////////////////////////////////////////////////////////////
// Matrix Utilities

void mat_multiply_naive(MatrixF& A, const MatrixF &B, const MatrixF &C) {
    const int rowsOut = B.extent(0);
    const int innerDim = B.extent(1);
    const int colsOut = C.extent(1);
    if (B.extent(1) != C.extent(0)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != rowsOut) || (A.extent(1) != C.extent(1))) {
        A.resize(rowsOut, colsOut);
    }
    float sum;
    // For each row of B
    for (int i = 0; i < rowsOut; i++) {
        // For each column of C
        for (int j = 0; j < colsOut; j++) {
            // Compute dot product of row i of B with column j of C.
            sum = 0;
            for (int k = 0; k < innerDim; k++) {
                sum += B(i, k) * C(k, j);
            }
            A(i, j) =  sum;
        }
    }
}


void mat_multiply(MatrixF& A, const MatrixF &B, const MatrixF &C) {
    
    mat_multiply_blas(A, B, C); // Optimized BLAS version.
    //mat_multiply_naive(A, B, C); // Super slow naive version.
}

void mat_multiply(MatrixF& A, const MatrixF& B, const MatrixF& C, float alpha, float beta) {
    mat_multiply_blas(A, B, C, alpha, beta);
}


// Use this if you have an optimized BLAS implementation (requires you include cblas.h)
void mat_multiply_blas(MatrixF& A, const MatrixF &B, const MatrixF &C) {
    mat_multiply_blas(A, B, C, 1.0f, 0.0f);
}

void mat_multiply_blas(MatrixF& A, const MatrixF &B, const MatrixF &C, float alpha, float beta) {
    if (B.extent(1) != C.extent(0)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }

    const int rows_A = B.extent(0);
    const int cols_A = C.extent(1);
    if ((A.order() != 2) || (A.size() != rows_A*cols_A)) {
        A.resize(rows_A, cols_A);
    }

    float* backingArrayA = A.get_backing_data();
    const float* backingArrayB = B.get_backing_data();
    const float* backingArrayC = C.get_backing_data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.extent(0), A.extent(1), B.extent(1), alpha,
                backingArrayB, B.extent(1), backingArrayC, C.extent(1), beta, backingArrayA, A.extent(1));
}


void randomize_uniform(MatrixF& A, float min, float max) {
    static std::random_device rand_dev;
    static std::mt19937 mersenne_twister_engine(rand_dev());
    //mersenne_twister_engine.seed(static_cast<unsigned long>(time(NULL)));
    std::uniform_real_distribution<float> uni(min, max);
    for (int i = 0; i < A.size(); i++) {
        A[i] = uni(mersenne_twister_engine);
    }
}


void randomize_normal(MatrixF& A, float mean, float std_deviation) {
    static std::random_device rand_dev;
    static std::mt19937 mersenne_twister_engine(rand_dev());
    std::normal_distribution<float> normal_dist(mean, std_deviation);
    for (int i = 0; i < A.size(); i++) {
        A[i] = normal_dist(mersenne_twister_engine);
    }
}


void check_matrix_factorization_dimensions(const MatrixF& X, const MatrixF& W, const MatrixF& H) {
    if (X.extent(0) != W.extent(0)) {
        error_exit("Error: X and W don't have the same number of rows!");
    }
    else if (X.extent(1) != H.extent(1)) {
        std::cerr << "Error: X and H don't have the same number of columns!" << std::endl;
        std::cerr << "Columns in X = " << X.extent(1) << std::endl;
        std::cerr << "Columns in H = " << H.extent(1) << std::endl;
        error_exit("");
    }
    else if (W.extent(1) != H.extent(0)) {
        std::cerr << "Error: Number of columns in W does not equal number of rows in H!" << std::endl;
        std::cerr << "Columns in W = " << W.extent(1) << std::endl;
        std::cerr << "Rows in H = " << H.extent(0) << std::endl;
        error_exit("");
    }
}


bool check_dimensions_a_eq_b_tran_times_c(const MatrixF& A, const MatrixF& B, const MatrixF& C) {
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


bool check_dimensions_a_eq_b_times_c_tran(const MatrixF& A, const MatrixF& B, const MatrixF& C) {
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

void mat_multiply_left_transpose(MatrixF& A, const MatrixF& B, const MatrixF& C) {
    // Compute A = B^T * C
    if (B.extent(0) != C.extent(0)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(1)) || (A.extent(1) != C.extent(1))) {
        A.resize(B.extent(1), C.extent(1));
    }
    float* backingArrayA = A.get_backing_data();
    const float* backingArrayB = B.get_backing_data();
    const float* backingArrayC = C.get_backing_data();

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A.extent(0), A.extent(1), B.extent(0), 1.0f,
                backingArrayB, B.extent(1), backingArrayC, C.extent(1), 0.0f, backingArrayA, A.extent(1));
}

void mat_multiply_left_transpose_naive(MatrixF& A, const MatrixF& B, const MatrixF& C) {
    // Compute A = B^T * C
    if (B.extent(0) != C.extent(0)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(1)) || (A.extent(1) != C.extent(1))) {
        A.resize(B.extent(1), C.extent(1));
    }
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

void mat_multiply_left_transpose_accumulate(MatrixF& A, const MatrixF& B, const MatrixF& C) {
    // Compute A = A + B^T * C
    if (B.extent(0) != C.extent(0)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(1)) || (A.extent(1) != C.extent(1))) {
        A.resize(B.extent(1), C.extent(1));
    }
    float* backingArrayA = A.get_backing_data();
    const float* backingArrayB = B.get_backing_data();
    const float* backingArrayC = C.get_backing_data();

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A.extent(0), A.extent(1), B.extent(0), 1.0f,
                backingArrayB, B.extent(1), backingArrayC, C.extent(1), 1.0f, backingArrayA, A.extent(1));
}

void mat_multiply_left_transpose_naive_accumulate(MatrixF& A, const MatrixF& B, const MatrixF& C) {
    // Compute A = A + B^T * C
    if (B.extent(0) != C.extent(0)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(1)) || (A.extent(1) != C.extent(1))) {
        A.resize(B.extent(1), C.extent(1));
    }
    int rows_A = A.extent(0);
    int cols_A = A.extent(1);
#pragma omp parallel for
    for (int col = 0; col < cols_A; col++) {
        for (int row = 0; row < rows_A; row++) {
            float new_val_A = 0.0f;
            for (int cur_feature = 0; cur_feature < B.extent(0); cur_feature++) {
                new_val_A += (B(cur_feature, row)) * (C(cur_feature, col));
            }
            A(row, col) += new_val_A;
        }
    }
}


void mat_multiply_right_transpose(MatrixF& A, const MatrixF& B, const MatrixF& C) {
    // Compute A = B * C^T
    if (B.extent(1) != C.extent(1)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(0)) || (A.extent(1) != C.extent(0))) {
        A.resize(B.extent(0), C.extent(0));
    }
    float* backingArrayA = A.get_backing_data();
    const float* backingArrayB = B.get_backing_data();
    const float* backingArrayC = C.get_backing_data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B.extent(0), C.extent(0), B.extent(1), 1.0f,
                backingArrayB, B.extent(1), backingArrayC, C.extent(1), 0.0f, backingArrayA, A.extent(1));
}

void mat_multiply_right_transpose_naive(MatrixF& A, const MatrixF& B, const MatrixF& C)  {
    // Compute A = B * C^T
    if (B.extent(1) != C.extent(1)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(0)) || (A.extent(1) != C.extent(0))) {
        A.resize(B.extent(0), C.extent(0));
    }
    int rows_A = A.extent(0);
    int cols_A = A.extent(1);
#pragma omp parallel for
    for (int col = 0; col < cols_A; col++) {
        for (int row = 0; row < rows_A; row++) {
            float new_val_A = 0.0f;
            for (int cur_feature = 0; cur_feature < B.extent(1); cur_feature++) {
                new_val_A += (B(row, cur_feature)) * (C(col, cur_feature));
            }
            A(row, col) = new_val_A;
        }
    }
}


void mat_multiply_right_transpose_accumulate(MatrixF& A, const MatrixF& B, const MatrixF& C) {
    // Compute A = B * C^T + A
    if (B.extent(1) != C.extent(1)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(0)) || (A.extent(1) != C.extent(0))) {
        A.resize(B.extent(0), C.extent(0));
    }
    float* backingArrayA = A.get_backing_data();
    const float* backingArrayB = B.get_backing_data();
    const float* backingArrayC = C.get_backing_data();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B.extent(0), C.extent(0), B.extent(1), 1.0f,
                backingArrayB, B.extent(1), backingArrayC, C.extent(1), 1.0f, backingArrayA, A.extent(1));
}

void mat_multiply_right_transpose_accumulate_naive(MatrixF& A, const MatrixF& B, const MatrixF& C)  {
    // Compute A = B * C^T + A
    if (B.extent(1) != C.extent(1)) {
        error_exit("Error: Inconsistent matrix dimensions! Exiting.");
    }
    if ((A.extent(0) != B.extent(0)) || (A.extent(1) != C.extent(0))) {
        A.resize(B.extent(0), C.extent(0));
    }
    int rows_A = A.extent(0);
    int cols_A = A.extent(1);
#pragma omp parallel for
    for (int col = 0; col < cols_A; col++) {
        for (int row = 0; row < rows_A; row++) {
            float new_val_A = 0.0f;
            for (int cur_feature = 0; cur_feature < static_cast<int>(B.extent(1)); cur_feature++) {
                new_val_A += (B(row, cur_feature)) * (C(col, cur_feature));
            }
            A(row, col) += new_val_A;
        }
    }
}


int sample_multinomial_distribution(const MatrixF& pdf) {
    static std::random_device rand_dev;
    static std::mt19937 mersenne_twister_engine(rand_dev());
    vector<float> pdf_vec;
    pdf_vec.reserve(pdf.size());
    for (auto i = 0; i < pdf.size(); ++i) {
        pdf_vec.push_back(pdf[i]);
    }
    std::discrete_distribution<> dist(pdf_vec.begin(), pdf_vec.end());
    return dist(mersenne_twister_engine);
}

int error_count(const MatrixF& predictions, const MatrixI target_labels) {
    const int num_classes = predictions.extent(0);
    const int test_data_count = predictions.extent(1);
    if (test_data_count != target_labels.extent(0)) {
        error_exit("error_count(): Inconsistent dimensions. Exiting.");
    }
    int errors = 0;
    for (int c = 0; c < test_data_count; ++c) {
        // Get max value for each column of network_output.
        int max_row = 0;
        float max_val = predictions(0, c);
        for (int r = 1; r < num_classes; ++r) {
            if (predictions(r, c) > max_val) {
                max_val = predictions(r, c);
                max_row = r;
            }
        }
        if (max_row != target_labels[c]) {
            ++errors;
        }
    }
    return errors;
}

/*
int confusion_matrix(const MatrixF& predictions, const MatrixI target_labels, MatrixI& confusion){
    const int num_classes = predictions.extent(0);
    const vector<int> expected_extents = {num_classes, num_classes};
    if (confusion.get_extents() != expected_extents) {
        confusion.resize(expected_extents);
        set_value(confusion, 0);
    }
    const int test_data_count = predictions.extent(1);
    if (test_data_count != target_labels.extent(0)) {
        error_exit("error_count(): Inconsistent dimensions. Exiting.");
    }
    int errors = 0;
    for (int c = 0; c < test_data_count; ++c) {
        // Get max value for each column of network_output.
        int max_row = 0;
        float max_val = predictions(0, c);
        for (int r = 1; r < num_classes; ++r) {
            if (predictions(r, c) > max_val) {
                max_val = predictions(r, c);
                max_row = r;
            }
        }
        if (max_row != target_labels[c]) {
            ++errors;
        }
        confusion(target_labels[c], max_row) += 1;
    }
    return errors;
}
*/


void forward_maxout(const MatrixF& input, MatrixF& output, Matrix<int>& state)  {
    int cols = input.extent(1);
    if (cols != output.extent(1)) {
        error_exit("forward_maxout(): Inconsistent dimensions. Exiting.");
    }
    if (output.extent(0) != state.extent(0)) {
        error_exit("forward_maxout(): nconsistent dimensions. Exiting.");
    }
    int in_rows = input.extent(0);
    int out_rows = output.extent(0);
    if (in_rows % out_rows != 0) {
        error_exit("forward_maxout(): Inconsistent dimensions. Exiting.");
    }
    if (state.get_extents() != output.get_extents()) {
        state.resize(output.get_extents());
    }
    int dim_reduce_factor = in_rows / out_rows;
    int row_in, out_ind, start, stop, c, row_out, temp_row;
    float out_val;
#pragma omp parallel for private(row_in, out_ind, start, stop, c, row_out, temp_row, out_val)
    for (c = 0; c < output.extent(1); ++c) {
        for (row_out = 0; row_out < output.extent(0); ++row_out) {
            // Number of parallel independent threads = size of output.
            row_in = row_out*dim_reduce_factor;
            out_val = input(row_in, c);
            out_ind = row_in;
            start = row_in + 1;
            stop = row_in + dim_reduce_factor;
            for (temp_row = start; temp_row < stop; ++temp_row) {
                if (input(temp_row, c) > out_val) {
                    out_val = input(temp_row, c);
                    out_ind = temp_row;
                }
            }
            output(row_out, c) = out_val;
            state(row_out, c) = out_ind;
        }
    }

}


void compute_reverse_maxout(MatrixF& input_backward, const MatrixF& output_backward, const Matrix<int>& state, bool accumulate) {
    int cols = input_backward.extent(1);
    if (cols != output_backward.extent(1)) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    if (output_backward.extent(0) != state.extent(0)) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    int in_rows = input_backward.extent(0);
    int out_rows = state.extent(0);
    if (in_rows < out_rows) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    if (in_rows % out_rows != 0) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    set_value(input_backward, 0.0f);
    // todo: parallelize
    for (int r = 0; r != out_rows; ++r) {
        for (int c = 0; c != cols; ++c) {
            int in_row = state(r, c);
            float val = output_backward(r, c);
            if (accumulate) {
                input_backward(in_row, c) += val;
            } else {
                input_backward(in_row, c) = val;
            }
        }
    }
}


void compute_reverse_maxout_decay_unused(MatrixF& input_backward, const MatrixF& input, const MatrixF& output_backward,
                                         const Matrix<int>& state, float decay_val, bool accumulate) {
    int cols = input_backward.extent(1);
    if (cols != output_backward.extent(1)) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    if (output_backward.extent(0) != state.extent(0)) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    int in_rows = input_backward.extent(0);
    int out_rows = state.extent(0);
    if (in_rows < out_rows) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    if (in_rows % out_rows != 0) {
        error_exit("Inconsistent dimensions. Exiting.");
    }
    //copy_matrix(input_backward, input);
    //scale(input_backward, decay_val);
    // fixme: accumulate correctly.

    map2(input_backward, input_backward, input, [=] (float b, float c) {
        if (accumulate) {
            return b + c*decay_val;
        } else {
            return c*decay_val;
        }
    });
    for (int r = 0; r != out_rows; ++r) {
        for (int c = 0; c != cols; ++c) {
            int in_row = state(r, c);
            float val = output_backward(r, c);
            if (accumulate) {
                input_backward(in_row, c) += val;
            } else {
                input_backward(in_row, c) = val;
            }
        }
    }
}

void forward_max_product(MatrixF& x, const MatrixF& W, const MatrixF& z, Matrix<int>& state) {
    const int M = x.extent(0);
    const int B = x.extent(1);
    const int N = W.extent(1);

    assertion(M == W.extent(0), "forward_max_product(): x and W have different number of rows.");
    assertion(B == z.extent(1), "forward_max_product(): x and z have different number of columns.");
    assertion(N == z.extent(0), "forward_max_product(): column count of W is not the same as row count of z.");
    if (x.get_extents() != state.get_extents()) {
        state.resize(x.get_extents());
    }
#pragma omp parallel for collapse(2)
    // i is row of x
    for (int i = 0; i < M; ++i) {
        // k is column of x
        for (int k = 0; k < B; ++k) {
            // j is column of W = row of z
            float max = W(i,0)*z(0,k);
            int j_max = 0;
            for (int j = 1; j < N; ++j) {
                if (W(i,j)*z(j,k) > max) {
                    max = W(i,j)*z(j,k);
                    j_max = j;
                }
            }
            x(i,k) = max;
            state(i,k) = j_max;
        }
    }
}

void backward_max_product_parameter_gradient(const MatrixF& x_grad, MatrixF& W_grad,
                                             const MatrixF& z, const Matrix<int>& state, bool accumulate) {
    const int M = x_grad.extent(0);
    const int B = x_grad.extent(1);
    const int N = W_grad.extent(1);

    assertion(M == W_grad.extent(0), "backward_max_product_parameter_gradient(): x and W_grad have different number of rows.");
    assertion(B == z.extent(1), "backward_max_product_parameter_gradient(): x and z have different number of columns.");
    assertion(N == z.extent(0), "backward_max_product_parameter_gradient(): column count of W is not the same as row count of z.");
    assertion(x_grad.get_extents() == state.get_extents(),
              "backward_max_product_parameter_gradient(): x_grad and state have different dimensions.");

    if (!accumulate) {
        set_value(W_grad, 0.0f);
    }
    // Note: Safe to parralize over i only.
#pragma omp parallel for
    // i is row of x
    for (int i = 0; i < M; ++i) {
        // k is column of x
        for (int k = 0; k < B; ++k) {
            const int j_max = state(i,k);
            W_grad(i, j_max) += x_grad(i,k)*z(j_max, k);
        }
    }
}

void backward_max_product_input_gradient(const MatrixF& x_grad, const MatrixF& W,
                                         MatrixF& z_grad, const Matrix<int>& state,
                                         bool accumulate) {
    const int M = x_grad.extent(0);
    const int B = x_grad.extent(1);
    const int N = W.extent(1);

    assertion(M == W.extent(0), "backward_max_product_input_gradient(): x and W have different number of rows.");
    assertion(B == z_grad.extent(1), "backward_max_product_input_gradient(): x and z_grad have different number of columns.");
    assertion(N == z_grad.extent(0), "backward_max_product_input_gradient(): column count of W is not the same as row count of z.");
    assertion(x_grad.get_extents() == state.get_extents(),
              "backward_max_product_input_gradient(): x_grad and state have different dimensions.");

    if (!accumulate) {
        set_value(z_grad, 0.0f);
    }

    // k is column of x
#pragma omp parallel for
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < M; ++i) {
            const int j_max = state(i,k);
            z_grad(j_max, k) += x_grad(i,k)*W(i, j_max);
        }
    }
}

void forward_max_product_blend(MatrixF& x, const MatrixF& W, const MatrixF& z,
                               Matrix<int>& state, float alpha) {
    const int M = x.extent(0);
    const int B = x.extent(1);
    const int N = W.extent(1);

    assertion(M == W.extent(0), "forward_max_product_blend(): x and W have different number of rows.");
    assertion(B == z.extent(1), "forward_max_product_blend(): x and z have different number of columns.");
    assertion(N == z.extent(0), "forward_max_product_blend(): column count of W is not the same as row count of z.");
    if (x.get_extents() != state.get_extents()) {
        state.resize(x.get_extents());
    }
#pragma omp parallel for collapse(2)
    // i is row of x
    for (int i = 0; i < M; ++i) {
        // k is column of x
        for (int k = 0; k < B; ++k) {
            // j is column of W = row of z
            float max = W(i,0)*z(0,k);
            int j_max = 0;
            for (int j = 1; j < N; ++j) {
                if (W(i,j)*z(j,k) > max) {
                    max = W(i,j)*z(j,k);
                    j_max = j;
                }
            }
            x(i,k) = max;
            state(i,k) = j_max;
            // Now add the alpha component:
            if (alpha > 0) {
                for (int j = 0; j < N; ++j) {
                    if (j != j_max) {
                        x(i,k) += alpha*W(i,j)*z(j,k);
                    }
                }
            }
        }
    }
}

void backward_max_product_blend_parameter_gradient(const MatrixF& x_grad, MatrixF& W_grad,
                                                   const MatrixF& z, const Matrix<int>& state,
                                                   float alpha, bool accumulate) {
    const int M = x_grad.extent(0);
    const int B = x_grad.extent(1);
    const int N = W_grad.extent(1);

    assertion(M == W_grad.extent(0), "backward_max_product_blend_parameter_gradient(): x and W_grad have different number of rows.");
    assertion(B == z.extent(1), "backward_max_product_blend_parameter_gradient(): x and z have different number of columns.");
    assertion(N == z.extent(0), "backward_max_product_blend_parameter_gradient(): column count of W is not the same as row count of z.");
    assertion(x_grad.get_extents() == state.get_extents(),
              "backward_max_product_blend_parameter_gradient(): x_grad and state have different dimensions.");

    if (!accumulate) {
        set_value(W_grad, 0.0f);
    }
    // Note: Safe to parralize over i only.
#pragma omp parallel for
    // i is row of x
    for (int i = 0; i < M; ++i) {
        // k is column of x
        for (int k = 0; k < B; ++k) {
            const int j_max = state(i,k);
            W_grad(i, j_max) += x_grad(i,k)*z(j_max, k);
            // Now add the alpha component:
            if (alpha > 0) {
                for (int j = 0; j < N; ++j) {
                    if (j != j_max) {
                        W_grad(i, j) += alpha*x_grad(i,k)*z(j, k);
                    }
                }
            }
        }
    }
}

void backward_max_product_blend_input_gradient(const MatrixF& x_grad, const MatrixF& W,
                                               MatrixF& z_grad, const Matrix<int>& state,
                                               float alpha, bool accumulate) {
    const int M = x_grad.extent(0);
    const int B = x_grad.extent(1);
    const int N = W.extent(1);

    assertion(M == W.extent(0), "backward_max_product_blend_input_gradient(): x and W have different number of rows.");
    if ((B != z_grad.extent(1)) or (N == z_grad.extent(0))) {
        z_grad.resize(N, B);
    }

    assertion(x_grad.get_extents() == state.get_extents(),
              "backward_max_product_blend_input_gradient(): x_grad and state have different dimensions.");

    if (!accumulate) {
        set_value(z_grad, 0.0f);
    }

    // k is column of x
#pragma omp parallel for
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < M; ++i) {
            const int j_max = state(i,k);
            z_grad(j_max, k) += x_grad(i,k)*W(i, j_max);
            // Now add the alpha component:
            if (alpha > 0) {
                for (int j = 0; j < N; ++j) {
                    if (j != j_max) {
                        z_grad(j, k) += alpha*x_grad(i,k)*W(i, j);
                    }
                }
            }
        }
    }
}


// deprecated and buggy.
void compute_forward_kmax(const MatrixF& kmax_in, MatrixF& kmax_out_values, Matrix<int>& kmax_out_indices,
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
        error_exit("");
    }
    int partition_size = M/partition_count;
    if (k > partition_size) {
        cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
        cerr << "k = " << k << endl;
        cerr << "partition_size = " << partition_size << endl;
        error_exit("");
    }
    if (N != kmax_out_indices.extent(1)) {
        cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
        cerr << "kmax_in.extent(1) = " << kmax_in.extent(1) << endl;
        cerr << "kmax_out_indices.extent(1) = " << kmax_out_indices.extent(1) << endl;
        error_exit("");
    }
    if (kmax_out_indices.extent(0) != k*partition_count) {
        cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
        cerr << "kmax_out_indices.extent(0) = " << kmax_out_indices.extent(0) << endl;
        cerr << "k*partition_count = " << k*partition_count << endl;
        error_exit("");
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
                    sorted_vals[q] = kmax_in(q + p*partition_size, col); // col out of bounds.
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


// todo: parallelize this function.
void compute_forward_kmax_v2(const MatrixF& kmax_in, MatrixF& kmax_out_values, Matrix<int>& kmax_out_indices,
                             int partition_count, int k) {
    // Implementation note: Each sub-partition of a column can be executed in parallel.
    // Each column can be executed in parallel.

    // Check dimensions.
    check_dimensions(kmax_in, kmax_out_values);
    check_dimensions(kmax_in, kmax_out_indices);

    int M = kmax_in.extent(0);
    int N = kmax_in.extent(1);
    if ((M % partition_count) != 0) {
        cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
        cerr << "kmax_in.extent(0) = " << kmax_in.extent(0) << endl;
        cerr << "partition_count = " << partition_count << endl;
        error_exit("");
    }
    int partition_size = M/partition_count;
    if (k > partition_size) {
        cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
        cerr << "k = " << k << endl;
        cerr << "partition_size = " << partition_size << endl;
        error_exit("");
    }

    set_value(kmax_out_values, 0.0f);
    set_value(kmax_out_indices, 0); // Might not be necessary.
    //vector<float> sorted_vals(partition_size);
    //vector<int> sorted_indices(partition_size);
    //#pragma omp parallel for private(sorted_vals, sorted_indices)
    //const int reuse_count = 16;
    //#pragma omp parallel for // fixme: re-enalbe after debug
    vector<float> sorted_vals(partition_size);
    vector<int> sorted_indices(partition_size);
    for (int col = 0; col < N; ++col) {
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

void forward_leaky_kmax(const MatrixF& kmax_in, MatrixF& kmax_out_values, MatrixI& kmax_out_indices,
                        int partition_count, int k, float alpha){
    // Implementation note: Each sub-partition of a column can be executed in parallel.
    // Each column can be executed in parallel.

    // Check dimensions.
    check_dimensions(kmax_in, kmax_out_values);
    if (kmax_in.get_extents() != kmax_out_indices.get_extents()) {
        kmax_out_indices.resize(kmax_in.get_extents());
    }

    int M = kmax_in.extent(0); // Activation unit count.
    int N = kmax_in.extent(1); // mini-batch size.
    if ((M % partition_count) != 0) {
        cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
        cerr << "kmax_in.extent(0) = " << kmax_in.extent(0) << endl;
        cerr << "partition_count = " << partition_count << endl;
        error_exit("Bad partition count.");
    }
    int partition_size = M/partition_count;
    if (k > partition_size) {
        cerr << "compute_forward_kmax(): Inconsistent parameters. Exiting." << endl;
        cerr << "k = " << k << endl;
        cerr << "partition_size = " << partition_size << endl;
        error_exit("Bad k value.");
    }

    set_value(kmax_out_values, 0.0f);
    set_value(kmax_out_indices, 0); // Might not be necessary.
    //vector<float> sorted_vals(partition_size);
    //vector<int> sorted_indices(partition_size);
    //#pragma omp parallel for private(sorted_vals, sorted_indices)
    //const int reuse_count = 16;
    //#pragma omp parallel for // fixme: re-enalbe after debug
    vector<float> sorted_vals(partition_size);
    vector<int> sorted_indices(partition_size);
    for (int col = 0; col < N; ++col) {
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
            // Now scale the remaining values (which are not in the k-largest set) by the leakiness factor:
            for (int n = k; n < partition_size; ++n) {
                kmax_out_values(sorted_indices[n] + p*partition_size, col) = alpha*sorted_vals[sorted_indices[n]];
                kmax_out_indices(sorted_indices[n] + p*partition_size, col) = 0;
            }
        }
    }
}



void compute_reverse_kmax(MatrixF& kmax_in, const MatrixF& kmax_out_values, const Matrix<int>& kmax_out_indices,
                          int partition_count, int k) {
    // Implementation note: Each sub-partition of a column can be executed in parallel.
    // Check dimensions.
    check_dimensions(kmax_in, kmax_out_values);
    int M = kmax_in.extent(0);
    int N = kmax_in.extent(1);
    auto message = "compute_forward_kmax(): Inconsistent parameters. Exiting.";
    if ((M % partition_count) != 0) {
        error_exit(message);
    }
    int partition_size = M/partition_count;
    if (k > partition_size) {
        error_exit(message);
    }
    if (N != kmax_out_indices.extent(1)) {
        error_exit(message);
    }
    if (kmax_out_indices.extent(0) != k*partition_count) {
        error_exit(message);
    }
    set_value(kmax_in, 0.0f);
    for (int row = 0; row != kmax_out_indices.extent(0); ++row) {
        for (int col = 0; col != N; ++col) {
            int in_row = kmax_out_indices(row, col);
            kmax_in(in_row, col) = kmax_out_values(in_row, col);
        }
    }
}

void compute_reverse_kmax_v2(MatrixF& kmax_in, const MatrixF& kmax_out_values, const Matrix<int>& kmax_out_indices,
                             int partition_count, int k) {
    // Implementation note: Each sub-partition of a column can be executed in parallel.
    // Check dimensions.
    check_dimensions(kmax_in, kmax_out_values);
    check_dimensions(kmax_in, kmax_out_indices);
    int M = kmax_in.extent(0);
    int N = kmax_in.extent(1);
    if ((M % partition_count) != 0) {
        error_exit("compute_forward_kmax_v2(): Inconsistent parameters. Exiting.");
    }
    int partition_size = M/partition_count;
    if (k > partition_size) {
        error_exit("compute_forward_kmax_v2(): Inconsistent parameters. Exiting.");
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

void compute_reverse_kmax_decay_unused(MatrixF& input_backward, const MatrixF& inputs, const MatrixF& output_backward, const MatrixI& state,
                                       int partition_count, int k, float decay_val) {
    // Implementation note: Each sub-partition of a column can be executed in parallel.
    // Check dimensions.
    check_dimensions(input_backward, output_backward);
    check_dimensions(input_backward, state);
    int M = input_backward.extent(0);
    int N = input_backward.extent(1);
    if ((M % partition_count) != 0) {
        error_exit("compute_reverse_kmax_decay_unused(): Inconsistent parameters. Exiting.");
    }
    int partition_size = M/partition_count;
    if (k > partition_size) {
        error_exit("compute_forward_kmax_v3(): Inconsistent parameters. Exiting.");
    }

    set_value(input_backward, 0.0f);
    for (int row = 0; row != M; ++row) {
        for (int col = 0; col != N; ++col) {
            if (state(row, col) == 1) {
                // This activation was chosen as one of the k-max during the forward pass, so its gradient
                // is the same as the corresponding output error.
                input_backward(row, col) = output_backward(row, col);
            } else {
                // This activation was NOT chosen as one of the k-max during the forward pass, so its gradient
                // is chosen to be equal to the value of the corresponding activation from the forward pass.
                // This is its error because this is the amount by which it differs from zero, which is the ideal
                // value. That is, if the network were perfectly efficient, only the k-max activations would be non-zero
                // and all other activations would be very close to zero.
                input_backward(row, col) = decay_val*inputs(row, col);
            }
        }
    }
}

void forward_3d_kmax(const MatrixF& input, MatrixF& output, Matrix<int>& state,
                     int box_depth, int box_height, int box_width, int k) {
    // input: (minibatch_size x depth x height x width) matrix containing the input values.
    const int minibatch_size = input.extent(0);
    const int depth = input.extent(1);
    const int height = input.extent(2);
    const int width = input.extent(3);
    bool bad_parameters = false;
    check_dimensions(input, output);
    check_dimensions(input, state);
    if (bad_parameters) {
        error_exit("forward_3d_kmax(): bad parameters.");
    }
    set_value(output, 0.0f);
    set_value(state, 0);
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
                                    temp = input(minibatch_index, l+ind_depth, m+ind_height, n+ind_width);
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
                                    output(minibatch_index, l+ind_depth, m+ind_height, n+ind_width) = temp;
                                    state(minibatch_index, l+ind_depth, m+ind_height, n+ind_width) = 1;
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

void reverse_3d_kmax(MatrixF& input_backward, const MatrixF& output_backward, const Matrix<int>& state) {
    // input_backward: (minibatch_size x depth x height x width) matrix containing the input values.
    check_dimensions(input_backward, output_backward);
    check_dimensions(input_backward, state);
    set_value(input_backward, 0.0f);
#pragma omp parallel for
    for (int n = 0; n < input_backward.size(); ++n) {
        if (1 == state[n]) {
            input_backward[n] = output_backward[n];
        }
    }

}

void reverse_3d_kmax_decay_unused(MatrixF& input_backward, const MatrixF& input, const MatrixF& output_backward,
                                  const Matrix<int>& state, float decay_val)  {
    // input_backward: (minibatch_size x depth x height x width) matrix containing the input values.
    check_dimensions(input_backward, output_backward);
    check_dimensions(input_backward, state);
    //set_value(input_backward, 0.0f);
#pragma omp parallel for
    for (int n = 0; n < input_backward.size(); ++n) {
        if (1 == state[n]) {
            input_backward[n] = output_backward[n];
        } else {
            input_backward[n] = decay_val*input[n];
        }
    }

}

void compute_forward_relu(const MatrixF& input, MatrixF& output, Matrix<int>& state) {
    check_dimensions(input, output);
    check_dimensions(input, state);
    set_value(output, 0.0f);
    set_value(state, 0);
#pragma omp parallel for
    for (int n = 0; n < input.size(); ++n) {
        double temp = input[n];
        if (temp > 0.0f) {
            output[n] = temp;
            state[n] = 1;
        }
    }
}

void compute_forward_leaky_relu(const MatrixF& input, MatrixF& output, Matrix<int>& state) {
    check_dimensions(input, output);
    check_dimensions(input, state);
    set_value(output, 0.0f);
    set_value(state, 0);
    //const float leakiness = 0.01f; // Normal leaky
    const float leakiness = 0.33f; // Very leaky
#pragma omp parallel for
    for (int n = 0; n < input.size(); ++n) {
        double temp = input[n];
        if (temp > 0.0f) {
            output[n] = temp;
            state[n] = 1;
        } else {
            output[n] = leakiness*temp; // leaky
            state[n] = -1;
        }
    }
}

void compute_forward_tanh(const MatrixF& input, MatrixF& output) {
    if (input.get_extents() != output.get_extents()) {
        output.resize(input.get_extents());
    }
#pragma omp parallel for
    for (int n = 0; n < input.size(); ++n) {
        output[n] = std::tanh(input[n]);
    }
}

void compute_reverse_tanh(MatrixF& input_backward, const MatrixF& output_forward, const MatrixF& output_backward, bool accumulate) {
    if (input_backward.get_extents() != output_forward.get_extents()) {
        input_backward.resize(output_forward.get_extents());
    }
#pragma omp parallel for
    for (int n = 0; n < output_forward.size(); ++n) {
        if (accumulate) {
            input_backward[n] += (1.0f - output_forward[n]*output_forward[n])*output_backward[n];
        } else {
            input_backward[n] = (1.0f - output_forward[n]*output_forward[n])*output_backward[n];
        }
    }
}

void compute_forward_sigmoid(const MatrixF& input, MatrixF& output) {
    if (input.get_extents() != output.get_extents()) {
        output.resize(input.get_extents());
    }
#pragma omp parallel for
    for (int n = 0; n < input.size(); ++n) {
        output[n] = 1.0f/(1.0f + std::exp(-input[n]));
    }
}

void compute_reverse_sigmoid(MatrixF& input_backward, const MatrixF& output_forward, const MatrixF& output_backward, bool accumulate) {
    if (input_backward.get_extents() != output_forward.get_extents()) {
        input_backward.resize(output_forward.get_extents());
    }
#pragma omp parallel for
    for (int n = 0; n < output_forward.size(); ++n) {
        if (accumulate) {
            input_backward[n] += (output_forward[n]*(1.0f - output_forward[n]))*output_backward[n];
        } else {
            input_backward[n] = (output_forward[n]*(1.0f - output_forward[n]))*output_backward[n];
        }
    }
}


void compute_forward_identity_activation(const MatrixF& in_vals, MatrixF& out_vals, Matrix<int>& out_indices) {
    check_dimensions(in_vals, out_vals);
    check_dimensions(in_vals, out_indices);
    set_value(out_vals, 0.0f);
    set_value(out_indices, 0);
#pragma omp parallel for
    for (int n = 0; n < in_vals.size(); ++n) {
        double temp = in_vals[n];
        if (true) {
            out_vals[n] = temp;
            out_indices[n] = 1;
        }
    }
}


void compute_reverse_identity_activation(MatrixF& in_vals, const MatrixF& out_vals, const Matrix<int>& out_indices, bool accumulate) {
    check_dimensions(in_vals, out_vals);
    check_dimensions(in_vals, out_indices);
    set_value(in_vals, 0.0f);
#pragma omp parallel for
    for (int n = 0; n < in_vals.size(); ++n) {
        if (true) {
            if (accumulate) {
                in_vals[n] += out_vals[n];
            } else {
                in_vals[n] = out_vals[n];
            }
        }
    }
}

void compute_reverse_relu(MatrixF& input_backward, const MatrixF& output_backward, const Matrix<int>& state, bool accumulate) {
    check_dimensions(input_backward, output_backward);
    check_dimensions(input_backward, state);
    set_value(input_backward, 0.0f);
#pragma omp parallel for
    for (int n = 0; n < input_backward.size(); ++n) {
        if (1 == state[n]) {
            if (accumulate) {
                input_backward[n] += output_backward[n];
            } else {
                input_backward[n] = output_backward[n];
            }
        }
    }
}

void compute_reverse_relu_decay_unused(MatrixF& input_backward, const MatrixF& input, const MatrixF& output_backward,
                                       const Matrix<int>& state, float decay_val)  {
    check_dimensions(input_backward, output_backward);
    check_dimensions(input_backward, state);
    check_dimensions(input_backward, input);
#pragma omp parallel for
    for (int n = 0; n < input_backward.size(); ++n) {
        //int temp = state[n];
        if (1 == state[n]) {
            input_backward[n] = output_backward[n];
        } else {
            // This element was set to 0 during the forward pass, and so was unused. Ideally it should
            // therefore be 0. Therefore, the error is equal to its value during the forward pass.
            input_backward[n] = decay_val*input[n];
        }
    }
}

void compute_reverse_leaky_relu(MatrixF& in_vals, const MatrixF& out_vals, const Matrix<int>& out_indices, bool accumulate) {
    check_dimensions(in_vals, out_vals);
    check_dimensions(in_vals, out_indices);
    set_value(in_vals, 0.0f);
    //const float leakiness = 0.01f; // Normal leaky
    const float leakiness = 0.33f; // Very leaky
    if (accumulate) {
#pragma omp parallel for
        for (int n = 0; n < in_vals.size(); ++n) {
            int temp = out_indices[n];
            if (temp == 1) {
                in_vals[n] += out_vals[n];
            } else if (temp == -1) {
                in_vals[n] += leakiness*out_vals[n];
            }
        }
    } else {
#pragma omp parallel for
        for (int n = 0; n < in_vals.size(); ++n) {
            int temp = out_indices[n];
            if (temp == 1) {
                in_vals[n] = out_vals[n];
            } else if (temp == -1) {
                in_vals[n] = leakiness*out_vals[n];
            }
        }
    }
}







//This computes grad_W = X_error*H^T or
// grad_W += X_error*H^T
// To get mean gradient, you should do element-wise divide by the mini-batch size.
void compute_weight_grad_sgd_minibatch(const MatrixF& X_error, MatrixF& W_grad, const MatrixF& H, bool accumulate) {
    check_matrix_factorization_dimensions(X_error, W_grad, H);
    if (accumulate) {
        mat_multiply_right_transpose_accumulate(W_grad, X_error, H);
    } else {
        mat_multiply_right_transpose(W_grad, X_error, H);
    }
}


void update_parameters_sgd(MatrixF& W, const MatrixF& W_grad, float alpha) {
    check_dimensions(W, W_grad);
#pragma omp parallel for
    for (int backing_index = 0; backing_index < W.size(); ++backing_index) {
        W[backing_index] -= alpha*W_grad[backing_index];
    }
}

void update_parameters_from_decay(MatrixF& W, float decay_val) {
#pragma omp parallel for
    for (int backing_index = 0; backing_index < W.size(); ++backing_index) {
        W[backing_index] = W[backing_index] - decay_val*W[backing_index];
    }
}

void update_parameters_sgd(MatrixF& W, const MatrixF& W_grad, float alpha, float lambda) {
    check_dimensions(W, W_grad);
#pragma omp parallel for
    for (int backing_index = 0; backing_index < W.size(); ++backing_index) {
        W[backing_index] = W[backing_index]  -alpha*W_grad[backing_index] - alpha*lambda*W[backing_index];
    }
}


void update_parameters_sgd(MatrixF& W, const MatrixF& W_grad, float alpha, float lambda,
                                  float sparsity_param, bool force_nonnegative) {
    check_dimensions(W, W_grad);

#pragma omp parallel for
    for (int backing_index = 0; backing_index < W.size(); ++backing_index) {
        float w_i = W[backing_index];
        w_i = w_i - alpha*W_grad[backing_index] - alpha*lambda*w_i;
        // I add the L1 penalty using SGD-L1 (Clipping) method from:
        // "Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty"
        // by Yoshimasa Tsuruoka et al.
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


void update_parameters_rmsprop(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_mean_square,
                                             float alpha, float rho, float epsilon) {
    check_dimensions(W, W_grad);
    // Resize the other matrices if necessary.
    if (W.get_extents() != W_grad_mean_square.get_extents()) {
        W_grad_mean_square.resize(W.get_extents());
    }
#pragma omp parallel for
    for (int i = 0; i < W.size(); ++i) {
        // Update sum of squares of gradients.
        W_grad_mean_square[i] = rho*W_grad_mean_square[i] + (1 -rho)*W_grad[i]*W_grad[i];
        float rms_grad = sqrt(W_grad_mean_square[i]) + epsilon;
        if (rms_grad > 0) {
            float w = W[i];
            w -= alpha*W_grad[i]/rms_grad;
            W[i] = w;
        }

    }

}


void update_parameters_rmsprop_momentum(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_mean_square,
                                                   MatrixF W_momentum, float alpha, float rho, float momentum,
                                                   float epsilon) {
    check_dimensions(W, W_grad);
    // Resize the other matrices if necessary.
    if (W.get_extents() != W_grad_mean_square.get_extents()) {
        W_grad_mean_square.resize(W.get_extents());
    }
    if (W.get_extents() != W_momentum.get_extents()) {
        W_momentum.resize(W.get_extents());
    }
#pragma omp parallel for
    for (int i = 0; i < W.size(); ++i) {
        // Update sum of squares of gradients.
        W_grad_mean_square[i] = rho*W_grad_mean_square[i] + (1 -rho)*W_grad[i]*W_grad[i];
        float rms_grad = sqrt(W_grad_mean_square[i]) + epsilon;
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

void update_parameters_adagrad(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_sum_square,
                                          float alpha) {
    check_dimensions(W, W_grad);
    if (W.get_extents() != W_grad_sum_square.get_extents()) {
        W_grad_sum_square.resize(W.get_extents());
    }
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



void do_backprop_update_sgd_minibatch(const MatrixF& X_error, const MatrixF& W, MatrixF& H_error, bool accumulate) {
    // We compute H_error = W^T * X_error
    if (accumulate) {
        mat_multiply_left_transpose_accumulate(H_error, W, X_error);
    } else {
        mat_multiply_left_transpose(H_error, W, X_error);
    }
}

void do_bias_update_sgd_minibatch(const MatrixF& X_error, const MatrixF& W, const MatrixF& H, std::vector<float>& b,
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
void compute_bias_grad_sgd_minibatch(const MatrixF& X_error, MatrixF& b_grad, bool accumulate) {
    const int minibatch_size = X_error.extent(1);

#pragma omp parallel for
    for (int row_b = 0; row_b < b_grad.size(); ++row_b) {
        // For each element b[row_W], loop over all columns in the mini-batch region of X_error, H to update.
        float avg_grad_w = 0.0f; // Will contain the gradient for W(row_W, col_W).
        for (int col_X_error = 0; col_X_error < minibatch_size; ++col_X_error) {
            avg_grad_w += X_error(row_b, col_X_error);
        }
        if (accumulate) {
            b_grad[row_b] += avg_grad_w;
        } else {
            b_grad[row_b] = avg_grad_w;
        }
    }
}

void do_product_update_naive(MatrixF& X, const MatrixF& W, const MatrixF& H, float update_weight) {
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

void do_product_update(MatrixF& X, const MatrixF& W, const MatrixF& H, float update_weight) {
    mat_multiply_blas(X, W, H, update_weight, 1.0f-update_weight);
}


void do_product_update(MatrixF& X, const MatrixF& W, const MatrixF& H, const MatrixF& b) {
    check_matrix_factorization_dimensions(X, W, H);
    if (b.size() != X.extent(0)) {
        std::cout << "size of bias " << b.size() << std::endl;
        std::cout << "Expected size of bias: " << X.extent(0) << std::endl;
        error_exit("Supplied bias matrix is wrong size.");
    }
    // X = W * H
    mat_multiply(X, W, H);
    //mat_multiply_naive(X, W, H);
    // Now add the bias component:
    int rows_X = X.extent(0);
    int cols_X = X.extent(1);
#pragma omp parallel for collapse(2)
    for (int col = 0; col < cols_X; ++col) {
        for (int row = 0; row < rows_X; ++row) {
            X(row, col) += b[row];
        }
    }

}


float compute_reconstruction_error(const MatrixF& X, const MatrixF& W, const MatrixF& H) {
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



float compute_rmse(const MatrixF& X) {
    float sum_errors = 0.0f;

    float cur_error;
    for (int n = 0; n < X.size(); ++n) {
        cur_error = X[n];
        sum_errors += cur_error*cur_error;
    }
    float rmse = sqrt(sum_errors /X.size());
    return rmse;

}

/*
   * Given two matrices, compute and return the RMSE of their element-wise differences.
   */
float compute_rmse(const MatrixF& A, const MatrixF& B)  {
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


void assert_almost_equal(float a, float b, float tolerance) {
    float diff = std::abs(a - b);
    // Note: score is close to 0 if the difference in magnitude between a and b is small compared
    // to their individual magnitudes.
    if (diff > tolerance) {
        cerr << "Tolerance of " << tolerance << " was exceeded!:" << endl;
        //cerr << "Tolerance exceeded!:" << endl;
        cerr << "a = " << a << endl;
        cerr << "b = " << b << endl;
        cerr << "difference magnitude = " << diff << endl;
        error_exit("");
    }
}


void assert_almost_equal(const MatrixF& A, const MatrixF& B, float tolerance) {
    check_dimensions(A, B);
    float score = relative_error(A, B);
    if (score > tolerance) {
        cerr << "Tolerance of " << tolerance << " was exceeded!:" << endl;
        error_exit("");
    }
}



void print_stats(const MatrixF& A, std::string name) {
    // Compute mean:
    float N = static_cast<float>(A.size());
    float mean = 0.0f;
    for (int i = 0; i != A.size(); ++i) {
        mean += A[i];
    }
    mean /= N;
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
    cout << "min value = " << min_value(A) << endl;
    cout << "max value = " << max_value(A) << endl;
    cout << "-------------------------------------------" << endl;
}





// Uses symmetric zero padding around the input image.
void convolve_3d_filter_with_bias_minibatch(MatrixF& Z2, const MatrixF& W, const MatrixF& bias, const MatrixF& A1) {
    // for i in [0,...,minibatch_size).
    // Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.

    const int R = W.extent(0);
    const int D = W.extent(1);
    const int P = W.extent(2); // conv filter height
    const int Q = W.extent(3); // conv filter width
    const int pad_height = P/2; // offset for roughly equal zero padding. Conv filter size should be odd.
    const int pad_width = Q/2; // offset for roughly equal zero padding. Conv filter size should be odd.

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
        error_exit("convolve_2d_filter_with_bias_minibatch(): Bad parameter sizes.");
    }

    const int M = A1.extent(2); // input image height
    const int N = A1.extent(3); // input image width
    const int minibatch_size = A1.extent(0);
    //#pragma omp parallel for
#pragma omp parallel for collapse(3)
    for (int cur_batch = 0; cur_batch < minibatch_size; ++cur_batch) {
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                // Note: for each work item (cur_batch, r,c), the work below can be performed in parallel.
                for (int k = 0; k < R; ++k) {
                    float sum = 0.0f;
                    // Now compute convultion of k'th filter for the pixel X(r,c).
                    for (int d = 0; d < D; ++d) {
                        for (int i = 0; i < P; ++i) {
                            for (int j = 0; j < Q; ++j) {
                                //if (((r - i) >= 0) && ((c - j) >= 0)) { // Don't allow out-of-bounds elements.
                                //sum += A1(cur_batch, d, r - i, c - j)*W(k, d, i,j);
                                //}
                                // if (((r - i + pad_height) >= 0) && ((c - j + pad_width) >= 0)) { // Don't allow out-of-bounds elements.
                                int t1 = r - i + pad_height;
                                int t2 = c - j + pad_width;
                                if ((t1 >= 0) && (t1 < M) && (t2 >= 0) && (t2 < N)) { // Don't allow out-of-bounds elements.
                                    sum += A1(cur_batch, d, t1, t2)*W(k, d, i,j);
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


void convolve_3d_filter_with_bias_minibatch_optimized(MatrixF& Z2, const MatrixF& W, const MatrixF& bias, const MatrixF& A1, MatrixF& temp_Z2, MatrixF& temp_A1, MatrixF& temp_W) {
    // for i in [0,...,minibatch_size).
    // Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.

    const int R = W.extent(0);
    const int D = W.extent(1);
    const int P = W.extent(2);
    const int Q = W.extent(3);
    const int pad_height = P/2; // offset for roughly equal zero padding. Conv filter size should be odd.
    const int pad_width = Q/2; // offset for roughly equal zero padding. Conv filter size should be odd.

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
        error_exit("convolve_3d_filter_with_bias_minibatch_optimized: Bad parameter sizes.");
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
    // Write the bias value in the last row;
    for (int k = 0; k < R; ++k) {
        temp_W(D*P*Q, k) = bias(k);
    }

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
                            int t1 = r - i + pad_height;
                            int t2 = c - j + pad_width;
                            if ((t1 >= 0) && (t1 < M) && (t2 >= 0) && (t2 < N)) { // Don't allow out-of-bounds elements.
                                temp_A1(row_ind, col_ind) = A1(cur_batch, d, t1, t2);
                            } else {
                                temp_A1(row_ind, col_ind) = 0.0f;
                            }
                            ++col_ind;
                        }
                    }
                }
                // Write a 1 for the bias term.
                temp_A1(row_ind, col_ind) = 1.0f;
            }
        }
    }

    // Now compute temp_Z2 = temp_A1 * temp_W using BLAS sgemm:
    mat_multiply(temp_Z2, temp_A1, temp_W);
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
}


void compute_convolutive_deltas_minibatch(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2) {
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
        error_exit("compute_convolutive_deltas_minibatch(): wrong dimensions.");
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


void compute_3d_convolutive_deltas_minibatch(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2) {
    // model: Z2(i'th minibatch) = W (convolve with) A1(i'th minibatch)
    const int R = W.extent(0);
    const int D = W.extent(1);
    const int P = W.extent(2);
    const int Q = W.extent(3);
    const int pad_height = P/2; // offset for roughly equal zero padding. Conv filter size should be odd.
    const int pad_width = Q/2; // offset for roughly equal zero padding. Conv filter size should be odd.
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
        error_exit("compute_convolutive_deltas_minibatch(): wrong dimensions.");
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
                                int t1 = r + i - pad_height;
                                int t2 = c + j - pad_width;
                                if ((t1 >= 0) && (t1 < M) && (t2 >= 0) && (t2 < N)) { // Don't allow out-of-bounds elements.
                                    sum += deltas_Z2(cur_batch, k, t1, t2)*W(k, d, i,j);
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


void compute_3d_convolutive_deltas_minibatch_optimized(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2,
                                                       MatrixF& temp_deltas_Z2, MatrixF& temp_deltas_A1, MatrixF& temp_W)  {
    // model: Z2(i'th minibatch) = W (convolve with) A1(i'th minibatch)
    const int R = W.extent(0);
    const int D = W.extent(1);
    const int P = W.extent(2);
    const int Q = W.extent(3);
    const int pad_height = P/2; // offset for roughly equal zero padding. Conv filter size should be odd.
    const int pad_width = Q/2; // offset for roughly equal zero padding. Conv filter size should be odd.
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
        error_exit("compute_convolutive_deltas_minibatch(): wrong dimensions.");
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
                            int t1 = r - i + pad_height;
                            int t2 = c - j + pad_width;
                            if ((t1 >= 0) && (t1 < M) && (t2 >= 0) && (t2 < N)) { // Don't allow out-of-bounds elements.
                                deltas_A1(cur_batch, d, t1, t2) += temp_deltas_A1(row_ind, col_ind);
                            }
                            ++col_ind;
                        }
                    }
                }
            }
        }
    }
}


void compute_convolutive_deltas_minibatch_optimized(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2,
                                                    MatrixF& temp_deltas_Z2, MatrixF& temp_deltas_A1, MatrixF& temp_W) {
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
        error_exit("compute_convolutive_deltas_minibatch_optimized(): wrong dimensions.");
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
}



void compute_weight_grad_convolutive_minibatch(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1) {
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
        error_exit("compute_weight_grad_convolutive_minibatch(): wrong dimensions.");
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
                                sum += deltas_Z2(cur_batch, k, r + i, c + j)*A1(cur_batch, i,j);
                            }
                        }
                    }
                }
                grad_W(k, r,c) = sum;
            }
        }
    }
}


void compute_3d_weight_grad_convolutive_minibatch(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1, bool accumulate)  {
    // convolutive model:  Z2 = W (convolve with) A1
    const int R = grad_W.extent(0);
    const int D = grad_W.extent(1);
    const int P = grad_W.extent(2);
    const int Q = grad_W.extent(3);
    const int pad_height = P/2; // offset for roughly equal zero padding. Conv filter size should be odd.
    const int pad_width = Q/2; // offset for roughly equal zero padding. Conv filter size should be odd.

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
        error_exit("compute_weight_grad_convolutive_minibatch(): wrong dimensions.");
    }
    const int M = deltas_Z2.extent(2);
    const int N = deltas_Z2.extent(3);
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
                                int t1 = r + i - pad_height;
                                int t2 = c + j - pad_width;
                                if ((t1 < M) && (t1 >= 0) && (t2 < N) && (t2 >= 0)) { // Don't allow out-of-bounds elements.
                                    sum += deltas_Z2(cur_batch, k, t1, t2)*A1(cur_batch, d, i,j);
                                }
                            }
                        }
                    }
                    if (accumulate) {
                        grad_W(k, d, r,c) += sum;
                    } else {
                        grad_W(k, d, r,c) = sum;
                    }
                }
            }
        }
    }
}

// deprecated?
void compute_weight_grad_convolutive_minibatch_optimized(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1,
                                                         MatrixF& temp_deltas_Z2, MatrixF& temp_A1,
                                                         MatrixF& temp_grad_W) {

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
        error_exit("compute_weight_grad_convolutive_minibatch(): wrong dimensions.");
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
}


void compute_3d_weight_grad_convolutive_minibatch_optimized(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1,
                                                            MatrixF& temp_deltas_Z2, MatrixF& temp_A1,
                                                            MatrixF& temp_grad_W, bool accumulate)   {
    // convolutive model:  Z2 = W (convolve with) A1
    const int R = grad_W.extent(0);
    const int D = grad_W.extent(1);
    const int P = grad_W.extent(2);
    const int Q = grad_W.extent(3);
    const int pad_height = P/2; // offset for roughly equal zero padding. Conv filter size should be odd.
    const int pad_width = Q/2; // offset for roughly equal zero padding. Conv filter size should be odd.
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
        error_exit("compute_weight_grad_convolutive_minibatch(): wrong dimensions.");
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
                            int t1 = r - i + pad_height;
                            int t2 = c - j + pad_width;
                            if ((t1 < M) && (t1 >= 0) && (t2 < N) && (t2 >= 0)) { // Don't allow out-of-bounds elements.
                                temp_A1(row_ind, col_ind) = A1(cur_batch, d, t1, t2);
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

    if (accumulate) {
#pragma omp parallel for collapse(3)
        for (int k = 0; k < R; ++k) {
            // For each of the R convolutional filters:
            for (int d = 0; d < D; ++d) {
                for (int i = 0; i < P; ++i) {
                    for (int j = 0; j < Q; ++j) {
                        grad_W(k, d, i,j) += temp_grad_W(d*P*Q + i*Q + j, k);
                    }
                }
            }
        }
    } else {
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
    }

}



void compute_bias_grad_convolutive_minibatch(MatrixF& grad_bias, const MatrixF& deltas_Z2, bool accumulate) {
    if ((grad_bias.order() != 1) || (grad_bias.size() != deltas_Z2.extent(1))) {
        grad_bias.resize(deltas_Z2.extent(1));
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
                    sum += deltas_Z2(cur_batch, k, r, c);
                }
            }
        }
        if (accumulate) {
            grad_bias[k] += sum;
        } else {
            grad_bias[k] = sum;
        }
    }
}


void multi_dim_minibatch_to_column_minibatch(MatrixF& A, const MatrixF&B) {
    const int combine_dims = B.order() -1;
    const int minibatch_size = B.extent(0);
    if (combine_dims == 2) {
        int P = B.extent(1)*B.extent(2);
        if (A.size() != B.size()) {
            A.resize(P, minibatch_size);
        }
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
        int P = B.extent(1)*B.extent(2)*B.extent(3);
        if (A.size() != B.size()) {
            A.resize(P, minibatch_size);
        }
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
        error_exit("multi_dim_minibatch_to_column_minibatch(): This size is not supported yet. Sorry.");
    }
}


void column_minibatch_to_multi_dim_minibatch(const MatrixF& A, MatrixF&B) {
    bool bad_parameters = false;
    if (A.size() != B.size()) {
        bad_parameters = true;
    }
    const int minibatch_size = A.extent(1);
    if (minibatch_size != B.extent(0)) {
        bad_parameters = true;
    }
    if (bad_parameters) {
        error_exit("multi_dim_minibatch_to_column_minibatch(): bad parameters.");
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
        error_exit("multi_dim_minibatch_to_column_minibatch(): This size is not supported yet. Sorry.");
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Convolution operations for factor models/deconvolutional networks
//
// Not included in the release version :)



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tensor operations and math functions


float relative_error(const MatrixF& A, const MatrixF& B) {
    if (A.size() != A.size()) {
        error_exit("relative_error(): Must be same sizes.");
    }
    // Use method from: http://cs231n.github.io/neural-networks-3/
    //
    // |A - B|
    // ---------
    // max(|A|, |B|)
    //
    float numer = 0.0f; // L2_norm(a - b)
    float a_norm = 0.0f;
    float b_norm = 0.0f;
    for (int i = 0; i != A.size(); ++i) {
        numer += (A[i] - B[i])*(A[i] - B[i]);
        a_norm += A[i]*A[i];
        b_norm += B[i]*B[i];
    }
    numer = sqrt(numer);
    a_norm = sqrt(a_norm);
    b_norm = sqrt(b_norm);
    if ((a_norm == 0) && (b_norm == 0)) {
        return 0.0f;
    } else {
        float res = numer/std::max(a_norm, b_norm);
        return res;
    }
}


void threshold_lower(MatrixF& X, float min_val) {
#pragma omp parallel for
    for (int i = 0; i < X.size(); ++i) {
        X[i] = max(X[i], min_val);
    }
}

void threshold_upper(MatrixF& X, float max_val)  {
#pragma omp parallel for
    for (int i = 0; i < X.size(); ++i) {
        X[i] = min(X[i], max_val);
    }
}




void normalize_columns_unit_sum(MatrixF& X) {
    for (int c = 0; c < X.extent(1); ++c) {
        float sum = 0.0f;
        for (int r = 0; r < X.extent(0); ++r) {
            sum += X(r,c);
        }
        if (sum > 0.0f) {
            const float scale = 1.0f/sum;
            for (int r = 0; r < X.extent(0); ++r) {
                X(r,c) *= scale;
            }
        }
    }
}

void normalize_columns_unit_max_val(MatrixF& X) {
    for (int c = 0; c < X.extent(1); ++c) {
        float maxv = X(0, c);
        for (int r = 1; r < X.extent(0); ++r) {
            if (X(r,c) > maxv) {
                maxv = X(r,c);
            }
        }
        if (maxv > 0.0f) {
            const float scale = 1.0f/maxv;
            for (int r = 0; r < X.extent(0); ++r) {
                X(r,c) *= scale;
            }
        }
    }
}



}
