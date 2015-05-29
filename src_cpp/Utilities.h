#ifndef _UTILITIES_H
#define	_UTILITIES_H
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
#include <iostream>
#include <vector>
#include "Matrix_list.h"
namespace kumozu {

    /*
     * Compute B x C and place the result in A. Note: all three matricies must
     * have already been allocated.
     *
     * Note: This method implements the basic easy-to-understand version. It is
     * not optimized in any way.
	 *
	 * Parameters
     *
     * A The result is returned in this matrix
     * B Input matrix which is not modified.
     * C Input matrix which is not modified.
     */
    void mat_multiply_naive(Matrix& A, const Matrix &B, const Matrix &C);



    /*
     * Compute B x C and place the result in A. Note: all three matricies must
     * have already been allocated.
     *
     * The implementation should call an optimized algorithm if one is available.
     *
     * A The result is returned in this matrix
     * B Input matrix which is not modified.
     * C Input matrix which is not modified.
     */
     void mat_multiply(Matrix& A, const Matrix& B, const Matrix& C);

	 /*
	  * Compute A = alpha*B*C + beta*A.
	  *
	  * Calls an optimized implementation, if possible.
	  */
	 void mat_multiply(Matrix& A, const Matrix& B, const Matrix& C, float alpha, float beta);

	 /*
	  * Compute A = B*C
	  */
	 void mat_multiply_blas(Matrix& A, const Matrix &B, const Matrix &C);

	 /*
	  * Compute A = alpha*B*C + beta*A
	  */
     void mat_multiply_blas(Matrix& A, const Matrix &B, const Matrix &C, float alpha, float beta);


    /*
     * Compute the element-wise product of B and C and then place the result in A.
     * Note: these matricies must have already been allocated and must have the
     * same dimensions. Otherwise, a runtime exception will occur.
     *
     * @param A Result is returned in this matrix.
     * @param B Input matrix which is not modified.
     * @param C Input matrix which is not modified.
     */
    void element_wise_multiply(Matrix& A, const Matrix &B, const Matrix &C);

	/*
	 * Set all values to be uniformly disstributed random values in [min, max].
	 */
	void randomize_uniform(Matrix& A, float min, float max);

	/*
	 * Set all values to be normally disstributed random values with "mean"
	 * and "std_deviation".
	 */
	void randomize_normal(Matrix& A, float mean, float std_deviation);

    /*
     * Compute the element-wise division B / C and then place the result in A.
     * Note: these matricies must have already been allocated and must have the
     * same dimensions. Otherwise, a runtime exception will occur.
     *
     * Specifically, compute:
     *
     *      B + epsilon
     * A = ------------
     *      C + epsilon
     *
     * @param A The result is returned in this matrix.
     * @param B Input matrix which is not modified.
     * @param C Input matrix which is not modified.
     * @param epsilon A positive constant that is added to both the numerator and denominator.
     */
    void element_wise_divide(Matrix& A, const Matrix &B, const Matrix &C, const float epsilon);

    /*
     * Compute the element-wise difference (B-C) and then place the result in A.
     * Note: these matricies must have already been allocated and must have the
     * same dimensions. Otherwise, a runtime exception will occur.
     * @param A The result is returned in this matrix.
     * @param B Input matrix which is not modified.
     * @param C Input matrix which is not modified.
     */
    void element_wise_difference(Matrix& A, const Matrix &B, const Matrix &C);


    /*
     * Compute the element-wise square B.^2 and then place the result in A.
     * Note: A and B must have already been initialized. It is allowed for A
     * and B to refer to the same object.
     * @param A Output matrix
     * @param B Input matrix which is not modified.
     */
    void element_wise_square(Matrix& A, const Matrix &B);

    /*
     * Compute the sum of all elements in <i>A</i> and return it.
     * @param A The input matrix.
     * @return The sum of all elements in A
     */
    float sum(Matrix& A);

    /*
     * Compute the transpose of B and put the result in A.
     * Both A and B must have already been initialized to consitent dimensions.
     *
     * @param A The result matrix.
     * @param B Input matrix which is not modified.
     */
    void transpose(Matrix& A, const Matrix &B);

    /*
     * Set all elements of the matrix <i>A</i> to have value <i>value</i> and
     * return the result in <i>A</i>.
     * @param A
     * @param value
     */
	template <typename T>
    void set_value(MatrixT<T>& A, T value) {
		#pragma omp parallel for
		for (int i = 0; i < A.size(); i++) {
			A[i] = value;
		}
	}

    /*
     * Take the element-wise square root of <i>A</i> and return the result in A.
     * A must have already been initialized.
     *
     * @param A
     *
     */
    void square_root(Matrix& A);

     /*
     * Add the scalar value b to each element of matrix A.
     *
     */
    void add_scalar(Matrix& A, float b);

    /*
     * Multiply each element of B by scaleFactor and put the result in A.
     * A and B are allowed to refer to the same matrix. A and B must have the
     * same dimensions. Otherwise, an exit error will occur.
     *
     * A <- scaleFactor*B
     *
     * @param A
     * @param B Input matrix which is not modified unless it is that same matrix as A..
     *
     */
    void scale(Matrix& A, const Matrix &B, float scaleFactor);


    /*
     * Compute the element-wise sum (B+C) and then place the result in A.
     *
     * A <- B + C
     *
     * Note: these matricies must have already been allocated and must have the
     * same dimensions. Otherwise, a runtime exception will occur.
     * @param A The result is returned in this matrix.
     * @param B Input matrix which is not modified.
     * @param C Input matrix which is not modified.
     */
    void element_wise_sum(Matrix& A, const Matrix &B, const Matrix &C);

    /*
     * Copy the contents of matrix B into matrix A.
     *
     * @param A The result is returned in this matrix. This matrix must have
     * already been allocated and must by the same size as B.
     * @param B Input matrix which is not modified.
     */
    void copy_matrix(Matrix& A, const Matrix& B);

	/*
	* Check that the supplied matrices have dimensions that are compatible with the factorization:
	*
	* X approx= W * H.
	*
	* If they are not compatible, exit with an error.
	*/
	void check_matrix_factorization_dimensions(const Matrix& X, const Matrix& W, const Matrix& H);


	/*
	 * Check that both matrices have the same dimensions. If they differ, exit with an error.
	 */
	template <typename T1, typename T2>
	void check_dimensions(const MatrixT<T1>& A, const MatrixT<T2>& B) {
		if (A.order() != B.order()) {
			std::cerr << "Error: Supplied matrices do not have the same order!" << std::endl;
			exit(1);
		}
		for (int i = 0; i != A.order(); ++i) {
			if (A.extent(i) != B.extent(i)) {
				std::cerr << "Error: Supplied matrices do not have the same extent for some dimension!" << std::endl;
				exit(1);
			}
		}
	}



	/*
	* Check that dimensions are consistant with A = B^T * C.
	*
	* If dimensions are consistant, return true. Otherwise, return false.
	*/
	bool check_dimensions_a_eq_b_tran_times_c(const Matrix& A, const Matrix& B, const Matrix& C);

	/*
	* Check that dimensions are consistant with A = B * C^T.
	*
	* If dimensions are consistant, return true. Otherwise, return false.
	*/
	bool check_dimensions_a_eq_b_times_c_tran(const Matrix& A, const Matrix& B, const Matrix& C);


	/*
	* Compute B^T x C and place the result in A. Note: all three matricies must
	* have already been allocated.
	*
	*
	* @param A The result is returned in this matrix
	* @param B Input matrix which is not modified. Note that the transpose of this matrix is used in the
	* multiplication.
	* @param C Input matrix which is not modified.
	*/
	void mat_multiply_left_transpose(Matrix& A, const Matrix& B, const Matrix& C);

	// Slow version for verifying correctness.
	void mat_multiply_left_transpose_naive(Matrix& A, const Matrix& B, const Matrix& C);

	/*
	* Compute B x C^T and place the result in A. Note: all three matricies must
	* have already been allocated.
	*
	* @param A The result is returned in this matrix
	* @param B Input matrix which is not modified.
	* @param C Input matrix which is not modified. Note that the transpose of this matrix is used in the
	* multiplication.
	*/
	void mat_multiply_right_transpose(Matrix& A, const Matrix& B, const Matrix& C);

	// Slow version for verifying correctness.
	void mat_multiply_right_transpose_naive(Matrix& A, const Matrix& B, const Matrix& C);

	/*
	* Return the test error as 1.0 - fraction correct, assuming binary-valued output data.
	* For each column n of network_output, compute the row corresponding to the maximum value and
	* then check if true_output(n) contains the same row index. If so, the network output at column
	* n is considered correct. Otherwise, it is considered an error.
	*
	* network_output: An M x N matrix. M is the number of class labels and N is the number of
	*                 output cases. Ideally, exactly one output class should be chosen and therefore
	*                 for each column, the correct row should have value = 1 and all other rows should
	*                 be equal to 0.
	*
	* true_output: array of class labels. A length N array. For each element, the value is an integer in 
	*              the range [0, N).
	*/
	float compute_test_error(const Matrix& network_output, const std::vector<float>& true_output);

	/*
	* Compute the maxout of "in_mat" and return the result in "out_mat" and return the corresponding
	* indices of the maximum elements in "maxout_indices_mat".
	*
	* For each column of in_mat, the maximum value is taken of each consecutive K rows and the result
	* is written into each consecutive row of out_mat. Therefore, it is required that in_mat have
	* size M x P and out_mat have size N x P where K = M/N is an integer. That is, N divides evenly
	* into M. The size of "maxout_indices_mat" must be the same as "maxout_values_mat".
	*
	* Return the result in "out_mat". If the matrices have inconsistent sizes, exit with an error.
	* fixme: change names to out_mat and out_state.
	*/
	void compute_forward_maxout(const Matrix& in_mat, Matrix& maxout_values_mat, MatrixT<int>& maxout_indices_mat);

	/*
	* For each element in maxout_values_mat, update the corresponding
	* value in input_mat.
	*
	* In this version, all elements of in_mat are updated. The elements of in_mat that were chosen as a "max value" are
	* updated with the corresponding max value. Also, all other elements of in_mat are set to 0.
	*/
	void compute_reverse_maxout_with_zeros(Matrix& in_mat, const Matrix& maxout_values_mat, const MatrixT<int>& maxout_indices_mat);


	/*
	 * Compute the forward-direction k-max operation independently on the partitioned columns of kmax_in.
	 *
	 * Intiution: This activation function corresponds to "forced sparsity" where the sparsity ratio is
	 * M/(partition_count*k). This type of sparsity corresponding to threshlding all but the largest k
	 * values in each partition region to 0. A given column (corresponding to 1 training or test sample)
	 * can be partitioned into 1 or more partition regions, specified by partition_count.
	 * Optionally, this function can be followed by a sub-sampling (i.e., pooling) operation such as max-pooling.
	 *
	 * kmax_in: An M x N matrix. The n'th column is assumed to contain the intput to the activation function (
	 * that is, this function) for
	 * the n'th example. Thus, each column as dimension M. We partition each column into partition_count partitions.
	 * M/partition_count must be an integer or else the program will exit with an error. 
	 * Within the p'th partition, we find the k largest values, storing the corresponding indices in
	 * kmax_out_indices and storing the corresponding values in kmax_out_values. All values other than
	 * the largest k values will be set to 0 in kmax_out_values.
	 * The value k of course must not be larger than the ratio M/partition_count.
	 *
	 * kmax_out_values: Same size as kmax_in (i.e., an M x N matrix). Within each partition, only the largest
	 *    k values are stored and all other values are set to 0.
	 *
	 * kmax_out_indices: A k*partition_count x N matrix. rmax = kmax_out_indices(r, n) contains the row index in both
	 * kmax_in and kmax_out_values of one of the k largest values. That is kmax_in(rmax, n) will then correspond
	 * to one of the k largest values.
	 *
	 * partition_count: The number of partitions for each column of kmax_in. All partitions are the same size:
	 *   parition_size = M/partition_count.
	 *   The partition_count values must be in [1...M].
	 *
	 * k: The number of largest values to keep from each column partition. Must be in the range [1...M/partition_count].  
	 *
	 * It may be interesting to feed the outptut from kmax_out_values into a sub-sampling layer.
	 */
	void compute_forward_kmax(const Matrix& kmax_in, Matrix& kmax_out_values, MatrixT<int>& kmax_out_indices,
							  int partition_count, int k);

	// Same as compute_forward_kmax() except kmax_out_indices is same size as kmax_in.
	void compute_forward_kmax_v2(const Matrix& kmax_in, Matrix& kmax_out_values, MatrixT<int>& kmax_out_indices,
							  int partition_count, int k);

	/*
	 * Same as compute_forward_kmax() except that kmax_in is updated using the values in kmax_out_values and
	 * kmax_out_indices.
	 *
	 * When updating kmax_in, all values that do not correspond to one of the k largest values are set to 0.
	 * Thus, all values will be overwritten, either with 0 or with a largest value from kmax_out_values.
	 */
	void compute_reverse_kmax(Matrix& kmax_in, const Matrix& kmax_out_values, const MatrixT<int>& kmax_out_indices,
							  int partition_count, int k);

	// fixme: change name: kmax_out_indices -> kmax_state.
	// Same as compute_reverse_kmax() except kmax_out_indices is same size as kmax_in.
	void compute_reverse_kmax_v2(Matrix& kmax_in, const Matrix& kmax_out_values, const MatrixT<int>& kmax_out_indices,
							  int partition_count, int k);

	/*
	 * Compute the forward-direction k-max operation independently on partitioned 3d sub-regions (boxes) of kmax_in.
	 *
	 * kmax_in is a matrix of size (minibatch_size x depth x height x width). kmax_in can be thought of as the collection
	 * of 3D (depth x height x width) matrices corresponding to one mini-batch of data. Each of these 3D matrices is
	 * partitioned into sub-matrices of size (depth/partitions_depth x height/partitions_height x width/partitions_width).
	 * For each sub-matrix, only the maximum k values are retained; all others are set to zero in the output matrix kmax_out.
	 * Thus, kmax_out is a sparse matrix of the same size as kmax_in. The kmax_state matrix is used to store the locations
	 * of the k elements in each submatrix that are retained, as this information will be needed later when the
	 * reverse-direction version of this function is called.
	 *
	 * Intiution: This activation function corresponds to "forced sparsity" where the sparsity ratio is
	 * (depth*height*width)/(partition_count*k). This type of sparsity corresponding to threshlding all but the largest k
	 * values in each partition region to 0.
	 *
	 * Parameters:
	 *
	 * kmax_in: (minibatch_size x depth x height x width) matrix containing the input values.
	 *
	 * kmax_out: (minibatch_size x depth x height x width) matrix containing the output values.
	 *
	 * kmax_state: (minibatch_size x depth x height x width) matrix containing the state information.
	 */
	void compute_forward_3d_kmax(const Matrix& kmax_in, Matrix& kmax_out, MatrixT<int>& kmax_state,
								 int box_depth, int box_height, int box_width, int k);

	/*
	 * This is the reverse-direction version of compute_forward_3d_kmax().
	 *
	 * Update the values in kmax_in given the values in kmax_out and the state information in kmax_state.
	 *
	 */
	void compute_reverse_3d_kmax(Matrix& kmax_in, const Matrix& kmax_out, const MatrixT<int>& kmax_state);

	/*
	 * Compute the forward-direction ReLU (Rectified Linear Unit) activation function on the input matrix "in_vals".
	 *
	 * The output values are placed into the matrix "out_vals" and the corresponding index information
	 * is placed into "out_indices".
	 *
	 * The function computed is out_vals[i] = max(0, in_vals[i]) for all indices i in the backing array.
	 * This function also updates out_indices so that
	 * out_indices[i] = 1 if out_vals[i] > 0. Otherwise, out_indices[i] = 0. The "out_indices" matrix
	 * will be needed by the function compute_reverse_relu().
	 *
	 * All supplied matrices should have the same size. Otherwise the program will exit with an error.
	 *
	 * Parameters:
	 *
	 * in_vals: An N-dimensional matrix.
	 *
	 * out_vals: An N-dimensional matrix of the same size as in_vals.
	 *
	 * out_indices: An N-dimensional matrix of the same size as in_vals.
	 */
	void compute_forward_relu(const Matrix& in_vals, Matrix& out_vals, MatrixT<int>& out_indices);

	void compute_forward_leaky_relu(const Matrix& in_vals, Matrix& out_vals, MatrixT<int>& out_indices);

	/*
	 * The identify function activation, forward direction.
	 *
	 * Used for gradient checking and debugging.
	 */
	void compute_forward_identity_activation(const Matrix& in_vals, Matrix& out_vals, MatrixT<int>& out_indices);

	/*
	 * The identify function activation, reverse direction.
	 *
	 * Used for gradient checking and debugging.
	 */
	void compute_reverse_identity_activation(Matrix& in_vals, const Matrix& out_vals, const MatrixT<int>& out_indices);

	/*
	 * Same as compute_forward_relu() except compute the reverse-direction ReLu to update "in_vals"
	 * from "out_vals" and "out_indices".
	 */
	void compute_reverse_relu(Matrix& in_vals, const Matrix& out_vals, const MatrixT<int>& out_indices);

	void compute_reverse_leaky_relu(Matrix& in_vals, const Matrix& out_vals, const MatrixT<int>& out_indices);

	// Todo: subsampling functions.
	

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	// Various SGD-related functions:


	/*
	* Given matrices X_error, W, H which are related according to
	*
	* X_pred = W * H + bias
	*
	* and
	*
	* X_error is the corresponding error matrix (deltas) from backpropagation
	* or some other method, compute the gradient matrix for W, W_grad.
	*
	* This computes grad_W = X_error*H^T / mini_batch_size
	*
	* This function does not compute the mean gradient. To get the mean gradient, you should do element-wise divide by the mini-batch size.
	*/
	void compute_weight_grad_sgd_minibatch(const Matrix& X_error, Matrix& W_grad, const Matrix& H);

	/*
	* Update the weights matrix W using the gradient W_grad and the learning rate alpha.
	*
	* W = W - alpha*W_grad
	*
	* is computed element-wise. 
	* 
	*/
	void update_weights_from_gradient(Matrix& W, const Matrix& W_grad, float alpha);

	/*
	* Update the weights matrix W using the gradient W_grad and the learning rate alpha.
	*
	* W = W - alpha*W_grad + ...
	*
	* is computed element-wise.
	* 
	*/
	void update_weights_from_gradient(Matrix& W, const Matrix& W_grad, float alpha, float lambda,
		float sparsity_param, bool force_nonnegative);

	/*
	* Update the weights matrix W using the gradient W_grad and the learning rate alpha.
	*
	* W = W + alpha*W_grad/sqrt(W_grad_mean_square)
	*
	* is computed element-wise.
	*
	* counter: Number of times this function has been called. Initial value on 1st call should be 1.
	*
	*/
	void update_weights_from_gradient_rmsprop_v2(Matrix& W, const Matrix& W_grad, Matrix& W_grad_sum_square, 
												 float alpha, float counter);


	void update_weights_from_gradient_rmsprop_v3(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
												 float alpha, float rho);

	void update_weights_from_gradient_rmsprop_kernel_ball_1(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
												 float alpha, float rho);

	/*
	 * Rmsprop with momentum.
	 */
	void update_weights_from_gradient_rmsprop_momentum(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
													   Matrix W_momentum, float alpha, float rho, float momentum);

	void update_weights_from_gradient_rmsprop_momentum_1d_kernel_ball(Matrix& W, const Matrix& W_grad, Matrix& W_grad_mean_square, 
													   Matrix W_momentum, float alpha, float rho, float momentum);

	void update_weights_from_gradient_adagrad(Matrix& W, const Matrix& W_grad, Matrix& W_grad_sum_square, 
												 float alpha);

	/*
	 * Gaussian kernel function.
	 */
	float gaussian_kernel(const std::vector<float>& x1, const std::vector<float>& x2, float sigma);




	// We compute H_error = W^T * X_error
	void do_backprop_update_sgd_minibatch(const Matrix& X_error, const Matrix& W, Matrix& H_error);

	/*
	* Compute the gradient of the bias vector and return it in b_grad.
	*/
	void compute_bias_grad_sgd_minibatch(const Matrix& X_error, Matrix& b_grad);

	/*
	* Use the current W and H to compute the product W*H and update X.
	*
	* Compute X = update_weight*W*H + (1 - udpate_weight)*X
	*
	* update_weight: Should be in the range (0, 1). It controls the relative weights of
	* the previous and new values. If 0, just use the old value. If 1, just use the
	* new value. If 0.1, use 0.1*new values + 0.9*old_value.
	*/
	void do_product_update_naive(Matrix& X, const Matrix& W, const Matrix& H, float update_weight);

	// Calls an optimized impelmentation.
	void do_product_update(Matrix& X, const Matrix& W, const Matrix& H, float update_weight);


	/*
	* Use the supplied W, H, and bias b to update X as
	*
	* X <= W * H 
	* and then add b to each column in X.
	*
	* The vector b must have size equal to the number of rows in X. That is, b is the same size as
	* a single column of X.
	*
	*/
	// todo: get rid of version that uses vector<float>.
	void do_product_update(Matrix& X, const Matrix& W, const Matrix& H, const std::vector<float>& b);

	// Slow but safe version. This version is needed to check that the result of the optimized version are correct.
	// todo: get rid of version that uses vector<float>.
	void do_product_update_naive(Matrix& X, const Matrix& W, const Matrix& H, const std::vector<float>& b);

	/*
	* Compute the root mean squared error (RMSE) for the factorization approximation
	*
	* X approx= W * H
	*
	* and return it.
	*
	* @return The current RMSE.
	*/
	float compute_reconstruction_error(const Matrix& X, const Matrix& W, const Matrix& H);

	/*
	* Given a matrix of error values, compute and return the RMSE.
	*/
	float compute_rmse(const Matrix& X);

	/*
	* Given two matrices, compute and return the RMSE of their element-wise differences.
	*/
	float compute_rmse(const Matrix& A, const Matrix& B);



	/*
	* Get the sub-matrix of B starting from column start_col with the same number of columns as A
	* and copy into A.
	*
	*
	* A and B must have the same number of rows. The number of columns copied is
	* equal to the number of columns in A. Thus, all data in A will be overwritten
	* by data in B.
	*
	* Parameters
	*
	* A: The output Matrix.
	*
	* B: The input Matirx.
	*
	* start_col: The starting column in B. Data is copied from columns [start_col, start_col + A.width). 
	*
	* Exit with an error if any of the following is found to occur:
	* 1. A and B do not have the same number of rows.
	* 2. start_col is not a valid column in B.
	* 3. There is insufficient data in B to fill A starting from the specified column in B.
	*/
	void get_sub_matrix(Matrix& A, const Matrix &B, int start_col);

	/*
	* Return (i.e., copy) the full contents of A into B starting from start_col in B.
	*
	* A and B must have the same number of rows. The number of columns copied is
	* equal to the number of columns in A. Thus, all data in A will be copied
	* into B.
	*
	* Parameters
	*
	* A: The input Matrix.
	*
	* B: The output Matirx.
	*
	* start_col: The starting column in B. Data is copied from A into columns [start_col, start_col + A.width)
	*	in B
	*
	* Exit with an error if any of the following is found to occur:
	* 1. A and B do not have the same number of rows.
	* 2. start_col is not a valid column in B.
	* 3. There is insufficient space in B to starting from the specified column in B.
	*/
	void return_sub_matrix(const Matrix& A, Matrix &B, int start_col);

	/*
	 * Hold the first dimension of B constant and copy all data for all values of remaining dimensions into A.
	 *
	 * This function is intended to be used for the case where B contains N samples such that each
	 * sample corresponds to a 3-dimensional matrix. This function copies the sample of interest
	 * into the supplied 3-dimensional matrix.
	 *
	 * Parameters:
	 *
	 * B: A N x dim1 x dim2 x dim3 matrix that contains N samples, each sample of dimensionality 
	 *   dim1 x dim2 x dim3.
	 *
	 * A: A dim1 x dim2 x dim3 matrix that can hold exactly one of the samples from B.
	 *
	 * sample_index: Specifies which of the N samples in B to copy into A.
	 *
	 * Exit with an error if inconsistant sizes are specified.
	 */
	void get_sample(Matrix& A, const Matrix& B, int sample_index);

	/*
	 * Sample as get_sample() except the data is copied from A to B.
	 */
	void return_sample(const Matrix& A, Matrix& B, int sample_index);

	/*
	* If abs(a -b) > tolerance, exit with an error.
	*/
	void assert_almost_equal(float a, float b, float tolerance);

	/*
	* For each element of the two suplied matrices, which must be of the same size,
	* test if the magnitude of the difference exceeds the tolerance. If so,
	* exit with an error.
	*/
	void assert_almost_equal(const Matrix& A, const Matrix& B, float tolerance);


	/*
	* Print out some basic statistics for the supplied matrix with the
	* supplied name. 
	*
	* The statistics include mean, std deviation, etc.
	*/
	void print_stats(const Matrix& A, std::string name);

	/*
	 * Same as convolve_2d_filter_with_bias() except operates on a mini-batch of data.
	 *
	 * That is, for i in [0,...,minibatch_size).
	 * Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
	 *
	 * Z2: minibatch_size x R x M x N output matrix.
	 * A1: minibatch_size x M x N input matrix
	 * W: R x P x Q input matrix containing R filters of size P x Q.
	 * bias: Size R matrix (1-dim of length R). 
	 */
	void convolve_2d_filter_with_bias_minibatch(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1); 

	/*
	 * Convolve several 3D filters with a 3D matrix.
	 *
	 * A1 contains "minibatch_size" 3D input images. For each of these input images, convolve with "R" 3D filters
	 * that are contained in W and bias and store the results in Z2. Although the convolution filters are 3D, the
	 * convolution is only performed along the second two dimensions (image height = P and image width = Q, not image depth = D).
	 *
	 * That is, for i in [0,...,minibatch_size).
	 * Compute Z2(i'th minibatch) = W (conv with) A1(i'th minibatch) + bias.
	 *
	 *
	 * Z2: minibatch_size x R x M x N output matrix.
	 * A1: minibatch_size x D x M x N input matrix
	 * W: R x D x P x Q input matrix containing R filters of size D x P x Q.
	 * bias: Size R matrix (1-dim of length R). 
	 *
	 * where
	 *
	 * minibatch_size = number of sample in one mini-batch.
	 * R = number of convolution filters.
	 * M = height of input image.
	 * N = width of input image.
	 * D = depth of input image = depth of convolution filter (for color images, D = 3, but for hidden layers, D can be much larger).
	 * P = height of convolution filter.
	 * Q = width of convolution filter.
	 */
	void convolve_3d_filter_with_bias_minibatch(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1); 

	/*
	 * Same as convolve_3d_filter_with_bias_minibatch() except this version is optimzied for speed. 
	 *
	 * This version uses BLAS matrix multiply (sgemm) to compute the convolution as described in:
	 * 
	 * K. Chellapilla, S. Puri, P. Simard, et al. High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition, 2006.
	 *
	 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.482&rep=rep1&type=pdf
	 *
	 * This version performs all of the convolutions in a mini-batch in a single matrix multiply.
	 *
	 * This version essentially converts the convolution into a matrix multiplication and therefore needs
	 * to make use of three extra temporarry matrices, which must be supplied to this function. They
	 * correspond to the matrix product: temp_Z2 = temp_A1 * temp_W.
	 * This bias terms are included in an extra row at the bottom of temp_W, which means we also need an extra
	 * column of 1's in the right-most column of temp_A1.
	 *
	 * temp_Z2: of size (M*N*minibatch_size) x R
	 *
	 * temp_A1: of size (M*N*minibatch_size) x (D*P*Q + 1)
	 *
	 * temp_W: of size (D*P*Q + 1) x R
	 *
	 * If any of the supplied matrices have inconsitent sizes, exit with an error.
	 */
	void convolve_3d_filter_with_bias_minibatch_optimized(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1, Matrix& temp_Z2, Matrix& temp_A1, Matrix& temp_W); 


	/*
	 * Same as convolve_2d_filter_with_bias_minibatch() except this version is optimzied for speed. 
	 *
	 * This version uses BLAS matrix multiply (sgemm) to compute the convolution as described in:
	 * 
	 * K. Chellapilla, S. Puri, P. Simard, et al. High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition, 2006.
	 *
	 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.482&rep=rep1&type=pdf
	 *
	 * This version performs all of the convolutions in a mini-batch in a single matrix multiply.
	 *
	 * This version essentially converts the convolution into a matrix multiplication and therefore needs
	 * to make use of three extra temporarry matrices, which must be supplied to this function. They
	 * correspond to the matrix product: temp_Z2 = temp_A1 * temp_W.
	 * This bias terms are included in an extra row at the bottom of temp_W, which means we also need an extra
	 * column of 1's in the right-most column of temp_A1.
	 *
	 * temp_Z2: of size (M*N*minibatch_size) x R
	 *
	 * temp_A1: of size (M*N*minibatch_size) x (P*Q + 1)
	 *
	 * temp_W: of size (P*Q + 1) x R
	 *
	 * If any of the supplied matrices have inconsitent sizes, exit with an error.
	 */
	void convolve_2d_filter_with_bias_minibatch_optimized(Matrix& Z2, const Matrix& W, const Matrix& bias, const Matrix& A1, Matrix& temp_Z2, Matrix& temp_A1, Matrix& temp_W); 


	/*
	 * Same as compute_convolutive_deltas() except operates on a mini-batch of data.
	 *
	 * This function performs the "update deltas" back-propagation step corresponding
	 * to the forward propagation performed by convolve_2d_filter_with_bias_minibatch().
	 *
	 * Parameters:
	 *
	 * deltas_A1: output matrix of same size as A1, which is minibatch_size x M x N.
	 *
	 * W: input R x P x Q matrix containing R filters of size P x Q.
	 *
	 * deltas_Z2: input matrix same size as Z2, which is minibatch_size x R x M x N.
	 *
	 */
	void compute_convolutive_deltas_minibatch(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2); 

	/*
	 *
	 * This function performs the "update deltas" back-propagation step corresponding
	 * to the forward propagation performed by convolve_3d_filter_with_bias_minibatch().
	 *
	 * Parameters:
	 *
	 * deltas_A1: output matrix of same size as A1, which is minibatch_size x D x M x N.
	 *
	 * W: input R x D x P x Q matrix containing R filters of size D x P x Q.
	 *
	 * deltas_Z2: input matrix same size as Z2, which is minibatch_size x R x M x N.
	 *
	 */
	void compute_3d_convolutive_deltas_minibatch(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2); 


	/*
	 * Same as compute_3d_convolutive_deltas_minibatch() except this version is optimized for speed at the
	 * expense of increased memory usage.
	 *
	 * This version uses BLAS matrix multiply (sgemm) to compute the convolution as described in:
	 * 
	 * K. Chellapilla, S. Puri, P. Simard, et al. High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition, 2006.
	 *
	 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.482&rep=rep1&type=pdf
	 *
	 * temp_deltas_Z2: of size (M*N*minibatch_size) x R
	 *
	 * temp_deltas_A1: of size (M*N*minibatch_size) x (D*P*Q + 1)
	 *
	 * temp_W: of size (D*P*Q + 1) x R
	 *
	 * If any of the supplied matrices have inconsitent sizes, exit with an error.
	 */
	void compute_3d_convolutive_deltas_minibatch_optimized(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2, 
														   Matrix& temp_deltas_Z2, Matrix& temp_deltas_A1, Matrix& temp_W); 

	/*
	 * Same as compute_convolutive_deltas_minibatch() except this version is optimized for speed at the
	 * expense of increased memory usage.
	 *
	 * This version uses BLAS matrix multiply (sgemm) to compute the convolution as described in:
	 * 
	 * K. Chellapilla, S. Puri, P. Simard, et al. High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition, 2006.
	 *
	 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.482&rep=rep1&type=pdf
	 *
	 * temp_deltas_Z2: of size (M*N*minibatch_size) x R
	 *
	 * temp_deltas_A1: of size (M*N*minibatch_size) x (P*Q + 1)
	 *
	 * temp_W: of size (P*Q + 1) x R
	 *
	 * If any of the supplied matrices have inconsitent sizes, exit with an error.
	 */
	void compute_convolutive_deltas_minibatch_optimized(Matrix& deltas_A1, const Matrix& W, const Matrix& deltas_Z2,
														   Matrix& temp_deltas_Z2, Matrix& temp_deltas_A1, Matrix& temp_W); 



	/*
	 * Same as compute_weight_grad_convolutive() except operates on a mini-batch of data.
	 *
	 * This function performs the "update weight gradients" back-propagation step corresponding
	 * to the forward propagation performed by convolve_2d_filter_with_bias_minibatch().
	 *
	 * Parameters:
	 *
	 * grad_W: Input matrix of same size as W: R x P x Q matrix containing R filter gradient matrices of size P x Q.
	 *
	 * deltas_Z2: Input matrix of same size as Z2: minibatch_size x R x M x N.
	 *
	 * A1: Input matrix of size minibatch_size x M x N.
	 *
	 * Note: This function computes the actual gradient of W. However, for SGD updates, the average (mean) gradient
	 * is probably desired. The mean gradient can be obtained by scaling the returned
	 * gradient by 1/(deltas_Z2.dim0*deltas_Z2.dim1*minibatch_size).
	 *
	 */
	void compute_weight_grad_convolutive_minibatch(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1);

	/*
	 * This function performs the "update weight gradients" back-propagation step corresponding
	 * to the forward propagation performed by convolve_3d_filter_with_bias_minibatch().
	 *
	 * Parameters:
	 *
	 * grad_W: Input matrix of same size as W: R x D x P x Q matrix containing R filter gradient matrices of size D x P x Q.
	 *
	 * deltas_Z2: Input matrix of same size as Z2: minibatch_size x R x M x N.
	 *
	 * A1: Input matrix of size minibatch_size x D x M x N.
	 *
	 * Note: This function computes the actual gradient of W. However, for SGD updates, the average (mean) gradient
	 * is probably desired. The mean gradient can be obtained by scaling the returned
	 * gradient by 1/(deltas_Z2.dim0*deltas_Z2.dim1*minibatch_size).
	 *
	 */
	void compute_3d_weight_grad_convolutive_minibatch(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1);


	/*
	 * Same as compute_weight_grad_convolutive_minibatch() except this version is optimized for speed at the
	 * expense of increased memory usage.
	 *
	 * This version uses BLAS matrix multiply (sgemm) to compute the convolution as described in:
	 * 
	 * K. Chellapilla, S. Puri, P. Simard, et al. High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition, 2006.
	 *
	 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.482&rep=rep1&type=pdf
	 *
	 * temp_deltas_Z2: of size (M*N*minibatch_size) x R
	 *
	 * temp_A1: of size (M*N*minibatch_size) x (P*Q + 1)
	 *
	 * temp_W: of size (P*Q + 1) x R
	 *
	 * If any of the supplied matrices have inconsitent sizes, exit with an error.
	 */
	void compute_weight_grad_convolutive_minibatch_optimized(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1,
															 Matrix& temp_deltas_Z2, Matrix& temp_A1, 
															 Matrix& temp_grad_W);

	/*
	 * Same as compute_3d_weight_grad_convolutive_minibatch() except this version is optimized for speed at the
	 * expense of increased memory usage.
	 *
	 * This version uses BLAS matrix multiply (sgemm) to compute the convolution as described in:
	 * 
	 * K. Chellapilla, S. Puri, P. Simard, et al. High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition, 2006.
	 *
	 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.482&rep=rep1&type=pdf
	 *
	 * temp_deltas_Z2: of size (M*N*minibatch_size) x R
	 *
	 * temp_A1: of size (M*N*minibatch_size) x (D*P*Q + 1)
	 *
	 * temp_W: of size (D*P*Q + 1) x R
	 *
	 * If any of the supplied matrices have inconsitent sizes, exit with an error.
	 */
	void compute_3d_weight_grad_convolutive_minibatch_optimized(Matrix& grad_W, const Matrix& deltas_Z2, const Matrix& A1,
																Matrix& temp_deltas_Z2, Matrix& temp_A1, 
															 Matrix& temp_grad_W);


	/*
	 * Same as compute_weight_grad_convolutive() except operates on a mini-batch of data.
	 *
	 * This function performs the "update bias gradients" back-propagation step corresponding
	 * to the forward propagation performed by convolve_2d_filter_with_bias_minibatch().
	 *
	 * Parameters:
	 *
	 * deltas_Z2: Input matrix of same size as Z2: minibatch_size x R x M x N.
	 *
	 * grad_bias: R x 1 matrix. That is, same as a vector of length R.
	 *
	 * Before performing gradient updates, you should scale "grad_bias" from this function by
	 * 1/(minibatch_size*M*N) to get the average gradient.
	 */
	void compute_bias_grad_convolutive_minibatch(Matrix& grad_bias, const Matrix& deltas_Z2);

	/*
	 * Return a "closeness score" for the two supplied Matrices. The returned value will be close 
	 * to 0 when a and b are similar and will be greater than 0 the more a and b disagree.
	 *
	 * Specifically, the returned value is computed as
	 *
	 *            |a - b|
	 * score =  ----------------
	 *            |a| + |b|
	 * 
	 * where |.| denotes the L2 norm.
	 * This method of computing error between two vectors or Matrices is intended to be used when the dynamic range of the
	 * entires in a and b is unknown. As long as the magnitude of the differences between corresponding
	 * values of a and b are small compared to their individual magnitudes, the score will be
	 * close to 0.
	 * This method is intended for vectors that have mostly non-zero elements.
	 * If both "a" and "b" are equal to 0, return 0.
	 */
	float relative_error(const Matrix& a, const Matrix& b);




	/*
	 * Copy the contents of B into the specified column of A.
	 *
	 * The data is copied in the same order that it is stored in the backing array of B.
	 *
	 * The order of B (i.e., number of elements in B) must equal the number of rows in A.
	 */
	template <typename T>
	void copy_matrix_to_column(MatrixT<T>& A, const MatrixT<T>& B, int column) { 
		// Copy all data in B into column "column" of A.
		bool inconsistent_parameters = false;
		if (B.size() != A.extent(0)) {
			inconsistent_parameters = true;
		}
		if (column >= A.extent(1)) {
			inconsistent_parameters = true;
		}
		if (inconsistent_parameters) {
			std::cerr << "copy_matrix_to_column(): Inconsistent parameter values." << std::endl;
			exit(1);
		}
		for (int i = 0; i != B.size(); ++i) {
			A(i, column) = B[i];
		}
	}


	/*
	 * Copy the contents of the specified column of A into B.
	 *
	 * The data is copied in the same order that it is stored in the backing array of B.
	 *
	 * The order of B (i.e., number of elements in B) must equal the number of rows in A.
	 */
	template <typename T>
	void copy_column_to_matrix(const MatrixT<T>& A, MatrixT<T>& B, int column) {
		// Copy all data from column "column" of A into B.
		bool inconsistent_parameters = false;
		if (B.size() != A.extent(0)) {
			inconsistent_parameters = true;
		}
		if (column >= A.extent(1)) {
			inconsistent_parameters = true;
		}
		if (inconsistent_parameters) {
			std::cerr << "copy_column_to_matrix(): Inconsistent parameter values." << std::endl;
			exit(1);
		}
		for (int i = 0; i != B.size(); ++i) {
			B[i] = A(i, column);
		}
	}


	float max_value(const std::vector<float>& a);

	float min_value(const std::vector<float>& a);


	/*
	 * Perform max-out operation on a 3D matrix H.
	 *
	 * The matrix H is assumed to contain activations for a hidden layer.
	 *
	 * Typically, this corresponds to the model:
	 * X = W (convolve with) H.
	 *
	 * where X is an 2D image of size dim0 x dim1.
	 *
	 * We partition the inside of H into boxes of size 
	 * dim0_box x dim1_box x dim2_box. For each box independently, we keep only
	 * the maximum element and set all other elements to 0.
	 *
	 * The dimensions of the box do not need to evenly didvide into the corresponding
	 * dimensions of H.
	 *
	 * Parameters:
	 *
	 * H: Matrix of size dim0 x dim1 x dim2. 
	 *
	 * dim0_box: Size of the max-out box along dim0.
	 *
	 * dim1_box: Size of the max-out box along dim1.
	 *
	 * dim2_box: Size of the max-out box along dim2.
	 *
	 * This version does not perform any sub-sampling.
	 */
	void max_out_3d(Matrix& H, int dim0_box, int dim1_box, int dim2_box);

	/*
	 * Same as max_out_3d() except also performs spatial sub-sampling.
	 * The sub-sampling is performed in the first two dimensions of H only, since the 3rd
	 * dimension is the channel filter index.
	 *
	 * The sub-sample factor for dim0 is the ratio H.dim0/H_sub_sampled.dim0 which must be an integer.
	 * The sub-sample factor for dim1 is the ratio H.dim1/H_sub_sampled.dim1 which must be an integer.
	 *
	 * The sub-sampled output is placed into H_sub_sampled and the input "H" is modified.
	 */
	void max_out_sub_sampling_3d(Matrix& H_sub_sampled, Matrix& H, int dim0_box, int dim1_box, int dim2_box);

	/*
	 * B is of size dim0 x dim1 x dim2 x dim3 where dim0 is the number of samples (i.e., training or testing samples)
	 * and dim1 x dim2 x dim3 is a 3D matrix that contains a single sample. So,
	 * B(i,:,:,:) refers to the i'th sample where : means to range over all indices.
	 * 
	 * For each i, copy all values corresponding to ":" in B(i,:,:,:) into A(:,i). Thus, the i'th column
	 * of A will contain the 3D box from B for the i'th sample reshaped into a column vector.
	 */
	void reshape_3d_features_to_1d_features(Matrix& A, const Matrix& B);

	/*
	 * Compute the 3D maxout of A.
	 *
	 * Input matrix A has size dim0_A x dim1_A x dim2_A.
	 * Output matrix maxout_vals has size dim0_out x dim1_out x dim2_out.
	 * Output matrix maxout_indices has size dim0_out x dim1_out x dim2_out x 3.
	 *
	 * The corresponding maxout factors are then
	 *
	 * maxout_factor_dim0 = dim0_A/dim0_out
	 *
	 * maxout_factor_dim1 = dim1_A/dim1_out
	 *
	 * maxout_factor_dim2 = dim2_A/dim2_out
	 *
	 * There must be no remainder in the above divisions. Otherwise the program will exit with an error.
	 *
	 * Matrix A is then partitioned into little cubes of size maxout_factor_dim0 x maxout_factor_dim1 x maxout_factor_dim2.
	 * Inside each of these cubes, the maximum value is taken and stored in the corresponding location in maxout_vals.
	 * Note that A and maxout_vals are of such size that each cube in A maps to a single element of maxout_vals.
	 *
	 * For each maximum value stored in maxout_vals, the corresponding location of this maximum value in A is stored
	 * in maxout_indices. Suppose that for a given cube inside A, the maximum value is found to occur at
	 * A(max_ind0, max_ind1, max_ind2) and that this maximum value is stored into maxout_vals(i,j,k).
	 * Then we store the indices in maxout_indices as
	 *
	 * max_ind0 = maxout_indices(i,j,k,0)
	 * max_ind1 = maxout_indices(i,j,k,1)
	 * max_ind2 = maxout_indices(i,j,k,2)
	 * 
	 */
	void compute_maxout_3d(const Matrix& A, Matrix& maxout_vals, MatrixT<int>& maxout_indices);

	/*
	 * This is the reverse of compute_maxout_3d().
	 *
	 * First zero out A. Then for each element of maxout_vals, write the max value into the corresponding location
	 * in A, using the index information in maxout_indices.
	 */
	void compute_reverse_maxout_3d(Matrix& A, const Matrix& maxout_vals, const MatrixT<int>& maxout_indices);


	/*
	 * Perform max-pooling on "in_activations" to product "out_activations."
	 *
	 * The approximate stride is given as the ratio of the input_extents to the output_extents. For example,
	 * the stride in the height dimension is given as height_in/height_out.
	 *
	 * in_activations: Input activations of size (minibatch_size, depth_in, height_in, width_in).
	 *
	 * out_activations: Output activations of size (minibatch_size, depth_out, height_out, width_out).
	 *
	 * state: State of the pooling layer of size (minibatch_size, depth_out, height_out, width_out, 3).
	 *
	 * pooling_region_extents: Extents that specify the size of each pooling region. This must correspond to a 3-D matrix of
	 *                         size (pooling_size_depth, pooling_size_height, pooling_size_width).
	 */
	void forward_3d_max_pool(const Matrix& in_activations, Matrix& out_activations, MatrixT<int>& state, 
							 const std::vector<int>& pooling_region_extents);

	/*
	 * Sames as forward_3d_max_pool() but perform reverse max-pooling to update "in_activations"
	 * from "out_activations."
	 */
	void reverse_3d_max_pool(Matrix& in_activations, const Matrix& out_activations, const MatrixT<int>& state);

	float max_value(const Matrix& A);
	
	float min_value(const Matrix& A);

	/*
	 * Extract one mini-batch of data from B and copy it into A.
	 *
	 * Data is copied from B( [start_col, start_col + minibatch_size), :, :) 
	 * and placed into A.
	 *
	 * Parameters:
	 *
	 * A: A matrix of size (minibatch_size x M x N) or of size (minibatch_size x D x M x N)
	 *
	 * B A matrix of size (P x M x N) or of size (P x D x M x N) where P >= minibatch_size.
	 *
	 * start_col: Data is copied starting from B(start_col, :, :) or B(start_col, :, :, :) 
	 *
	 * minibatch_size: One past the ending first index of B.
	 *
	 * Exit with an error if any of the parameter values are found to be inconsistent.
	 */
	void extract_3d_minibatch(Matrix& A, const Matrix& B, int start_col, int minibatch_size);

	/*
	 * Convert between two different formats for a mini-batch of data.
	 *
	 * B is a multi-dimensional matrix of size minibatch_size x dim1 x dim2 ... dimR. This typically
	 * corresponds to one minibatch of output activations from a convolutional layer.
	 *
	 * This function simply copies data from B to A, converting between the too formats. Note that
	 * A and B must therefore contain the same number of elements.
	 *
	 * A: Size P x minibatch_size matrix. This typically corresponds to one mini-batch of input activations
	 * to a fully-connected layer.
	 *
	 * B: Size minibatch_size x dim1 x dim2 ... dimR matrix where P = (dim1 x dim2 ... dimR).
	 *
	 */
	void multi_dim_minibatch_to_column_minibatch(Matrix& A, const Matrix&B);

	/*
	 * Same as multi_dim_minibatch_to_column_minibatch() except copies in the opposite
	 * direction.
	 */
	void column_minibatch_to_multi_dim_minibatch(const Matrix& A, Matrix&B);


	/*
	 * For each element x_i of X, set x_i = max(x_i, min_val).
	 */
	void threshold_lower(Matrix& X, float min_val);

	/*
	 * For each element x_i of X, set x_i = min(x_i, min_val).
	 */
	void threshold_upper(Matrix& X, float max_val);

	
	/*
	 * Given a length-M vector of N distinct class labels, return 
	 * a N x M matrix.
	 *
	 * Return a N x M matrix X where X(i,j) is 1 iff labels[j] = i.
	 */
	// todo: flot -> int
	Matrix labels_to_mat(const std::vector<float>& labels);
	
}

#endif	/* _UTILITIES_H */
