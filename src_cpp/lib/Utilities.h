#ifndef _UTILITIES_H
#define _UTILITIES_H
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
#include <iostream>
#include <vector>
#include "Matrix_list.h"
#include <algorithm>
#include "Assertions.h"
#include "Variable.h"

namespace kumozu {

//
// Conventions:
//
// Input matrices to functions are specified as const references. Output matrices are specified as (non-const) references.
//
// Automatic resizing of output matrices:
//
// Most (eventually all) functions that modify a matrix will resize the matrix to the appropriate dimensions, if necessary.
// This feature makes these functions easier to use because the user is releived from the burden of having to
// determine and set the appropriate dimensions before the function call. For example, to compute the matrix product:
//
// A <- B x C
//
// The user can simply allocate an empty Matrix A such as
//
// Matrix A;
// mat_multiply(A, B, C); // Assume B and C are already available and filled with data.
//
// The matrix A will be empty (size 0) when passed in to the function, but will be resized to the appropriate
// dimensions during the function call.
//
// Other functions that take a modifiable matrix reference generally behave similarly.
//
// Since resizing a matrix can be expensive, it is good practice to write code such that resizing is only performed during
// the "initialization" stage. For debugging purposes, these functions can be configured to print a "resized" message to
// stdout whenever a matrix is resized by defining KUMOZU_DEBUG in the makefile.


// todo: remove "compute" in all function names since this is already assumed.

////////////////////////////////////////////////////////////////////////////////////////////
// Matrix Utilities

/**
 * Print all elements in the vector to std out.
 */
template <typename T>
void print_vector(std::vector<T> vec) {
    for_each(begin(vec), end(vec), [] (T val) {
        std::cout << val << " ";
    });
    std::cout << std::endl;
}

/**
 * These "looper" functions take a vector of extents and apply the supplied function "func()" to
 * each combination of values in the extent. The supplied function should take only integer
 * arguments and the argument count must be equal to the length of "extents".
 *
 * The length of the supplied "extents" vector is the same as "x" in
 * looper_<x>_dim().
 *
 * The function is potentially applied in parallel, and so there must be no inter-loop dependencies.
 * Otherwise, the results are undefined.
 */
template <typename Func>
void looper_1_dim(const std::vector<int>& extents, Func func) {
#pragma omp parallel for
    for (int i = 0; i < extents[0]; ++i) {
        func(i);
    }
}


/**
 * These "looper" functions take a vector of extents and apply the supplied function "func()" to
 * each combination of values in the extent. The supplied function should take only integer
 * arguments and the argument count must be equal to the length of "extents".
 *
 * The length of the supplied "extents" vector is the same as "x" in
 * looper_<x>_dim().
 *
 * The function is potentially applied in parallel, and so there must be no inter-loop dependencies.
 * Otherwise, the results are undefined.
 */
template <typename Func>
void looper_2_dim(const std::vector<int>& extents, Func func) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < extents[0]; ++i) {
        for (int j = 0; j < extents[1]; ++j) {
            func(i,j);
        }
    }
}


/**
 * These "looper" functions take a vector of extents and apply the supplied function "func()" to
 * each combination of values in the extent. The supplied function should take only integer
 * arguments and the argument count must be equal to the length of "extents".
 *
 * The length of the supplied "extents" vector is the same as "x" in
 * looper_<x>_dim().
 */
template <typename Func>
void looper_3_dim(const std::vector<int>& extents, Func func) {
#pragma omp parallel for collapse(3)
    for (int i = 0; i < extents[0]; ++i) {
        for (int j = 0; j < extents[1]; ++j) {
            for (int k = 0; k < extents[2]; ++k) {
                func(i,j,k);
            }
        }
    }
}


/**
 * These "looper" functions take a vector of extents and apply the supplied function "func()" to
 * each combination of values in the extent. The supplied function should take only integer
 * arguments and the argument count must be equal to the length of "extents".
 *
 * The length of the supplied "extents" vector is the same as "x" in
 * looper_<x>_dim().
 *
 * The function is potentially applied in parallel, and so there must be no inter-loop dependencies.
 * Otherwise, the results are undefined.
 */
template <typename Func>
void looper_4_dim(const std::vector<int>& extents, Func func) {
#pragma omp parallel for collapse(4)
    for (int i = 0; i < extents[0]; ++i) {
        for (int j = 0; j < extents[1]; ++j) {
            for (int k = 0; k < extents[2]; ++k) {
                for (int l = 0; l < extents[3]; ++l) {
                    func(i,j,k,l);
                }
            }
        }
    }
}


/**
 * These "looper" functions take a vector of extents and apply the supplied function "func()" to
 * each combination of values in the extent. The supplied function should take only integer
 * arguments and the argument count must be equal to the length of "extents".
 *
 * The length of the supplied "extents" vector is the same as "x" in
 * looper_<x>_dim().
 *
 * The function is potentially applied in parallel, and so there must be no inter-loop dependencies.
 * Otherwise, the results are undefined.
 */
template <typename Func>
void looper_5_dim(const std::vector<int>& extents, Func func) {
#pragma omp parallel for collapse(5)
    for (int i = 0; i < extents[0]; ++i) {
        for (int j = 0; j < extents[1]; ++j) {
            for (int k = 0; k < extents[2]; ++k) {
                for (int l = 0; l < extents[3]; ++l) {
                    for (int m = 0; m < extents[4]; ++m) {
                        func(i,j,k,l,m);
                    }
                }
            }
        }
    }
}


/**
  * Return a MatrixF in "outmat" that is a slice of MatrixF "inmat" where the dimension
  * "dimension" has its index set to "index." The returned matrix will have one
  * less dimension than "inmat."
  *
  * This function performs the same operation as the "select()" function in Torch.
  *
  * The matrix "outmat" will be resized to the correct dimensions if necessary.
  *
  * For example, if "inmat" is a 3 x 2 matrix:
  * [1 2]
  * [3 4]
  * [5 6]
  *
  * Then select(inmat, 0, 2) will return the length-2 array
  * [5 6]
  *
  * And select(inmat, 1, 0) will return the length-3 array
  * [1 3 5]
  */
template <typename T>
void select(Matrix<T>& outmat, const Matrix<T>& inmat, int dimension, int index) {
    // Can we use variodic templates to make code more compact?
    if (dimension >= inmat.order()) {
        error_exit("select(): Invalid dimension.");
    }
    const int order = inmat.order();
    if (0 == order) {
        error_exit("select(): order too small.");
    } else if (1 == order) {
        error_exit("select(): order too small.");
    }
    const int out_order = order - 1; // order of the output matrix.

    std::vector<int> outmat_extents;
    int cur_dim = 0;
    for (int n = 0; n != out_order; ++n) {
        if (n == dimension) {
            ++cur_dim;
        }
        outmat_extents.push_back(inmat.extent(cur_dim));
        ++cur_dim;
    }

    if (outmat_extents != outmat.get_extents()) {
        outmat.resize(outmat_extents);
    }
    // todo: maybe better performance to move the dimension check
    // outside of the looper calls.
    if (out_order == 1) {
        looper_1_dim(outmat.get_extents(), [&] (int i) {
            if (dimension == 0) {
                outmat(i) = inmat(index, i);
            } else if (dimension == 1) {
                outmat(i) = inmat(i, index);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (out_order == 2) {
        looper_2_dim(outmat.get_extents(), [&] (int i, int j) {
            if (dimension == 0) {
                outmat(i, j) = inmat(index, i, j);
            } else if (dimension == 1) {
                outmat(i, j) = inmat(i, index, j);
            } else if (dimension == 2) {
                outmat(i, j) = inmat(i, j, index);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (out_order == 3) {
        looper_3_dim(outmat.get_extents(), [&] (int i, int j, int k) {
            if (dimension == 0) {
                outmat(i, j, k) = inmat(index, i, j, k);
            } else if (dimension == 1) {
                outmat(i, j, k) = inmat(i, index, j, k);
            } else if (dimension == 2) {
                outmat(i, j, k) = inmat(i, j, index, k);
            } else if (dimension == 3) {
                outmat(i, j, k) = inmat(i, j, k, index);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (out_order == 4) {
        looper_4_dim(outmat.get_extents(), [&] (int i, int j, int k, int l) {
            if (dimension == 0) {
                outmat(i, j, k, l) = inmat(index, i, j, k, l);
            } else if (dimension == 1) {
                outmat(i, j, k, l) = inmat(i, index, j, k, l);
            } else if (dimension == 2) {
                outmat(i, j, k, l) = inmat(i, j, index, k, l);
            } else if (dimension == 3) {
                outmat(i, j, k, l) = inmat(i, j, k, index, l);
            } else if (dimension == 4) {
                outmat(i, j, k, l) = inmat(i, j, k, l, index);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (out_order == 5) {
        looper_5_dim(outmat.get_extents(), [&] (int i, int j, int k, int l, int m) {
            if (dimension == 0) {
                outmat(i, j, k, l, m) = inmat(index, i, j, k, l, m);
            } else if (dimension == 1) {
                outmat(i, j, k, l, m) = inmat(i, index, j, k, l, m);
            } else if (dimension == 2) {
                outmat(i, j, k, l, m) = inmat(i, j, index, k, l, m);
            } else if (dimension == 3) {
                outmat(i, j, k, l, m) = inmat(i, j, k, index, l, m);
            } else if (dimension == 4) {
                outmat(i, j, k, l, m) = inmat(i, j, k, l, index, m);
            } else if (dimension == 5) {
                outmat(i, j, k, l, m) = inmat(i, j, k, l, m, index);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else {
        // Add support for larger dimension values on an as-needed basis.
        std::cerr << "select(): Sorry, this dimension value not yet supported. ";
        error_exit("Ask Brian.");
    }
}


/*
   * Return a new matrix that is a slice of MatrixF "inmat" where the dimension
   * "dimension" has its index set to "index." The returned matrix will have one
   * less dimension than "inmat."
   *
   * This function performs the same operation as the "select()" function in Torch.
   *
   * For example, if "inmat" is a 3 x 2 matrix:
   * [1 2]
   * [3 4]
   * [5 6]
   *
   * Then select(inmat, 0, 2) will return the length-2 array
   * [5 6]
   *
   * And select(inmat, 1, 0) will return the length-3 array
   * [1 3 5]
   */
// deprecated: use the version that does not return a matrix.
template <typename T>
Matrix<T> select(const Matrix<T>& inmat, int dimension, int index) {
    // Can we use variodic templates to make code more compact?
    if (dimension >= inmat.order()) {
        error_exit("select(): Invalid dimension.");
    }
    const int order = inmat.order();
    if (0 == order) {
        error_exit("select(): order too small.");
    } else if (1 == order) {
        error_exit("select(): order too small.");
    }
    const int out_order = order - 1; // order of the output matrix.

    std::vector<int> outmat_extents;
    int cur_dim = 0;
    for (int n = 0; n != out_order; ++n) {
        if (n == dimension) {
            ++cur_dim;
        }
        outmat_extents.push_back(inmat.extent(cur_dim));
        ++cur_dim;
    }
    Matrix<T> outmat(outmat_extents);
    select(outmat, inmat, dimension, index);
    return outmat;
}


/**
 * Return a Matrix in "submat" that is the sub-matrix in "fullmat" where "dimension"
 * has its extent narrowed to "size." The elements corresponding to the "size"
 * possible index values in "dimension" of "submat" correspond to the same
 * "size" values obtained by selecting "index" to "index" + size -1 in "fullmat."
 *
 * This function performs the same operation as the "narrow()" function in Torch.
 *
 * The returned matrix "submat" will be resized to the correct dimensions if necessary.
 *
 * For example, if "fullmat" is a 4 x 2 matrix:
 * [1 2]
 * [3 4]
 * [5 6]
 * [7 8]
 *
 * Then narrow(submat, fullmat, 0, 1, 2) will return a 2 x 2 "submat" of:
 * [3 4]
 * [5 6]
 *
 * And narrow(submat, fullmat, 1, 0, 1) will return a 4 x 1 "submat" of:
 * [1]
 * [3]
 * [5]
 * [7]
 *
 * @param submat The result is returned in this matrix, which will be resized to the
 * correct dimensions if necessary.
 *
 * @param fullmat The input matrix.
 *
 * @param dimension The dimension of \p fullmat that will be narrowed to \p size.
 *
 * @param index The starting index value.
 *
 * @param size The size of the output narrowed dimension.
 */
template <typename T>
void narrow(Matrix<T>& submat, const Matrix<T>& fullmat, int dimension, int index, int size) {
    if (dimension >= fullmat.order()) {
        error_exit("narrow(): Invalid dimension.");
    }
    const int order = fullmat.order();
    if (0 == order) {
        error_exit("narrow(): order too small.");
    }
    if (index+size >= fullmat.extent(dimension)) {
        error_exit("narrow(): Error: out of range in the specified dimension.");
    }
    
    std::vector<int> submat_extents = fullmat.get_extents();
    submat_extents.at(dimension) = size;

    if (submat_extents != submat.get_extents()) {
        submat.resize(submat_extents);
    }

    if (order == 1) {
        looper_1_dim(submat.get_extents(), [&] (int i) {
            if (dimension == 0) {
                submat(i) = fullmat(index + i);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 2) {
        looper_2_dim(submat.get_extents(), [&] (int i, int j) {
            if (dimension == 0) {
                submat(i, j) = fullmat(index + i, j);
            } else if (dimension == 1) {
                submat(i, j) = fullmat(i, index + j);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 3) {
        looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
            if (dimension == 0) {
                submat(i, j, k) = fullmat(index + i, j, k);
            } else if (dimension == 1) {
                submat(i, j, k) = fullmat(i, index + j, k);
            } else if (dimension == 2) {
                submat(i, j, k) = fullmat(i, j, index + k);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 4) {
        looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
            if (dimension == 0) {
                submat(i, j, k, l) = fullmat(index + i, j, k, l);
            } else if (dimension == 1) {
                submat(i, j, k, l) = fullmat(i, index + j, k, l);
            } else if (dimension == 2) {
                submat(i, j, k, l) = fullmat(i, j, index + k, l);
            } else if (dimension == 3) {
                submat(i, j, k, l) = fullmat(i, j, k, index + l);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 5) {
        looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
            if (dimension == 0) {
                submat(i, j, k, l, m) = fullmat(index + i, j, k, l, m);
            } else if (dimension == 1) {
                submat(i, j, k, l, m) = fullmat(i, index + j, k, l, m);
            } else if (dimension == 2) {
                submat(i, j, k, l, m) = fullmat(i, j, index + k, l, m);
            } else if (dimension == 3) {
                submat(i, j, k, l, m) = fullmat(i, j, k, index + l, m);
            } else if (dimension == 4) {
                submat(i, j, k, l, m) = fullmat(i, j, k, l, index + m);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else {
        // Add support for larger dimension values on an as-needed basis.
        std::cerr << "narrow(): Sorry, this dimension value not yet supported. ";
        error_exit("Ask Brian.");
    }
}


/*
   * Return a new matrix that is the sub-matrix in "fullmat" where "dimension"
   * has its extent narrowed to "size." The elements corresponding to the "size"
   * possible index values in "dimension" of "submat" correspond to the same
   * "size" values obtained by selecting "index" to "index" + size -1 in "fullmat."
   *
   * This function performs the same operation as the "narrow()" function in Torch.
   *
   * Note that the returned matrix will have the same size as "fullmat"
   * except the dimension "dimension" will have size of "size."
   *
   * For example, if "fullmat" is a 4 x 2 matrix:
   * [1 2]
   * [3 4]
   * [5 6]
   * [7 8]
   *
   * Then narrow(fullmat, 0, 1, 2) will return a 2 x 2 matrix:
   * [3 4]
   * [5 6]
   *
   * And narrow(fullmat, 1, 0, 1) will return a 4 x 1 matrix:
   * [1]
   * [3]
   * [5]
   * [7]
   */
// deprecated: use version that does not return a matrix.
template <typename T>
Matrix<T> narrow(const Matrix<T>& fullmat, int dimension, int index, int size) {
    if (dimension >= fullmat.order()) {
        error_exit("narrow(): Invalid dimension.");
    }
    const int order = fullmat.order();
    if (0 == order) {
        error_exit("narrow(): order too small.");
    }

    std::vector<int> submat_extents = fullmat.get_extents();
    submat_extents.at(dimension) = size;
    Matrix<T> submat(submat_extents);
    narrow(submat, fullmat, dimension, index, size);
    return submat;
}


/**
 * Return a MatrixF in "submat" that is the sub-matrix in "fullmat" where "dimension"
 * has its extent narrowed to "size." The elements corresponding to the "size"
 * possible index values in "dimension" of "submat" correspond to the same
 * "size" values obtained by selecting "permuted_indices(index)" to "permuted_indices(index + size -1)" in "fullmat."
 * Thus, this is the same as narrow() except the the index into "fullmat" is obtained by
 * looking up the index in the "permuted_indices" vector. Note that it is required that
 * "permuted_indices" have the same size as the extent of "dimension" in "fullmat" and that
 * the values in "permuted_indices" must correspond to valid indices into fullmat. It is
 * not required that they correspond to an actual permutation, however, but that would be the
 * typical use case.
 *
 */
template <typename T>
void narrow_permuted(Matrix<T>& submat, const Matrix<T>& fullmat, int dimension, int index, int size,
                     const std::vector<int>& permuted_indices) {
    if (dimension >= fullmat.order()) {
        error_exit("narrow_permuted(): Invalid dimension.");
    }
    const int order = fullmat.order();
    if (0 == order) {
        error_exit("narrow_permuted(): order too small.");
    }

    std::vector<int> submat_extents = fullmat.get_extents();
    submat_extents.at(dimension) = size;

    if (submat_extents != submat.get_extents()) {
        submat.resize(submat_extents);
    }

    if (order == 1) {
        looper_1_dim(submat.get_extents(), [&] (int i) {
            if (dimension == 0) {
                submat(i) = fullmat(permuted_indices[index + i]);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 2) {
        looper_2_dim(submat.get_extents(), [&] (int i, int j) {
            if (dimension == 0) {
                submat(i, j) = fullmat(permuted_indices[index + i], j);
            } else if (dimension == 1) {
                submat(i, j) = fullmat(i, permuted_indices[index + j]);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 3) {
        looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
            if (dimension == 0) {
                submat(i, j, k) = fullmat(permuted_indices[index + i], j, k);
            } else if (dimension == 1) {
                submat(i, j, k) = fullmat(i, permuted_indices[index + j], k);
            } else if (dimension == 2) {
                submat(i, j, k) = fullmat(i, j, permuted_indices[index + k]);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 4) {
        looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
            if (dimension == 0) {
                submat(i, j, k, l) = fullmat(permuted_indices[index + i], j, k, l);
            } else if (dimension == 1) {
                submat(i, j, k, l) = fullmat(i, permuted_indices[index + j], k, l);
            } else if (dimension == 2) {
                submat(i, j, k, l) = fullmat(i, j, permuted_indices[index + k], l);
            } else if (dimension == 3) {
                submat(i, j, k, l) = fullmat(i, j, k, permuted_indices[index + l]);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 5) {
        looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
            if (dimension == 0) {
                submat(i, j, k, l, m) = fullmat(permuted_indices[index + i], j, k, l, m);
            } else if (dimension == 1) {
                submat(i, j, k, l, m) = fullmat(i, permuted_indices[index + j], k, l, m);
            } else if (dimension == 2) {
                submat(i, j, k, l, m) = fullmat(i, j, permuted_indices[index + k], l, m);
            } else if (dimension == 3) {
                submat(i, j, k, l, m) = fullmat(i, j, k, permuted_indices[index + l], m);
            } else if (dimension == 4) {
                submat(i, j, k, l, m) = fullmat(i, j, k, l, permuted_indices[index + m]);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else {
        // Add support for larger dimension values on an as-needed basis.
        std::cerr << "narrow(): Sorry, this dimension value not yet supported. ";
        error_exit("Ask Brian.");
    }
}


/**
 * Copy all data in submat into the locations in fullmat specified by the
 * supplied "dimension", "index", and "size" parameters.
 *
 * The elements corresponding to the "size"
 * possible index values in "dimension" of "submat" correspond to the same
 * "size" values obtained by selecting "index" to "index" + size -1 in "fullmat."
 *
 * Thus, this function is similar to "narrow()" expect that the data transfer goes in
 * the opposite direction.
 *
 * Before calling this function, the matrices "submat" and "fullmat" must have already been allocated
 * and have the correct dimensions. Otherwise, and error will occur.
 * Note that "submat" will have the same size as "fullmat"
 * except the "dimension" of "submat" will have size of "size."
 *
 */
template <typename T>
void reverse_narrow(const Matrix<T>& submat, Matrix<T>& fullmat, int dimension, int index, int size) {
    if (dimension >= fullmat.order()) {
        error_exit("reverse_narrow(): Invalid dimension.");
    }
    const int order = fullmat.order();
    if (0 == order) {
        error_exit("reverse_narrow(): order too small.");
    } else if (1 == order) {
        error_exit("reverse_narrow(): order too small.");
    }

    std::vector<int> submat_extents = fullmat.get_extents();
    submat_extents.at(dimension) = size;

    if (submat_extents != submat.get_extents()) {
        std::cerr << "reverse_narrow(): submat matrix has wrong dimensions." << std::endl;
    }

    if (order == 1) {
        looper_1_dim(submat.get_extents(), [&] (int i) {
            if (dimension == 0) {
                fullmat(index + i) = submat(i);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 2) {
        looper_2_dim(submat.get_extents(), [&] (int i, int j) {
            if (dimension == 0) {
                fullmat(index + i, j) = submat(i, j);
            } else if (dimension == 1) {
                fullmat(i, index + j) = submat(i, j);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 3) {
        looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
            if (dimension == 0) {
                fullmat(index + i, j, k) = submat(i, j, k);
            } else if (dimension == 1) {
                fullmat(i, index + j, k) = submat(i, j, k);
            } else if (dimension == 2) {
                fullmat(i, j, index + k) = submat(i, j, k);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 4) {
        looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
            if (dimension == 0) {
                fullmat(index + i, j, k, l) = submat(i, j, k, l);
            } else if (dimension == 1) {
                fullmat(i, index + j, k, l) = submat(i, j, k, l);
            } else if (dimension == 2) {
                fullmat(i, j, index + k, l) = submat(i, j, k, l);
            } else if (dimension == 3) {
                fullmat(i, j, k, index + l) = submat(i, j, k, l);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else if (order == 5) {
        looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
            if (dimension == 0) {
                fullmat(index + i, j, k, l, m) = submat(i, j, k, l, m);
            } else if (dimension == 1) {
                fullmat(i, index + j, k, l, m) = submat(i, j, k, l, m);
            } else if (dimension == 2) {
                fullmat(i, j, index + k, l, m) = submat(i, j, k, l, m);
            } else if (dimension == 3) {
                fullmat(i, j, k, index + l, m) = submat(i, j, k, l, m);
            } else if (dimension == 4) {
                fullmat(i, j, k, l, index + m) = submat(i, j, k, l, m);
            } else {
                error_exit("Bad dim size");
            }
        });
    } else {
        // Add support for larger dimension values on an as-needed basis.
        std::cerr << "narrow(): Sorry, this dimension value not yet supported. ";
        error_exit("Ask Brian.");
    }
}


/**
 * Copy the contents of the supplied 2D matrix into a larger 2D matrix.
 *
 * The contents of a 2-dimensional matrix \p submat will be copied into another larger
 * 2-dimensional matrix \p fullmat such that the location of the upper-left
 * corner of \p submat inside \p fullmat will be fullmat(row_offset, col_offset). Thus,
 * after the copy operation, \p submat will be a sub-matrix of \p fullmat.
 *
 * Both \p submat and \p fullmat must be 2-dimensional.
 *
 * @param submat A 2-dim submatrix to be copied.
 * @param fullmat A 2-dim matrix such that the supplied submat at the specified offset can
 * fit completely inside. submat will be copied into a sub-matrix of this matrix.
 * @param row_offset The upper-left corner of submat will be placed at this row offset inside fullmat.
 * @param col_offset The upper-left corner of submat will be placed at this column offset inside fullmat.
 */
template <typename T>
void copy_small_to_large_mat_2d(const Matrix<T>& submat, Matrix<T>& fullmat, int row_offset, int col_offset) {
    if (submat.order() != 2) {
        error_exit("copy_from_submatrix(): Error: submat is not 2-dimensional.");
    }
    if (fullmat.order() != 2) {
        error_exit("copy_from_submatrix(): Error: fullmat is not 2-dimensional.");
    }
#pragma omp parallel for
    for (int r = 0; r < submat.extent(0); ++r) {
        for (int c = 0; c < submat.extent(1); ++ c) {
            fullmat(row_offset + r, col_offset + c) = submat(r,c);
        }
    }
}


/**
 * Copy a submatrix of the supplied larger matrix into the supplied smaller matrix.
 *
 * A submatrix will be copied from \p fullmat into \p submat such that the
 * upper-left corner of the extracted submatrix is at fullmat(row_offset, col_offset).
 *
 * Both \p submat and \p fullmat must be 2-dimensional. Otherwise the program will exit with an error message.
 *
 * @param submat The 2-dimensional output matrix into which the sub-matrix values will be copied.
 * @param fullmat A 2-dimensional matrix such that the supplied submat at the specified offset will fit inside.
 * @param row_offset The upper-left corner of submat will correspond to this row offset inside fullmat.
 * @param col_offset The upper-left corner of submat will corresopnd to this column offset inside fullmat.
 */
template <typename T>
void copy_large_to_small_mat_2d(Matrix<T>& submat, const Matrix<T>& fullmat, int row_offset, int col_offset) {
    if (submat.order() != 2) {
        error_exit("copy_to_submatrix(): Error: submat is not 2-dimensional.");
    }
    if (fullmat.order() != 2) {
        error_exit("copy_to_submatrix(): Error: fullmat is not 2-dimensional.");
    }
#pragma omp parallel for
    for (int r = 0; r < submat.extent(0); ++r) {
        for (int c = 0; c < submat.extent(1); ++ c) {
            submat(r,c) = fullmat(row_offset + r, col_offset + c);
        }
    }
}

/**
 * Copy the contents of the supplied matrix along an axis into an output matrix.
 *
 * Copy the supplied matrix \p submat into the output matrix \p fullmat so that
 * the contents of \p submat become a sub-matrix of \p fullmat which is shifted
 * by \p offset along the specified \p axis. All extents of both matrices must
 * agree except for \p axis.
 *
 * Note that this function can be used to concatenate several matrices along a
 * specified axiss by making multiple calls with appropriate offset values to
 * position in matrix to be concatenated at the desired offset.
 *
 *
 * @param submat The input matrix to be copied.
 * @param fullmat The output matrix.
 * @param offset The number of elements along axis in fullmat that submat will
 * be shifted.
 * @param axis The axis along which to shift and copy submat inside fullmat.
 */
template <typename T>
void copy_along_axis(const Matrix<T>& submat, Matrix<T>& fullmat, int offset, int axis=0) {
    // Both matrices must be same order and all extents must match, except for the
    // axis extent:
    if (submat.order() == fullmat.order()) {
        // todo: check that all extents match excpet for the axis extent.
    } else {
        error_exit("copy_along_axis(): Both matrices must have same order.");
    }

    if (submat.order() == 1) {
        if (axis != 0) {
            error_exit("copy_along_axis(): Bad axis value.");
        }
        looper_1_dim(submat.get_extents(), [&] (int i) {
            fullmat(i + offset) = submat(i);
        });
    } else if (submat.order() == 2) {
        if (axis == 0) {
            looper_2_dim(submat.get_extents(), [&] (int i, int j) {
                fullmat(i + offset, j) = submat(i, j);
            });
        } else if (axis == 1) {
            looper_2_dim(submat.get_extents(), [&] (int i, int j) {
                fullmat(i, j + offset) = submat(i, j);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else if (submat.order() == 3) {
        if (axis == 0) {
            looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
                fullmat(i + offset, j, k) = submat(i, j, k);
            });
        } else if (axis == 1) {
            looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
                fullmat(i, j + offset, k) = submat(i, j, k);
            });
        } else if (axis == 2) {
            looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
                fullmat(i, j, k + offset) = submat(i, j, k);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else if (submat.order() == 4) {
        if (axis == 0) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                fullmat(i + offset, j, k, l) = submat(i, j, k, l);
            });
        } else if (axis == 1) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                fullmat(i, j + offset, k, l) = submat(i, j, k, l);
            });
        } else if (axis == 2) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                fullmat(i, j, k + offset, l) = submat(i, j, k, l);
            });
        } else if (axis == 3) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                fullmat(i, j, k, l + offset) = submat(i, j, k, l);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else if (submat.order() == 5) {
        if (axis == 0) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                fullmat(i + offset, j, k, l, m) = submat(i, j, k, l, m);
            });
        } else if (axis == 1) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                fullmat(i, j + offset, k, l, m) = submat(i, j, k, l, m);
            });
        } else if (axis == 2) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                fullmat(i, j, k + offset, l, m) = submat(i, j, k, l, m);
            });
        } else if (axis == 3) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                fullmat(i, j, k, l + offset, m) = submat(i, j, k, l, m);
            });
        } else if (axis == 4) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                fullmat(i, j, k, l, m + offset) = submat(i, j, k, l, m);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else {
        error_exit("copy_along_axis(): Sorry, not supported yet.");
    }
}


/**
 * Extract a submatrix along an axis of a larger matrix.
 *
 * This function performs the reverse of the operation performed by
 * copy_along_axis().
 *
 * @param submat The output matrix to be copied.
 * @param fullmat The input matrix. A sub-matrix of the same size as submat will
 * be copied from fullmat into submat.
 * @param offset The number of elements along axis in fullmat in which to
 * extract the sub-matrix.
 * @param axis The axis along which to extract the submatrix.
 */
template <typename T>
void extract_along_axis(Matrix<T>& submat, const Matrix<T>& fullmat, int offset, int axis=0) {
    // Both matrices must be same order and all extents must match, except for the
    // axis extent:
    if (submat.order() == fullmat.order()) {
        // todo: check that all extents match excpet for the axis extent.
    } else {
        error_exit("copy_along_axis(): Both matrices must have same order.");
    }

    if (submat.order() == 1) {
        if (axis != 0) {
            error_exit("copy_along_axis(): Bad axis value.");
        }
        looper_1_dim(submat.get_extents(), [&] (int i) {
            //fullmat(i + offset) = submat(i);
            submat(i) = fullmat(i + offset);
        });
    } else if (submat.order() == 2) {
        if (axis == 0) {
            looper_2_dim(submat.get_extents(), [&] (int i, int j) {
                //fullmat(i + offset, j) = submat(i, j);
                submat(i, j) = fullmat(i + offset, j);
            });
        } else if (axis == 1) {
            looper_2_dim(submat.get_extents(), [&] (int i, int j) {
                //fullmat(i, j + offset) = submat(i, j);
                submat(i, j) = fullmat(i, j + offset);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else if (submat.order() == 3) {
        if (axis == 0) {
            looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
                //fullmat(i + offset, j, k) = submat(i, j, k);
                submat(i, j, k) = fullmat(i + offset, j, k);
            });
        } else if (axis == 1) {
            looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
                //fullmat(i, j + offset, k) = submat(i, j, k);
                submat(i, j, k) = fullmat(i, j + offset, k);
            });
        } else if (axis == 2) {
            looper_3_dim(submat.get_extents(), [&] (int i, int j, int k) {
                //fullmat(i, j, k + offset) = submat(i, j, k);
                submat(i, j, k) = fullmat(i, j, k + offset);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else if (submat.order() == 4) {
        if (axis == 0) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                //fullmat(i + offset, j, k, l) = submat(i, j, k, l);
                submat(i, j, k, l) = fullmat(i + offset, j, k, l);
            });
        } else if (axis == 1) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                //fullmat(i, j + offset, k, l) = submat(i, j, k, l);
                submat(i, j, k, l) = fullmat(i, j + offset, k, l);
            });
        } else if (axis == 2) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                //fullmat(i, j, k + offset, l) = submat(i, j, k, l);
                submat(i, j, k, l) = fullmat(i, j, k + offset, l);
            });
        } else if (axis == 3) {
            looper_4_dim(submat.get_extents(), [&] (int i, int j, int k, int l) {
                //fullmat(i, j, k, l + offset) = submat(i, j, k, l);
                submat(i, j, k, l) = fullmat(i, j, k, l + offset);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else if (submat.order() == 5) {
        if (axis == 0) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                //fullmat(i + offset, j, k, l, m) = submat(i, j, k, l, m);
                submat(i, j, k, l, m) = fullmat(i + offset, j, k, l, m);
            });
        } else if (axis == 1) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                //fullmat(i, j + offset, k, l, m) = submat(i, j, k, l, m);
                submat(i, j, k, l, m) = fullmat(i, j + offset, k, l, m);
            });
        } else if (axis == 2) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                //fullmat(i, j, k + offset, l, m) = submat(i, j, k, l, m);
                submat(i, j, k, l, m) = fullmat(i, j, k + offset, l, m);
            });
        } else if (axis == 3) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                //fullmat(i, j, k, l + offset, m) = submat(i, j, k, l, m);
                submat(i, j, k, l, m) = fullmat(i, j, k, l + offset, m);
            });
        } else if (axis == 4) {
            looper_5_dim(submat.get_extents(), [&] (int i, int j, int k, int l, int m) {
                //fullmat(i, j, k, l, m + offset) = submat(i, j, k, l, m);
                submat(i, j, k, l, m) = fullmat(i, j, k, l, m + offset);
            });
        } else {
            error_exit("copy_along_axis(): Bad axis value.");
        }
    } else {
        error_exit("copy_along_axis(): Sorry, not supported yet.");
    }
}


/**
 * Apply the function to each element of the supplied matrix X.
 *
 * This function performs the same operation as the "apply()" function in Torch, except that
 * this function is data parallel.
 *
 * For each element in X, apply the function "func()" to the
 * value and set the result of the function as the new value in X.
 * The function is applied in parallel and so the value of "func()" should
 * only depend on its input value. Otherwise, the results are not defined.
 */
template <typename T, typename Func>
void apply(Matrix<T>& X, Func func) {
#pragma omp parallel for
    for (int i = 0; i < X.size(); i++) {
        X[i] = func(X[i]);
    }
}


/**
 * Apply the function to each element of the supplied matrix X.
 *
 * This function performs the same operation as the "apply()" function in Torch.
 *
 * For each element in X, apply the function "func()" to the
 * value and set the result of the function as the new value in X.
 *
 * This is the same as "apply()" except that the function "func()" is
 * applied in the same order as they appear in the underlying
 * backing array of X.
 *
 */
template <typename T, typename Func>
void apply_sequential(Matrix<T>& X, Func func) {
    for (int i = 0; i < X.size(); i++) {
        X[i] = func(X[i]);
    }
}


/**
 * Apply the function to each element of the supplied const matrix to compute a new
 * value for "outmat."
 *
 * For each index i in backing array of the supplied matrices, compute a new value for outmat[i]
 * as:
 * outmat[i] = func(inmat[i]).
 *
 * It is allowed for "outmat" and one or more of the input matrices to reference the same matrix.
 *
 * The function is applied in parallel and so the value of "func()" should
 * only depend on its input value. Otherwise, the results are not defined.
 *
 * The number of elements in all supplied matrices must be the same but the dimensions may
 * be different.
 */
template <typename T, typename Func>
void map1(Matrix<T>& outmat, const Matrix<T>& inmat, Func func) {
    if (outmat.size() != inmat.size()) {
        error_exit("map1(): wrong matrix size.");
    }
#pragma omp parallel for
    for (int i = 0; i < outmat.size(); i++) {
        outmat[i] = func(inmat[i]);
    }
}


/**
 * Apply the function to each element of the supplied const matrices to compute a new
 * value for "outmat."
 *
 * For each index i in backing array of the supplied matrices, compute a new value for outmat[i]
 * as:
 * outmat[i] = func(inmat1[i], inmat2[i]).
 *
 * It is allowed for "outmat" and one or more of the input matrices to reference the same matrix.
 *
 * The function is applied in parallel and so the value of "func()" should
 * only depend on its input value. Otherwise, the results are not defined.
 *
 * The number of elements in "inmat1" and "inmat2" must be the same but the dimensions may
 * be different. If the number of elements in "outmat" is not the same, it will be resized to the
 * same size as "inmat1" and "inmat2."
 */
template <typename T, typename Func>
void map2(Matrix<T>& outmat, const Matrix<T>& inmat1, const Matrix<T>& inmat2, Func func) {
    if (inmat1.size() != inmat2.size()) {
        error_exit("map2(): inmat1 must have the same size as inmat2");
    }
    if (outmat.size() != inmat2.size()) {
        outmat.resize(inmat2.get_extents());
    }
#pragma omp parallel for
    for (int i = 0; i < outmat.size(); i++) {
        outmat[i] = func(inmat1[i], inmat2[i]);
    }
}


/**
 * Apply the function to each element of the supplied const matrices to compute a new
 * value for "outmat."
 *
 * For each index i in backing array of the supplied matrices, compute a new value for outmat[i]
 * as:
 * outmat[i] = func(inmat1[i], inmat2[i], inmat3[i])
 *
 * It is allowed for "outmat" and one or more of the input matrices to reference the same matrix.
 *
 * The function is applied in parallel and so the value of "func()" should
 * only depend on its input value. Otherwise, the results are not defined.
 *
 * The number of elements in all supplied matrices must be the same but the dimensions may
 * be different.
 */
template <typename T, typename Func>
void map3(Matrix<T>& outmat, const Matrix<T>& inmat1, const Matrix<T>& inmat2, const Matrix<T>& inmat3, Func func) {
    if ((outmat.size() != inmat1.size()) || (outmat.size() != inmat2.size()) || (outmat.size() != inmat3.size())) {
        error_exit("map3(): wrong matrix size.");
    }
#pragma omp parallel for
    for (int i = 0; i < outmat.size(); i++) {
        outmat[i] = func(inmat1[i], inmat2[i], inmat3[i]);
    }
}


/**
 * Copy the contents of B into the specified column of A.
 *
 * The data is copied in the same order that it is stored in the backing array of B.
 *
 * The order of B (i.e., number of elements in B) must equal the number of rows in A.
 */
template <typename T>
void copy_matrix_to_column(Matrix<T>& A, const Matrix<T>& B, int column) {
    // Copy all data in B into column "column" of A.
    bool inconsistent_parameters = false;
    if (B.size() != A.extent(0)) {
        inconsistent_parameters = true;
    }
    if (column >= A.extent(1)) {
        inconsistent_parameters = true;
    }
    if (inconsistent_parameters) {
        error_exit("copy_matrix_to_column(): Inconsistent parameter values.");
    }
    for (int i = 0; i != B.size(); ++i) {
        A(i, column) = B[i];
    }
}


/**
 * Copy the contents of the specified column of A into B.
 *
 * The data is copied in the same order that it is stored in the backing array of B.
 *
 * The order of B (i.e., number of elements in B) must equal the number of rows in A.
 */
template <typename T>
void copy_column_to_matrix(const Matrix<T>& A, Matrix<T>& B, int column) {
    // Copy all data from column "column" of A into B.
    bool inconsistent_parameters = false;
    if (B.size() != A.extent(0)) {
        inconsistent_parameters = true;
    }
    if (column >= A.extent(1)) {
        inconsistent_parameters = true;
    }
    if (inconsistent_parameters) {
        error_exit("copy_column_to_matrix(): Inconsistent parameter values.");
    }
    for (int i = 0; i != B.size(); ++i) {
        B[i] = A(i, column);
    }
}


/**
 * Return the maximum value.
 */
template <typename T>
T max_value(const std::vector<T>& a) {
    T max_val = a[0];
    for (size_t i = 0; i != a.size(); ++i) {
        max_val = std::max(max_val, a[i]);
    }
    return max_val;
}


/**
 * Return the minimum value.
 */
template <typename T>
T min_value(const std::vector<T>& a) {
    T min_val = a[0];
    for (size_t i = 0; i != a.size(); ++i) {
        min_val = std::min(min_val, a[i]);
    }
    return min_val;
}


/**
 * Return the maximum value.
 */
template <typename T>
T max_value(const Matrix<T>& A) {
    return max_value(A.get_backing_vector());
}


/**
 * Return the minimum value.
 */
template <typename T>
T min_value(const Matrix<T>& A) {
    return min_value(A.get_backing_vector());
}


/**
 * Return the magnitude of the value with the largest magnitude.
 */
template <typename T>
T max_magnitude_value(const Matrix<T>& A) {
    T min = min_value(A.get_backing_vector());
    T max = max_value(A.get_backing_vector());
    if ((min < 0) && (max < 0)) {
        return 0;
    } else {
        if (-min > max) {
            return -min;
        } else {
            return max;
        }
    }
}


/**
 * Return the maximum of the supplied matrix along the specified axis.
 *
 * @param out_mat The output matrix. This matrix will be resized to the
 * correct dimensions by this function, if necessary.
 *
 * @param axis The axis along which to compute the maximum values.
 * Currently only 0 and 1 are supported. Note that 0 means to
 * compute the maximum value in each column of \A while 1 means
 * to compute the maximum value of each row of \p A.
 *
 * @param A The input matrix. Currently, this must be 2-dimensional.
 */
template <typename T>
void amax(Matrix<T>& out_mat, int axis, const Matrix<T>& A) {
    if (A.order() != 2) {
        error_exit("amax(): Invalid dimensions of supplied matrix.");
    }
    if (axis == 0) {
        const int out_size = A.extent(1);
        if (out_mat.size() != out_size) {
            out_mat.resize(out_size);
        }
        for (int c = 0; c < out_size; ++c) {
            T maxv = A(0, c);
            for (int r = 1; r < A.extent(0); ++r) {
                if (A(r, c) > maxv) {
                    maxv = A(r, c);
                }
            }
            out_mat(c) = maxv;
        }
    } else if (axis == 1) {
        const int out_size = A.extent(0);
        if (out_mat.size() != out_size) {
            out_mat.resize(out_size);
        }
        for (int r = 0; r < out_size; ++r) {
            T maxv = A(r, 0);
            for (int c = 1; c < A.extent(1); ++c) {
                if (A(r, c) > maxv) {
                    maxv = A(r, c);
                }
            }
            out_mat(r) = maxv;
        }
    } else {
        error_exit("amax(): Invalid axis value.");
    }
}


/**
 * Return the maximum of the supplied matrix along the specified axis
 * including the indices of the maximum values.
 *
 * @param out_mat The output matrix of maximum values. This matrix
 * will be resized to the correct dimensions by this function, if necessary.
 *
 * @param indices The output matrix of indices corresponding to the
 * maximum values returned in \p out_mat.
 *
 * @param axis The axis along which to compute the maximum values.
 * Currently only 0 and 1 are supported. Note that 0 means to
 * compute the maximum value in each column of \A while 1 means
 * to compute the maximum value of each row of \p A.
 *
 * @param A The input matrix. Currently, this must be 2-dimensional.
 */
template <typename T>
void amax_indices(Matrix<T>& out_mat, MatrixI& indices, int axis,
                  const Matrix<T>& A) {
    if (A.order() != 2) {
        error_exit("amax(): Invalid dimensions of supplied matrix.");
    }
    if (axis == 0) {
        const int out_size = A.extent(1);
        if (out_mat.size() != out_size) {
            out_mat.resize(out_size);
            indices.resize(out_size);
        }
        for (int c = 0; c < out_size; ++c) {
            int max_ind = 0;
            T maxv = A(0, c);
            for (int r = 1; r < A.extent(0); ++r) {
                if (A(r, c) > maxv) {
                    maxv = A(r, c);
                    max_ind = r;
                }
            }
            out_mat(c) = maxv;
            indices(c) = max_ind;
        }
    } else if (axis == 1) {
        const int out_size = A.extent(0);
        if (out_mat.size() != out_size) {
            out_mat.resize(out_size);
            indices.resize(out_size);
        }
        for (int r = 0; r < out_size; ++r) {
            int max_ind = 0;
            T maxv = A(r, 0);
            for (int c = 1; c < A.extent(1); ++c) {
                if (A(r, c) > maxv) {
                    maxv = A(r, c);
                    max_ind = c;
                }
            }
            out_mat(r) = maxv;
            indices(r) = max_ind;
        }
    } else {
        error_exit("amax(): Invalid axis value.");
    }
}

////////////////////////////////////////////////////


/**
 * Compute the element-wise product of B and C and then place the result in A.
 *
 * Matrices B and C must have the same number of elements. If Matrix A does not also
 * have the same number of elements, it will be resized to the same dimensions as
 * matrix B.
 *
 * Note that although B and C are required to have the same number of elements, their dimensions
 * are allowed to differ. For example if B is of size (2 x 5) and C is of size (10 x 1), both
 * matrices have a size of 10, which is allowed.
 *
 * @param A Result is returned in this matrix, which will be resized to the same dimensions as
 * matrix B if it does not already have the same number of elements.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
template <typename T>
void element_wise_multiply(Matrix<T>& A, const Matrix<T> &B, const Matrix<T> &C) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map2(A, B, C, [] (T b, T c) {
        return b*c;
    });
}


/**
 * Set all values to be uniformly disstributed random values in [min, max].
 *
 */
void randomize_uniform(MatrixF& A, float min, float max);


/**
 * Set all values to be normally disstributed random values with "mean"
 * and "std_deviation".
 *
 */
void randomize_normal(MatrixF& A, float mean, float std_deviation);


/**
 * Compute the element-wise division B / C and then place the result in A.
 *
 * Matrices B and C must have the same number of elements.
 *
 * Specifically, compute:
 *
 *      B + epsilon
 * A = ------------
 *      C + epsilon
 *
 * @param A The result is returned in this matrix, which will be resized to the same dimensions as
 * matrix B if it does not already have the same number of elements.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 * @param epsilon A small positive constant that is added to both the numerator and denominator.
 */
template <typename T>
void element_wise_divide(Matrix<T>& A, const Matrix<T> &B, const Matrix<T> &C, T epsilon) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map2(A, B, C, [=] (T b, T c) {
        return (b + epsilon)/(c + epsilon);
    });
}


/**
 * Element-wise divide.
 *
 * A = B/C
 *
 * Matrices B and C must have the same number of elements.
 *
 * @param A The result is returned in this matrix, which will be resized to the same dimensions as
 * matrix B if it does not already have the same number of elements.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
template <typename T>
void element_wise_divide(Matrix<T>& A, const Matrix<T> &B, const Matrix<T> &C) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map2(A, B, C, [=] (T b, T c) {
        return b/c;
    });
}


/**
 * Compute the element-wise difference (B-C) and then place the result in A.
 *
 * Matrices B and C must have the same number of elements.
 *
 * @param A The result is returned in this matrix, which will be resized to the same dimensions as
 * matrix B if it does not already have the same number of elements.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
template <typename T>
void element_wise_difference(Matrix<T>& A, const Matrix<T> &B, const Matrix<T> &C) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map2(A, B, C, [] (T b, T c) {
        return b - c;
    });
}


/**
 * Compute the element-wise square B.^2 and then place the result in A.
 *
 * Matrices A and B must have the same number of elements.
 * It is allowed for A
 * and B to refer to the same object.
 *
 * @param A Output matrix, which will be resized to the same dimensions as
 * matrix B if it does not already have the same number of elements.
 * @param B Input matrix which is not modified.
 */
template <typename T>
void element_wise_square(Matrix<T>& A, const Matrix<T>& B) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map1(A, B, [] (T b) {
        return b*b;
    });
}


/**
 * Compute the element-wise natural log and then place the result in A.
 *
 * Matrices A and B must have the same number of elements.
 * It is allowed for A
 * and B to refer to the same object.
 *
 * @param A Output matrix, which will be resized to the same dimensions as
 * matrix B if it does not already have the same number of elements.
 * @param B Input matrix which is not modified.
 */
template <typename T>
void element_wise_ln(Matrix<T>& A, const Matrix<T>& B) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map1(A, B, [] (T b) {
        return std::log(b);
    });
}


/**
 * Compute the element-wise natural exponential and then place the result in A.
 *
 * Matrices A and B must have the same number of elements.
 * It is allowed for A
 * and B to refer to the same object.
 *
 * @param A Output matrix, which will be resized to the same dimensions as
 * matrix B if it does not already have the same number of elements.
 * @param B Input matrix which is not modified.
 */
template <typename T>
void element_wise_exp(Matrix<T>& A, const Matrix<T>& B) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map1(A, B, [] (T b) {
        return std::exp(b);
    });
}


/**
 * Compute the sum of all elements in <i>A</i> and return it.
 *
 * @param A The input matrix.
 * @return The sum of all elements in A
 */
template <typename T>
T sum(const Matrix<T>& A) {
    T sum = 0;
    for (int i = 0; i != A.size(); ++i) {
        sum += A[i];
    }
    return sum;
}


/**
 * Compute the transpose of B and put the result in A.
 *
 * B must be 2-dimensional matrices. Otherwise, and error will be thrown.
 *
 * @param A The result matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 */
template <typename T>
void transpose(Matrix<T>& A, const Matrix<T> &B) {
    const int rowsB = B.extent(0);
    const int colsB = B.extent(1);
    bool do_resize = false;
    if ((A.size() != B.size()) || (A.order() != 2)) {
        do_resize = true;
    } else if ((A.extent(0) != B.extent(1)) || (A.extent(1) != B.extent(0))) {
        do_resize = true;
    }
    if (do_resize) {
        A.resize(colsB, rowsB);
    }
    // For each row of B
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rowsB; i++) {
        // For each column of C
        for (int j = 0; j < colsB; j++) {
            A(j, i) =  B(i, j);
        }
    }
}


/**
 * Set all elements of the matrix <i>A</i> to have value <i>value</i> and
 * return the result in <i>A</i>.
 * @param A
 * @param value
 */
template <typename T1, typename T2>
void set_value(Matrix<T1>& A, T2 value) {
#pragma omp parallel for
    for (int i = 0; i < A.size(); i++) {
        A[i] = value;
    }
}


/**
 * Take the element-wise square root of <i>A</i> and return the result in A.
 *
 */
template <typename T>
void square_root(Matrix<T>& A) {
    apply(A, [] (T a) {
        return std::sqrt(a);
    });
}


/**
 * Take the element-wise tanh of <i>A</i> and return the result in A.
 *
 */
template <typename T>
void element_wise_tanh(Matrix<T>& A) {
    apply(A, [] (T a) {
        return std::tanh(a);
    });
}


/**
 * Take the element-wise absolute value of <i>A</i> and return the result in A.
 *
 */
template <typename T>
void absolute(Matrix<T>& A) {
    apply(A, [] (T a) {
        return std::abs(a);
    });
}


/**
 * Perform the following operation element-wise on \p A and return
 * the result in \p A:
 *
 * Let x denote the element value. If the absolute value of
 * x exceeds \p thresh, the slope of further increase is linear with
 * slope alpha.
 *
 */
template <typename T>
void soft_thresh(Matrix<T>& A, T thresh, T alpha) {
    apply(A, [=] (T a) {
        //return std::abs(a);
        if (a > 0) {
            if (a < thresh) {
                return a;
            } else {
                return thresh + alpha*(a - thresh);
            }
        } else {
            if (a > -thresh) {
                return a;
            } else {
                return -thresh + alpha*(a + thresh);
            }
        }
    });
}


/**
 * Multiply each element of B by scale_factor and put the result in A.
 * A and B are allowed to refer to the same matrix.
 *
 * A <- scale_factor*B
 *
 * @param A Result is returned in this matrix, which will be resized to the same dimensions as
 * matrix B if necessary.
 * @param B Input matrix which is not modified.
 * @param scale_factor The scale factor.
 *
 */
template <typename T>
void scale(Matrix<T>& A, const Matrix<T> &B, T scale_factor) {
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map1(A, B, [=] (T b) {
        return b*scale_factor;
    });
}


/**
 * Multiply each element of A by scale_factor.
 *
 * A <- A*scale_factor
 *
 * @param A The source and result matrix.
 * @param scale_factor The scale factor.
 *
 */
template <typename T>
void scale(Matrix<T>& A, T scale_factor) {
    apply(A, [=] (T a) {
        return a*scale_factor;
    });
}


/**
   * Add a scalar value to each element of the supplied matrix.
   *
   * @param A The matrix which will be modified.
   * @param x The scalar which will be added to each element in A.
   */
template <typename T>
void add_scalar(Matrix<T>& A, T x) {
    apply(A, [=] (T a) {
        return a + x;
    });
}


/**
   * Compute the element-wise sum (B+C) and then place the result in A.
   *
   * A <- B + C
   *
   * Matrices B and C must have the same number of elements. Otherwise, a runtime exception will occur.
   * @param A The result is returned in this matrix, which will be resized to the same dimensions as
   * matrix B if it does not already have the same number of elements.
   * @param B Input matrix which is not modified.
   * @param C Input matrix which is not modified.
   */
template <typename T>
void element_wise_sum(Matrix<T>& A, const Matrix<T> &B, const Matrix<T> &C) {
    if (B.size() != C.size()) {
        error_exit("element_wise_sum(): Error: B and C do not have the same size.");
    }
    if (A.size() != B.size()) {
        A.resize(B.get_extents());
    }
    map2(A, B, C, [] (T b, T c) {
        return b + c;
    });
}


/**
 * Check that the supplied matrices have dimensions that are compatible with the factorization:
 *
 * X approx= W * H.
 *
 * If they are not compatible, exit with an error.
 */
void check_matrix_factorization_dimensions(const MatrixF& X, const MatrixF& W, const MatrixF& H);


/**
  * Check that both matrices have the same dimensions. If they differ, exit with an error.
  */
template <typename T1, typename T2>
void check_dimensions(const Matrix<T1>& A, const Matrix<T2>& B) {
    if (A.get_extents() != B.get_extents()) {
        std::cerr << "A extents: " << std::endl;
        print_vector(A.get_extents());
        std::cerr << "B extents: " << std::endl;
        print_vector(B.get_extents());
        error_exit("Error: Supplied matrices A and B do not have the same extents!");
    }
}


/**
 * Check that dimensions are consistant with A = B^T * C.
 *
 * If dimensions are consistant, return true. Otherwise, return false.
 */
bool check_dimensions_a_eq_b_tran_times_c(const MatrixF& A, const MatrixF& B, const MatrixF& C);


/**
 * Check that dimensions are consistant with A = B * C^T.
 *
 * If dimensions are consistant, return true. Otherwise, return false.
 */
bool check_dimensions_a_eq_b_times_c_tran(const MatrixF& A, const MatrixF& B, const MatrixF& C);


/**
 * Compute B x C and place the result in A.
 *
 * Note: This method implements the basic easy-to-understand version. It is
 * not optimized in any way.
 *
 * Parameters
 *
 * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
void mat_multiply_naive(MatrixF& A, const MatrixF &B, const MatrixF &C);



/**
 * Compute B x C and place the result in A.
 *
 * This implementation will call an optimized BLAS if one is available.
 *
 * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
 * if necessary.
 * @param B Input matrix which is not modified.
 * @param C Input matrix which is not modified.
 */
void mat_multiply(MatrixF& A, const MatrixF& B, const MatrixF& C);


/**
   * Compute A = alpha*B*C + beta*A.
   *
   * This implementation will call an optimized BLAS if one is available.
   *
   * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
   * if necessary.
   * @param B Input matrix which is not modified.
   * @param C Input matrix which is not modified.
   */
void mat_multiply(MatrixF& A, const MatrixF& B, const MatrixF& C, float alpha, float beta);


/**
   * Compute A = B*C using an optimized BLAS.
   *
   * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
   * if necessary.
   * @param B Input matrix which is not modified.
   * @param C Input matrix which is not modified.
   */
void mat_multiply_blas(MatrixF& A, const MatrixF &B, const MatrixF &C);


/**
   * Compute A = alpha*B*C + beta*A using an optimzied BLAS.
   *
   * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
   * if necessary.
   * @param B Input matrix which is not modified.
   * @param C Input matrix which is not modified.
   */
void mat_multiply_blas(MatrixF& A, const MatrixF &B, const MatrixF &C, float alpha, float beta);


/**
   * Compute A <- B^T x C.
   *
   *
   * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
   * if necessary.
   * @param B Input matrix which is not modified. Note that the transpose of this matrix is used in the
   * multiplication.
   * @param C Input matrix which is not modified.
   */
void mat_multiply_left_transpose(MatrixF& A, const MatrixF& B, const MatrixF& C);

// Slow version for verifying correctness.
void mat_multiply_left_transpose_naive(MatrixF& A, const MatrixF& B, const MatrixF& C);

/**
   * Compute A <- A + B^T x C
   *
   *
   * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
   * if necessary.
   * @param B Input matrix which is not modified. Note that the transpose of this matrix is used in the
   * multiplication.
   * @param C Input matrix which is not modified.
   */
void mat_multiply_left_transpose_accumulate(MatrixF& A, const MatrixF& B, const MatrixF& C);

// Slow version for verifying correctness.
void mat_multiply_left_transpose_naive_accumulate(MatrixF& A, const MatrixF& B, const MatrixF& C);

/**
   * Compute A <- A + B x C^T.
   *
   * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
   * if necessary.
   * @param B Input matrix which is not modified.
   * @param C Input matrix which is not modified. Note that the transpose of this matrix is used in the
   * multiplication.
   */
void mat_multiply_right_transpose(MatrixF& A, const MatrixF& B, const MatrixF& C);

// Slow version for verifying correctness.
void mat_multiply_right_transpose_naive(MatrixF& A, const MatrixF& B, const MatrixF& C);

/**
   * Compute A <- A + B x C^T.
   *
   *
   * @param A The result is returned in this matrix, which will be resized to the appropriate dimensions
   * if necessary.
   * @param B Input matrix which is not modified.
   * @param C Input matrix which is not modified. Note that the transpose of this matrix is used in the
   * multiplication.
   */
void mat_multiply_right_transpose_accumulate(MatrixF& A, const MatrixF& B, const MatrixF& C);

// Slow version for verifying correctness.
void mat_multiply_right_transpose_accumulate_naive(MatrixF& A, const MatrixF& B, const MatrixF& C);


/**
   * Sample from a multinomial distribution.
   *
   * Given a vector (Matrix) of pdf values, sample from this distribution and return
   * the index that was chosen. The values in pdf must sum to 1.
   *
   * @parm pdf 1-dim Matrix (i.e., vector) containing the probability distribution
   * function (pdf) to sample from.
   *
   * @return The index of the entry in pdf that was chosen in the random sample.
   * The returned value is in the range [0, length_of_pdf).
   */
int sample_multinomial_distribution(const MatrixF& pdf);


////////////////////////////////////////////////////////////////////////////////////////////
// Network/layer utilities

/**
 * Return the number of errors in the supplied predictions matrix, given the true class labels.
 *
 * The supplied network predictions contains a mini-batch of predictions, where each column
 * in the matrix corresponds to a single prediction and the row index is the class label index.
 * For each column n of \p network_output, compute the row corresponding to the maximum value and
 * then check if target_labels(n) contains the same row index. If so, the network output at column
 * n is considered correct. Otherwise, it is considered an error.
 *
 * @param network_output: An M x N matrix. M is the number of class labels and N is the number of
 *                 output cases. Ideally, exactly one output class should be chosen and therefore
 *                 for each column, the correct row should have value = 1 and all other rows should
 *                 be equal to 0.
 *
 * @param target_labels: A 1-D matrix (array) of class labels of length N. For each element, the value is an integer in
 *              the range [0, N).
 * @return the number of errors in all of the predictions.
 */
int error_count(const MatrixF& predictions, const MatrixI target_labels);


/**
 * Return the number of errors in the supplied predictions matrix and update counts in
 * the supplied confusion matrix.
 *
 * This function performs the same operation as error_count() except that a confusion
 * matrix is also produced.
 *
 * The supplied network predictions consists of a mini-batch of predictions, where each column
 * in the matrix corresponds to a single prediction and the row index is the class label index.
 * For each column n of \p predictions, compute the row index corresponding to the maximum value and
 * then check if target_labels(n) contains the same row index. If so, the network output at column
 * n is considered correct. Otherwise, it is considered an error. In both cases,
 * the prediction result is used to update a count in the supplied confusion matrix.
 *
 *
 * @param predictions an M x N matrix, where M is the number of class labels and N is the
 *     number of examples in the mini-batch.
 * @param target_labels A 1-D matrix (array) of class labels of length N. For each element,
 *     the value is an integer in the range [0, N).
 * @param confusion The confusion matrix which is updated by this function. Before calling
 *     this function for the first time, the user only needs to obtain a new MatrixI and
 *     pass it into this function, as this function will take care of resizing the matrix
 *     to the correct dimensions (M x M).
 *     Each row of this matrix corresponds to the actual class label and each column corresponds
 *     to the predicted class label. For example, confusion(i,j) contains the number of times
 *     that the network predicted class j when the actual (correct) class was i. Thus,
 *     each row i sums to the total number of actual instances of class i.
 * @return the number of errors in all of the predictions.
 */
template <typename T>
int confusion_matrix(const MatrixF& predictions, const MatrixI target_labels, Matrix<T>& confusion){
    const int num_classes = predictions.extent(0);
    const std::vector<int> expected_extents = {num_classes, num_classes};
    if (confusion.get_extents() != expected_extents) {
        confusion.resize(expected_extents);
        set_value(confusion, 0);
    }
    const int test_data_count = predictions.extent(1);
    if (test_data_count != target_labels.extent(0)) {
        error_exit("confusion_matrix(): Inconsistent dimensions. Exiting.");
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


/**
 * Return the number of errors in the supplied predictions matrix and update counts in
 * the supplied confusion matrix.
 *
 * This function performs the same operation as error_count() except that a confusion
 * matrix is also produced.
 *
 *
 * @param predictions a 1-dim matrix of size N is the
 *     number of examples in the mini-batch.
 * @param target_labels A 1-D matrix (array) of class labels of length N. For each element,
 *     the value is an integer in the range [0, N).
 * @param confusion The confusion matrix which is updated by this function. Before calling
 *     this function for the first time, the user only needs to obtain a new MatrixI and
 *     pass it into this function, as this function will take care of resizing the matrix
 *     to the correct dimensions (M x M).
 *     Each row of this matrix corresponds to the actual class label and each column corresponds
 *     to the predicted class label. For example, confusion(i,j) contains the number of times
 *     that the network predicted class j when the actual (correct) class was i. Thus,
 *     each row i sums to the total number of actual instances of class i.
 * @param num_classes The number of class labels.
 * @return the number of errors in all of the predictions.
 */
template <typename T>
int confusion_matrix(const MatrixI& predictions, const MatrixI target_labels, Matrix<T>& confusion,
                     int num_classes){

    const std::vector<int> expected_extents = {num_classes, num_classes};
    if (confusion.get_extents() != expected_extents) {
        confusion.resize(expected_extents);
        set_value(confusion, 0);
    }
    //const int test_data_count = predictions.extent(1);
    if (predictions.size() != target_labels.size()) {
        error_exit("confusion_matrix(): Inconsistent dimensions. Exiting.");
    }
    int errors = 0;
    for (int c = 0; c < predictions.size(); ++c) {
        // Get max value for each column of network_output.

        if (predictions(c) != target_labels(c)) {
            ++errors;
        }
        confusion(target_labels[c], predictions(c)) += 1;
    }
    return errors;
}



/**
 * Compute the maxout of "input" and return the result in "output" and return the corresponding
 * indices of the maximum elements in "state".
 *
 * For each column of "input", the maximum value is taken of each consecutive K rows and the result
 * is written into each consecutive row of "output." Therefore, it is required that "input" have
 * size M x P and "output" have size N x P where K = M/N is an integer. That is, N divides evenly
 * into M.
 *
 * Return the result in "output." If the matrices have inconsistent sizes, exit with an error.
 *
 * @param input An M x P matrix containing P data vectors, each of dimension M.
 * @param output The output N x P matrix containing the P output vectors, each of dimension N = M/K.
 * @param state The N x P state matrix. If the supplied matrix is a different size, it will be
 * resized to the correct dimensions. It is therefore allowed for the user to simply allocate
 * a (size 0) matrix and pass it to this function.
 */
void forward_maxout(const MatrixF& input, MatrixF& output, Matrix<int>& state);

/**
 * For each element in "output_backward", update the corresponding
 * value in "input_backward."
 *
 * In this version, all elements of "input_backward" are updated. The elements of "input_backward" that were chosen as a "max value" are
 * updated with the corresponding max value. All other elements of "input_backward" are set to 0.
 */
void compute_reverse_maxout(MatrixF& input_backward, const MatrixF& output_backward, const Matrix<int>& state, bool accumulate=true);

/**
 * Parameters:
 *
 * decay_val: The penalty for unused weights. Must be in the range [0, 1] where 0 imposes no penalty and 1 imposes the maximum penalty.
 */
void compute_reverse_maxout_decay_unused(MatrixF& input_backward, const MatrixF& input, const MatrixF& output_backward,
                                         const Matrix<int>& state, float decay_val, bool accumulate=true);


/**
 * Compute the forward max-product [1].
 *
 * The forward pass computes the following function:
 *
 * x(i, k) = max_over_j W_i_j*z(j, k)
 *
 * for each k in 1...B where B is the mini-batch size.
 * and for i in 0,...,M-1 where M is the number of output units
 * and for j in 0,...,N-1 where N is the number of input units.
 *
 * This version does not include a bias term.
 *
 * where "x" is a M x B matrix,
 * "W" is a M x N matrix,
 * and "z" is a N x B matrix.
 *
 * The input matrix "z" can be interpreted as containing B input examples, each
 * of dimension N. The output matrix "x" can be interpreted as containing B
 * output vectors, each of dimension M. The matrix "W" can be interpreted as
 * the (learnable) parameters.
 *
 * Note: This version does not support bias.
 *
 * Note also the similarity to a linear layer. In a linear layer, the
 * output is given as a linear combination of the columns of W. That is,
 * after scaling the columns of W by input_forward(m), they are then summed.
 * However, in
 * the max-product layer, after scaling the columns by input_forward(m), the
 * element-wise max() operation is performed instead of the usual sum.
 *
 * The interpretation in terms of neurons is simple:
 * Each (scalar) element x(m,i) can be thought of as the output (i.e., axon) of the
 * neuron. The dendrites (inputs) correspond to the z(m,j) , j=0,...,N-1. So we see that
 * each of the N inputs z(m,j) is multiplied by a corresponding (scalar) weight W_i_j
 * to yeild the product z(m,j)*W_i_j. So, there will be N of these product values arriving
 * into the neuron, which then simply selects the largest one as the output. Note that
 * this operation is even simpler than the linear layer since we only need to identify
 * the largest of the N products and pass it through as the output. Whereas in a linear
 * layer, the sum of these N products would be computed as the output instead.
 *
 * Format of the state matrix:
 * The \p state matrix will be resized to the same dimensions as \p x if it is not already
 * the same size.
 *
 * \p state(i,k) = j_h where j_h in [0, N) maximizes W(i,j)*z(j,k)
 *
 * [1] "Max-product networks" by Brian K. Vogel (July 2016, unpublished).
 *
 * @param x The output matrix of size M x B where B is the mini-batch size.
 * @param W The M x N matrix of parameters.
 * @param z The input matrix of size N x B where B is the mini-batch size.
 * @param state The M x B state matrix. The supplied matrix can be of any size since this
 * function will resize it to the correct size if necessary.
 */
void forward_max_product(MatrixF& x, const MatrixF& W, const MatrixF& z, Matrix<int>& state);

/**
 * Compute weight gradients for the max-product.
 *
 * Note: This version does not support bias.
 *
 * Note: This function assumes that forward_max_product() was previously called and that
 * \p z and \p state have not changed.
 *
 * @param x_grad Gradients (i.e., deltas) for the "x" activations.
 * @param W_grad The computed weights gradients will be placed in this matrix, which must
 * have the same dimensions as W.
 * @param z Input activations.
 * @param state State matrix that was computed in forward_max_product().
 * @param accumulate Accumulate gradients if set to true. Otherwise, do not accumulate. Default is true.
 */
void backward_max_product_parameter_gradient(const MatrixF& x_grad, MatrixF& W_grad,
                                             const MatrixF& z, const Matrix<int>& state,
                                             bool accumulate=true);

/**
 * Compute gradients for the input activations for the max-product.
 *
 * Note: This version does not support bias.
 *
 * Note: This function assumes that forward_max_product() was previously called and that
 * \p z and \p state have not changed.
 *
 * @param x_grad Gradients (i.e., deltas) for the "x" activations.
 * @param W The M x N matrix of parameters.
 * @param z_grad The gradients for the input activations will be returned in this matrix, which
 * must already have the same dimensions as "z."
 * @param state State matrix that was computed in forward_max_product().
 * @param accumulate
 */
void backward_max_product_input_gradient(const MatrixF& x_grad, const MatrixF& W,
                                         MatrixF& z_grad, const Matrix<int>& state,
                                         bool accumulate=true);


/**
 * Compute the forward max-product blend [1].
 *
 * The forward pass computes the following function:
 *
 * x(i, k) = (max_over_j W_i_j*z(j, k)) + alpha*(sum_j_not_equal_j_max W_i_j*z(j,k))
 *
 * where alpha is in [0, 1]. For alpha = 0, this reduces to the max-product, and for alpha
 * =1, this reduces to the standard matarix multiplication. Thus, as alpha approaches 1,
 * the non-linearity decreases, becoming linear at alpha = 1.
 *
 * for each k in 1...B where B is the mini-batch size.
 * and for i in 0,...,M-1 where M is the number of output units
 * and for j in 0,...,N-1 where N is the number of input units.
 *
 * This version does not include a bias term.
 *
 * where "x" is a M x B matrix,
 * "W" is a M x N matrix,
 * and "z" is a N x B matrix.
 *
 * The input matrix "z" can be interpreted as containing B input examples, each
 * of dimension N. The output matrix "x" can be interpreted as containing B
 * output vectors, each of dimension M. The matrix "W" can be interpreted as
 * the (learnable) parameters.
 *
 * Note: This version does not support bias.
 *
 * Note also the similarity to a linear layer. In a linear layer, the
 * output is given as a linear combination of the columns of W. That is,
 * after scaling the columns of W by input_forward(m), they are then summed.
 * However, in
 * the max-product layer, after scaling the columns by input_forward(m), the
 * element-wise max() operation is performed instead of the usual sum.
 *
 * The interpretation in terms of neurons is simple:
 * Each (scalar) element x(m,i) can be thought of as the output (i.e., axon) of the
 * neuron. The dendrites (inputs) correspond to the z(m,j) , j=0,...,N-1. So we see that
 * each of the N inputs z(m,j) is multiplied by a corresponding (scalar) weight W_i_j
 * to yeild the product z(m,j)*W_i_j. So, there will be N of these product values arriving
 * into the neuron, which then simply selects the largest one as the output. Note that
 * this operation is even simpler than the linear layer since we only need to identify
 * the largest of the N products and pass it through as the output. Whereas in a linear
 * layer, the sum of these N products would be computed as the output instead.
 *
 * Format of the state matrix:
 * The \p state matrix will be resized to the same dimensions as \p x if it is not already
 * the same size.
 *
 * \p state(i,k) = j_h where j_h in [0, N) maximizes W(i,j)*z(j,k)
 *
 * [1] "Max-product networks" by Brian K. Vogel (July 2016, unpublished).
 *
 * @param x The output matrix of size M x B where B is the mini-batch size.
 * @param W The M x N matrix of parameters.
 * @param z The input matrix of size N x B where B is the mini-batch size.
 * @param state The M x B state matrix. The supplied matrix can be of any size since this
 * function will resize it to the correct size if necessary.
 * @param alpha Value in the range [0, 1] which interpolates between max-product (alpha=0) and linear
 * layer (alpha=1).
 */
void forward_max_product_blend(MatrixF& x, const MatrixF& W, const MatrixF& z, Matrix<int>& state, float alpha);

/**
 * Compute weight gradients for the max-product blend.
 *
 * Note: This version does not support bias.
 *
 * Note: This function assumes that forward_max_product() was previously called and that
 * \p z and \p state have not changed.
 *
 * @param x_grad Gradients (i.e., deltas) for the "x" activations.
 * @param W_grad The computed weights gradients will be placed in this matrix, which must
 * have the same dimensions as W.
 * @param z Input activations.
 * @param state State matrix that was computed in forward_max_product().
 * @param alpha Value in the range [0, 1] which interpolates between max-product (alpha=0) and linear
 * layer (alpha=1).
 * @param accumulate Accumulate gradients if set to true. Otherwise, do not accumulate. Default is true.
 */
void backward_max_product_blend_parameter_gradient(const MatrixF& x_grad, MatrixF& W_grad,
                                                   const MatrixF& z, const Matrix<int>& state,
                                                   float alpha, bool accumulate=true);

/**
 * Compute gradients for the input activations for the max-product blend.
 *
 * Note: This version does not support bias.
 *
 * Note: This function assumes that forward_max_product() was previously called and that
 * \p z and \p state have not changed.
 *
 * @param x_grad Gradients (i.e., deltas) for the "x" activations.
 * @param W The M x N matrix of parameters.
 * @param z_grad The gradients for the input activations will be returned in this matrix. If
 * the dimensions are different than expected, it will be resized to the required dimensions
 * and initialized with zeros and then the gradients will be accumulated.
 *
 * @param state State matrix that was computed in forward_max_product().
 * @param alpha Value in the range [0, 1] which interpolates between max-product (alpha=0) and linear
 * layer (alpha=1).
 * @param accumulate
 */
void backward_max_product_blend_input_gradient(const MatrixF& x_grad, const MatrixF& W,
                                               MatrixF& z_grad, const Matrix<int>& state,
                                               float alpha, bool accumulate=true);



/*
   * Compute the forward-direction k-max operation independently on each column of kmax_in.
   *
   * Intiution: This activation function corresponds to "forced sparsity." A given column (corresponding to 1 training or test sample)
   * can be partitioned into 1 or more partition regions, specified by partition_count. This type of sparsity corresponds to threshlding all but the largest k
   * values in each partition region to 0.
   *
   * kmax_in: An M x N matrix. The n'th column is assumed to contain the intput to the activation function (
   * that is, this function) for
   * the n'th example. Typically, N is the mini-batch size. Each column as dimension M. We partition each column into partition_count partitions.
   * M/partition_count must be an integer or else the program will exit with an error.
   * Within the p'th partition, we find the k largest values, storing the corresponding indices in
   * "state" and storing the corresponding values in "output." All values other than
   * the largest k values will be set to 0 in "output."
   * The sparsity ratio is then
   * M/(partition_count*k) where M is the number of rows in kmax_in.
   * The value k of course must not be larger than the ratio M/partition_count.
   *
   * Parameters:
   *
   * output: Same size as kmax_in (i.e., an M x N matrix). Within each partition, only the largest
   *    k values are stored and all other values are set to 0.
   *
   * state: A k*partition_count x N matrix. rmax = kmax_out_indices(r, n) contains the row index in both
   * kmax_in and kmax_out_values of one of the k largest values. That is kmax_in(rmax, n) will then correspond
   * to one of the k largest values.
   *
   * partition_count: The number of partitions for each column of kmax_in. All partitions are the same size:
   *   parition_size = M/partition_count.
   *   The partition_count values must be in [1...M].
   *
   * k: The number of largest values to keep from each column partition. Must be in the range [1...M/partition_count].
   *
   */
// deprecated and buggy. col out of bound error.
void compute_forward_kmax(const MatrixF& kmax_in, MatrixF& kmax_out_values, Matrix<int>& kmax_out_indices,
                          int partition_count, int k);

// Same as compute_forward_kmax() except kmax_out_indices is same size as kmax_in.
void compute_forward_kmax_v2(const MatrixF& kmax_in, MatrixF& kmax_out_values, Matrix<int>& kmax_out_indices,
                             int partition_count, int k);


/**
 * This activation function partitions the activation units into \p partition_count partitions.
 * Within each partition, the \p k largest values are passed through unmodified. The remaining
 * values are scaled by a "leakyness" parameter \p alpha which typicall is a small positive number.
 *
 * @brief forward_leaky_kmax
 * @param kmax_in The M x N input matrix to the activation function, where M is the number of
 * activation units and N is the mini-batch size.
 * @param kmax_out_values The output M x N matrix of the activation function.
 * @param kmax_out_indices The state matrix, which will be set to the appropriate dimensions by
 * this function.
 * @param partition_count The number of partitions. For M activations, the ratio M/partition_count
 * must be an integer.
 * @param k The number of largest values to keep within each partition.
 * @param alpha "Leaky" scaling factor. All values that are not in the k-largest set will be scaled by
 * this value.
 */
void forward_leaky_kmax(const MatrixF& kmax_in, MatrixF& kmax_out_values, MatrixI& kmax_out_indices,
                        int partition_count, int k, float alpha=0.02);

/*
   * Same as compute_forward_kmax() except that kmax_in is updated using the values in kmax_out_values and
   * kmax_out_indices.
   *
   * When updating kmax_in, all values that do not correspond to one of the k largest values are set to 0.
   * Thus, all values will be overwritten, either with 0 or with a largest value from kmax_out_values.
   */
void compute_reverse_kmax(MatrixF& kmax_in, const MatrixF& kmax_out_values, const Matrix<int>& kmax_out_indices,
                          int partition_count, int k);

// fixme: change name: kmax_out_indices -> kmax_state.
// Same as compute_reverse_kmax() except kmax_out_indices is same size as kmax_in.
void compute_reverse_kmax_v2(MatrixF& kmax_in, const MatrixF& kmax_out_values, const Matrix<int>& kmax_out_indices,
                             int partition_count, int k);


/*
   * Same as kmax_v2 except that the input activations ("inputs") that were computed during the forward pass also need to
   * be provided. The supplied "input_backward" that represent the error gradients with respect to the input activations are
   * updated.
   *
   * This is basically, kmax with weight decay (actually, activation decay) for the activations that did not make it into the
   * kmax set during the previous forward pass.
   *
   * This version penalizes the input activations that were not part of the k-max during the previous forward pass. Note that
   * these input activations had no effect on the output acctivations during the forward pass since they were too small (but
   * generally still nonzero). Ideally, for maximum efficiency, these unused activations were useless and so we might wish
   * them to be close to zero. This function therefore assigns a nonzero error gradient to these "useless" input activations.
   * The error gradient is just selected to be the same as the value of the corresponding activation from the forward pass.
   * Then, if we used SGD with a learning rate of 1 and updated the input activations from the forward pass, all of the "useless"
   * activations would become zero. Thus, the error gradient computed in this way tries to push these "useless" activations to be closer to zero.
   *
   * This same idea can also be extended to ReLU and Maxout activations (todo).
   *
   * Parameters:
   *
   * decay_val: The penalty for unused weights. Must be in the range [0, 1] where 0 imposes no penalty and 1 imposes the maximum penalty.
   */
void compute_reverse_kmax_decay_unused(MatrixF& input_backward, const MatrixF& inputs, const MatrixF& output_backward, const MatrixI& state,
                                       int partition_count, int k, float decay_val);

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
   * input: (minibatch_size x depth x height x width) matrix containing the input values.
   *
   * output: (minibatch_size x depth x height x width) matrix containing the output values.
   *
   * state: (minibatch_size x depth x height x width) matrix containing the state information.
   */
void forward_3d_kmax(const MatrixF& input, MatrixF& output, Matrix<int>& state,
                     int box_depth, int box_height, int box_width, int k);

/*
   * This is the reverse-direction version of compute_forward_3d_kmax().
   *
   * Update the values in kmax_in given the values in kmax_out and the state information in kmax_state.
   *
   */
void reverse_3d_kmax(MatrixF& input_backward, const MatrixF& output_backward, const Matrix<int>& state);

void reverse_3d_kmax_decay_unused(MatrixF& input_backward, const MatrixF& input, const MatrixF& output_backward,
                                  const Matrix<int>& state, float decay_val);


/**
 * Forward-direction ReLU (Rectified Linear Unit) activation function.
 *
 * The output values are placed into the matrix "output" and the corresponding state information
 * is placed into "state".
 *
 * The function computed is output[i] = max(0, input[i]) for all indices i in the backing array.
 * This function also updates "state" so that
 * state[i] = 1 if output[i] > 0. Otherwise, state[i] = 0. The "state" matrix
 * will be needed by the function compute_reverse_relu().
 *
 * All supplied matrices should have the same size. Otherwise the program will exit with an error.
 *
 * Parameters:
 *
 * @param input An N-dimensional matrix.
 *
 * @param output An N-dimensional matrix of the same extents as in_vals. It will be resized if
 *        not already the required extents.
 *
 * @param state An N-dimensional matrix of the same size as in_vals. This is used for storing
 *              state information that will be needed later by the reverse-direction relu function.
 *              This does not need to be initialized to
 *              any particular values since calling this function will overwrite its contents anyway.
 */
// todo: auto-resize output and state.
void compute_forward_relu(const MatrixF& input, MatrixF& output, Matrix<int>& state);

/**
 * Same as compute_forward_relu(), except this is the "leaky" version with hard-coded leakyness
 * parameter (see code for value).
 */
// todo: auto-resize output and state.
void compute_forward_leaky_relu(const MatrixF& input, MatrixF& output, Matrix<int>& state);

/**
 * Forward-direction tanh activation function
 *
 * output <- tanh(input)
 *
 * @param input The input N-dimensional matrix.
 * @param output The output N-dimensional matrix. This matrix will be resized to the same size as
 *  "input" if necessary.
 */
void compute_forward_tanh(const MatrixF& input, MatrixF& output);

/**
   * Compute the reverse-direction tanh activation function
   *
   * This function computes the derivates as a function of the output activations computed during the
   * forward pass. Note that "output_forward" consists of the tanh values from the forward pass, not
   * the error gradients of the outputs.
   *
   * @param input_backward The input N-dimensional matrix. This matrix will be resized to the same size as
   *  "output" if necessary.
   * @param output_forward The tanh function values that were computed during the forward pass.
   * @param output_backward The error gradients corresponding to output_forward.
   */
void compute_reverse_tanh(MatrixF& input_backward, const MatrixF& output_forward, const MatrixF& output_backward, bool accumulate=true);

/**
   * Compute the forward-direction sigmoid activation function
   *
   * output <- sigmoid(input)
   *
   * @param input The input N-dimensional matrix.
   * @param output The output N-dimensional matrix. This matrix will be resized to the same size as
   *  "input" if necessary.
   */
void compute_forward_sigmoid(const MatrixF& input, MatrixF& output);

/**
   * Compute the reverse-direction sigmoid activation function
   *
   * This function computes the derivates as a function of the output activations computed during the
   * forward pass. Note that "output_forward" consists of the sigmoid values from the forward pass, not
   * the error gradients of the outputs.
   *
   * @param input_backward The input N-dimensional matrix. This matrix will be resized to the same size as
   *  "output" if necessary.
   * @param output_forward The sigmoid function values that were computed during the forward pass.
   * @param output_backward The error gradients corresponding to output_forward.
   */
void compute_reverse_sigmoid(MatrixF& input_backward, const MatrixF& output_forward, const MatrixF& output_backward, bool accumulate=true);

/*
   * The identify function activation, forward direction.
   *
   * Used for gradient checking and debugging.
   */
void compute_forward_identity_activation(const MatrixF& in_vals, MatrixF& out_vals, Matrix<int>& out_indices);

/*
   * The identify function activation, reverse direction.
   *
   * Used for gradient checking and debugging.
   */
void compute_reverse_identity_activation(MatrixF& in_vals, const MatrixF& out_vals, const Matrix<int>& out_indices, bool accumulate=true);

/*
   * Same as compute_forward_relu() except compute the reverse-direction ReLu to update "input_backward"
   * from "output_backward" and "state".
   */
void compute_reverse_relu(MatrixF& input_backward, const MatrixF& output_backward, const Matrix<int>& state, bool accumulate=true);

/*
   * Same as compute_forward_relu() except compute the reverse-direction ReLu to update "input_backward"
   * from "output_backward" and "state".
   *
   * Parameters:
   *
   * decay_val: The penalty for unused weights. Must be in the range [0, 1] where 0 imposes no penalty and 1 imposes the maximum penalty.
   */
void compute_reverse_relu_decay_unused(MatrixF& input_backward, const MatrixF& input, const MatrixF& output_backward,
                                       const Matrix<int>& state, float decay_val);

void compute_reverse_leaky_relu(MatrixF& in_vals, const MatrixF& out_vals, const Matrix<int>& out_indices, bool accumulate=true);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Various SGD-related functions:


/**
   * Given matrices X_error, W, H which are related according to
   *
   * X_pred = W * H + bias
   *
   * or without bias (it makes no difference to gradients of W):
   *
   * X_pred = W * H
   *
   * and
   *
   * X_error is the corresponding error matrix (deltas) from backpropagation
   * or some other method, compute the gradient matrix for W, W_grad.
   *
   * If "accumulate" is true (which is the default), compute:
   *
   * grad_W += X_error*H^T / mini_batch_size
   *
   * Otherwise, if "accumulate" is false, compute:
   *
   * grad_W = X_error*H^T / mini_batch_size
   *
   * This function does not compute the mean gradient. To get the mean gradient, you should do element-wise divide by the mini-batch size.
   *
   * @param X_error
   * @param W_grad The output gradients matrix.
   * @param H
   * @param accumulate Accumulate gradients if set to true. Otherwise, do not accumulate. Default is true.
   */
void compute_weight_grad_sgd_minibatch(const MatrixF& X_error, MatrixF& W_grad, const MatrixF& H, bool accumulate=true);

/**
 * Update the weights matrix W using the gradient W_grad and the learning rate alpha.
 *
 * W = W - alpha*W_grad
 *
 * is computed element-wise.
 *
 */
void update_parameters_sgd(MatrixF& W, const MatrixF& W_grad, float alpha);

/**
 * Set weight/activatin decay to the specified value.
 *
 * The weight decay for each weight or activation w_i is set according to:
 *
 * w_i = w_i - decay_val*w_i
 */
void update_parameters_from_decay(MatrixF& W, float decay_val);

/**
 * Update the weights matrix W using the gradient W_grad and the learning rate alpha and
 * weight decay lambda.
 *
 * W = W - alpha*W_grad -alpha*lambda*W
 *
 * is computed element-wise.
 *
 */
void update_parameters_sgd(MatrixF& W, const MatrixF& W_grad, float alpha, float lambda);

/**
 * Update the weights matrix W using the gradient W_grad and the learning rate alpha.
 *
 * W = W - alpha*W_grad + ...
 *
 * is computed element-wise.
 *
 */
void update_parameters_sgd(MatrixF& W, const MatrixF& W_grad, float alpha, float lambda,
                                  float sparsity_param, bool force_nonnegative);


/**
 * Rmsprop.
 *
 * @param W The parameters matrix.
 * @param W_grad The gradients matrix.
 * @param W_grad_mean_square Used to store running statistics. It will be set to the appropriate extents by
 * this function and so it is fine for the caller to pass an uninitialized matrix.
 * @param alpha in [0, 1].
 * @param rho in [0, 1].
 * @param epsilon A small value for numerical stability.
 */
void update_parameters_rmsprop(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_mean_square,
                                             float alpha, float rho, float epsilon=1e-8f);


/**
 * Rmsprop with momentum
 *
 *
 * @param W The parameters matrix.
 * @param W_grad The gradients matrix.
 * @param W_grad_mean_square Used to store running statistics. It will be set to the appropriate extents by
 * this function and so it is fine for the caller to pass an uninitialized matrix.
 * @param W_momentum Used to store running statistics. It will be set to the appropriate extents by
 * this function and so it is fine for the caller to pass an uninitialized matrix.
 * @param alpha in [0, 1].
 * @param rho in [0, 1].
 * @param momentum in [0, 1].
 * @param epsilon A small value for numerical stability.
 */
void update_parameters_rmsprop_momentum(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_mean_square,
                                                   MatrixF W_momentum, float alpha, float rho, float momentum,
                                                   float epsilon=1e-8f);


/**
 * Adagrad.
 *
 * @param W The parameters matrix.
 * @param W_grad The gradients matrix.
 * @param W_grad_sum_square Used to store running statistics. It will be set to the appropriate extents by
 * this function and so it is fine for the caller to pass an uninitialized matrix.
 * @param alpha
 */
void update_parameters_adagrad(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_sum_square,
                                          float alpha);

/*
   * Gaussian kernel function.
   */
float gaussian_kernel(const std::vector<float>& x1, const std::vector<float>& x2, float sigma);




/**
   * Compute the input layer gradients for a mini-batch of data in a linear layer.
   *
   * The linear model is
   *
   * X = W*H + bias
   *
   * where X is one mini-batch of output activations, W is the weights matrix, and H is one mini-batch of input activations.
   * The bias term is optional.
   *
   * Given the gradients with respect to the output, X_error, this function computes the gradients with respect to
   * the inputs, H_error as
   *
   * H_error <- W^T * X_error
   *
   * if "accumulate" is set to false.
   *
   * If "accumulate" is set to true (the default value), then the errors are computed as
   *
   *  H_error <- H_error + W^T * X_error
   *
   * @param X_error A (N x minibatch_size) matrix containing the output errors (deltas) where N is the size of the output layer.
   * @param W A (N x M) weights matrix.
   * @param H_error A (M x minibatch_size) matrix containing the input errors, which are computed by this function. The
   * supplied matrix will be resized to the correct dimensions if necessary.
   */
void do_backprop_update_sgd_minibatch(const MatrixF& X_error, const MatrixF& W, MatrixF& H_error, bool accumulate=true);

/**
   * Compute the gradients for the bias vector (a 1-dim Matrix) and return it in b_grad.
   *
   * @param X_error The 2-dimensional errors matrix of size M x minibatch_size.
   * @param b_grad The 1-dimensional output gradients matrix of size M.
   * @param accumulate Accumulate gradients if set to true. Otherwise, do not accumulate. Default is true.
   */
void compute_bias_grad_sgd_minibatch(const MatrixF& X_error, MatrixF& b_grad, bool accumulate=true);

/*
   * Use the current W and H to compute the product W*H and update X.
   *
   * Compute X = update_weight*W*H + (1 - udpate_weight)*X
   *
   * update_weight: Should be in the range (0, 1). It controls the relative weights of
   * the previous and new values. If 0, just use the old value. If 1, just use the
   * new value. If 0.1, use 0.1*new values + 0.9*old_value.
   */
void do_product_update_naive(MatrixF& X, const MatrixF& W, const MatrixF& H, float update_weight);

// Calls an optimized impelmentation.
void do_product_update(MatrixF& X, const MatrixF& W, const MatrixF& H, float update_weight);


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
void do_product_update(MatrixF& X, const MatrixF& W, const MatrixF& H, const MatrixF& b);

/*
   * Compute the root mean squared error (RMSE) for the factorization approximation
   *
   * X approx= W * H
   *
   * and return it.
   *
   * @return The current RMSE.
   */
float compute_reconstruction_error(const MatrixF& X, const MatrixF& W, const MatrixF& H);

/*
   * Given a matrix of error values, compute and return the RMSE.
   */
float compute_rmse(const MatrixF& X);

/*
   * Given two matrices, compute and return the RMSE of their element-wise differences.
   */
float compute_rmse(const MatrixF& A, const MatrixF& B);

/*
   * If abs(a -b) > tolerance, exit with an error.
   */
void assert_almost_equal(float a, float b, float tolerance=1.0e-3f);

/*
   * For each element of the two suplied matrices, which must be of the same size,
   * test if the magnitude of the difference exceeds the tolerance. If so,
   * exit with an error.
   */
void assert_almost_equal(const MatrixF& A, const MatrixF& B, float tolerance=1.0e-3f);


/*
   * Print some basic statistics to stdout for the supplied matrix with the
   * supplied name.
   *
   * The statistics include mean, std deviation, etc.
   */
void print_stats(const MatrixF& A, std::string name);



/**
 * Convolve several 3D filters with a 3D matrix. This corresponds to the usual convolution operation in a convnet.
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
 * minibatch_size = number of samples in one mini-batch.
 * R = number of convolution filters.
 * M = height of input image.
 * N = width of input image.
 * D = depth of input image = depth of convolution filter (for color images, D = 3, but for hidden layers, D can be much larger).
 * P = height of convolution filter.
 * Q = width of convolution filter.
 */
void convolve_3d_filter_with_bias_minibatch(MatrixF& Z2, const MatrixF& W, const MatrixF& bias, const MatrixF& A1);

/**
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
void convolve_3d_filter_with_bias_minibatch_optimized(MatrixF& Z2, const MatrixF& W,
                                                      const MatrixF& bias, const MatrixF& A1,
                                                      MatrixF& temp_Z2, MatrixF& temp_A1,
                                                      MatrixF& temp_W);


/**
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
void compute_convolutive_deltas_minibatch(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2);

/**
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
void compute_3d_convolutive_deltas_minibatch(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2);


/**
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
void compute_3d_convolutive_deltas_minibatch_optimized(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2,
                                                       MatrixF& temp_deltas_Z2, MatrixF& temp_deltas_A1, MatrixF& temp_W);

/**
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
void compute_convolutive_deltas_minibatch_optimized(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2,
                                                    MatrixF& temp_deltas_Z2, MatrixF& temp_deltas_A1, MatrixF& temp_W);


/**
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
void compute_weight_grad_convolutive_minibatch(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1);

/**
 * This function performs the "update weight gradients" back-propagation step corresponding
 * to the forward propagation performed by convolve_3d_filter_with_bias_minibatch().
 *
 * Parameters:
 *
 * @param grad_W: Input matrix of same size as W: R x D x P x Q matrix containing R filter gradient matrices of size D x P x Q.
 *
 * @param deltas_Z2: Input matrix of same size as Z2: minibatch_size x R x M x N.
 *
 * @param A1: Input matrix of size minibatch_size x D x M x N.
 *
 * @param accumulate Accumulate gradients if set to true. Otherwise, do not accumulate. Default is true.
 *
 * Note: This function computes the actual gradient of W. However, for SGD updates, the average (mean) gradient
 * might be desired. The mean gradient can be obtained by scaling the returned
 * gradient by 1/(deltas_Z2.dim0*deltas_Z2.dim1*minibatch_size).
 *
 */
void compute_3d_weight_grad_convolutive_minibatch(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1, bool accumulate=true);


/**
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
void compute_weight_grad_convolutive_minibatch_optimized(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1,
                                                         MatrixF& temp_deltas_Z2, MatrixF& temp_A1,
                                                         MatrixF& temp_grad_W);

/**
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
 * @param accumulate Accumulate gradients if set to true. Otherwise, do not accumulate. Default is true.
 * If any of the supplied matrices have inconsitent sizes, exit with an error.
 */
void compute_3d_weight_grad_convolutive_minibatch_optimized(MatrixF& grad_W, const MatrixF& deltas_Z2, const MatrixF& A1,
                                                            MatrixF& temp_deltas_Z2, MatrixF& temp_A1,
                                                            MatrixF& temp_grad_W, bool accumulate=true);


/**
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
 * @param accumulate Accumulate gradients if set to true. Otherwise, do not accumulate. Default is true.
 * If any of the supplied matrices have inconsitent sizes, exit with an error.
 */
void compute_bias_grad_convolutive_minibatch(MatrixF& grad_bias, const MatrixF& deltas_Z2, bool accumulate=true);


/**
 * Convert a mini-batch of examples from image format to vector format.
 *
 * B is a multi-dimensional matrix of size minibatch_size x dim1 x dim2 ... dimR. This typically
 * corresponds to one minibatch of output activations from a convolutional layer.
 *
 * This function simply copies data from B to A, converting between the too formats. Note that
 * A and B will therefore contain the same number of elements. If the supplied "A" will be resized
 * if necessary to satisfy this requirement.
 *
 * A: Size P x minibatch_size matrix where P = (dim1 x dim2 ... dimR) of "B".
 * This typically corresponds to one mini-batch of input activations
 * to a fully-connected layer. If the supplied A" does not have these dimensions, it
 * will be resized to the correct dimensions.
 *
 * B: Size minibatch_size x dim1 x dim2 ... dimR matrix.
 *
 */
void multi_dim_minibatch_to_column_minibatch(MatrixF& A, const MatrixF&B);


/**
 * Convert a mini-batch of examples from vector format to image format.
 *
 * Same as multi_dim_minibatch_to_column_minibatch() except copies in the opposite
 * direction.
 *
 * Parameters:
 *
 * A: It is assumed that "A" already has been set to a size that is consistent with
 *    the size of the supplied "B." Otherwise, an error will occur.
 *
 * B: The input matrix that will be copied into "A."
 */
void column_minibatch_to_multi_dim_minibatch(const MatrixF& A, MatrixF&B);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tensor operations and math functions


/**
 * Use method from: http://cs231n.github.io/neural-networks-3/
 *
 * |A - B|
 * ---------
 * max(|A|, |B|)
 *
 * Note: do not use this method if we expect both matrices to be nearly 0.
 */
float relative_error(const MatrixF& A, const MatrixF& B);

/**
 * For each element x_i of X, set x_i = max(x_i, min_val).
 */
// todo: make templated
void threshold_lower(MatrixF& X, float min_val);

/**
 * For each element x_i of X, set x_i = min(x_i, min_val).
 */
// todo: make templated
void threshold_upper(MatrixF& X, float max_val);

/**
 * Clip the entries in the supplied matrix to have the specified range.
 *
 * @param min_val The minimum value. Any smaller values will be clipped to this value.
 * @param max_val The maximum value. Any larger values will be clipped to this value.
 */
template <typename T>
void clip_to_range(Matrix<T>& A, T min_val, T max_val) {
    map1(A, A, [=](T x){
        if (x > max_val) {
            return max_val;
        } else if (x < min_val) {
            return min_val;
        } else {
            return x;
        }
    });
}


/**
 * Normalize X so that each column sums to 1.
 * @param X
 */
void normalize_columns_unit_sum(MatrixF& X);


void normalize_columns_unit_max_val(MatrixF& X);

}

#endif  /* _UTILITIES_H */
