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
//#include <functional>
#include "Matrix_list.h"
#include <algorithm>

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

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Error-related

  /**
   * Print an error message and exit.
   *
   * @param reason Reason for error.
   */
  void error_exit(std::string reason);

  /**
   * Depending on the debug mode, do nothing or print a message that something was resized.
   *
   * Several utility functions will resize the output matrix if necessary, making these functions
   * more convinient to use. However, such resizing is bad for performance and should be avoided
   * after the initialization stage.
   *
   * For debugging purposes, when in "debug" mode, this function will simply print a message
   * to std out stating that a resize operation has occured. If this messages continue to
   * be displayed after initialization, there is likely a bug. In this case, it is recommended
   * to set a break point in this function and check the stack trace to find where the unintended
   * resizing is occuring.
   */
  void resized();

  ////////////////////////////////////////////////////////////////////////////////////////////
  // Matrix Utilities

  /*
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

  /*
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

  /*
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

  /*
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

  /*
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
      std::cerr << "select(): Invalid dimension." << std::endl;
      exit(1);
    }
    const int order = inmat.order();
    if (0 == order) {
      std::cerr << "select(): order too small." << std::endl;
      exit(1);
    } else if (1 == order) {
      std::cerr << "select(): order too small." << std::endl;
      exit(1);
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
      resized();
    }

    if (out_order == 1) {
      looper_1_dim(outmat.get_extents(), [&] (int i) {
          if (dimension == 0) {
            outmat(i) = inmat(index, i);
          } else if (dimension == 1) {
            outmat(i) = inmat(i, index);
          } else {
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
          }
        });
    } else {
      // Add support for larger dimension values on an as-needed basis.
      std::cerr << "select(): Sorry, this dimension value not yet supported. ";
      std::cerr << "Ask Brian." << std::endl;
      exit(1);
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
      std::cerr << "select(): Invalid dimension." << std::endl;
      exit(1);
    }
    const int order = inmat.order();
    if (0 == order) {
      std::cerr << "select(): order too small." << std::endl;
      exit(1);
    } else if (1 == order) {
      std::cerr << "select(): order too small." << std::endl;
      exit(1);
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
   * Return a MatrixF in "submat" that is the sub-matrix in "fullmat" where "dimension"
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
   * Then narrow(fullmat, 0, 1, 2) will return a 2 x 2 "submat" of:
   * [3 4]
   * [5 6]
   *
   * And narrow(fullmat, 1, 0, 1) will return a 4 x 1 "submat" of:
   * [1]
   * [3]
   * [5]
   * [7]
   */
  template <typename T>
    void narrow(Matrix<T>& submat, const Matrix<T>& fullmat, int dimension, int index, int size) {
    if (dimension >= fullmat.order()) {
      std::cerr << "narrow(): Invalid dimension." << std::endl;
      exit(1);
    }
    const int order = fullmat.order();
    if (0 == order) {
      std::cerr << "narrow(): order too small." << std::endl;
      exit(1);
    }

    std::vector<int> submat_extents = fullmat.get_extents();
    submat_extents.at(dimension) = size;

    if (submat_extents != submat.get_extents()) {
      submat.resize(submat_extents);
      resized();
    }

    if (order == 1) {
      looper_1_dim(submat.get_extents(), [&] (int i) {
          if (dimension == 0) {
            submat(i) = fullmat(index + i);
          } else {
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
          }
        });
    } else if (order == 2) {
      looper_2_dim(submat.get_extents(), [&] (int i, int j) {
          if (dimension == 0) {
            submat(i, j) = fullmat(index + i, j);
          } else if (dimension == 1) {
            submat(i, j) = fullmat(i, index + j);
          } else {
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
          }
        });
    } else {
      // Add support for larger dimension values on an as-needed basis.
      std::cerr << "narrow(): Sorry, this dimension value not yet supported. ";
      std::cerr << "Ask Brian." << std::endl;
      exit(1);
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
      std::cerr << "narrow(): Invalid dimension." << std::endl;
      exit(1);
    }
    const int order = fullmat.order();
    if (0 == order) {
      std::cerr << "narrow(): order too small." << std::endl;
      exit(1);
    } //else if (1 == order) {
    //std::cerr << "narrow(): order too small." << std::endl;
    //exit(1);
    //}

    std::vector<int> submat_extents = fullmat.get_extents();
    submat_extents.at(dimension) = size;
    Matrix<T> submat(submat_extents);
    narrow(submat, fullmat, dimension, index, size);
    return submat;
  }


  /*
   * Return a MatrixF in "submat" that is the sub-matrix in "fullmat" where "dimension"
   * has its extent narrowed to "size." The elements corresponding to the "size"
   * possible index values in "dimension" of "submat" correspond to the same
   * "size" values obtained by selecting "permuted_indices(index)" to "permuted_indices(index + size -1)" in "fullmat."
   * Thus, this is the same as narrow() except the the index into "fullmat" is obtained by
   * looking up the index in the "permuted_indices" vector. Note that it is required that
   * "permuted_indices" have the same size as the extent of "dimension" in "fullmat" and that
   * the values in "permuted_indices" must correspond to valid indices into fullmat. It is
   * not required that they correspond to an actual permutation, however.
   *
   */
  template <typename T>
    void narrow_permuted(Matrix<T>& submat, const Matrix<T>& fullmat, int dimension, int index, int size,
                         const std::vector<int>& permuted_indices) {
    if (dimension >= fullmat.order()) {
      std::cerr << "narrow_permuted(): Invalid dimension." << std::endl;
      exit(1);
    }
    const int order = fullmat.order();
    if (0 == order) {
      std::cerr << "narrow_permuted(): order too small." << std::endl;
      exit(1);
    }

    std::vector<int> submat_extents = fullmat.get_extents();
    submat_extents.at(dimension) = size;

    if (submat_extents != submat.get_extents()) {
      std::cerr << "narrow_permuted(): Supplied submat matrix has wrong dimensions." << std::endl;
    }

    if (order == 1) {
      looper_1_dim(submat.get_extents(), [&] (int i) {
          if (dimension == 0) {
            submat(i) = fullmat(permuted_indices[index + i]);
          } else {
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
          }
        });
    } else if (order == 2) {
      looper_2_dim(submat.get_extents(), [&] (int i, int j) {
          if (dimension == 0) {
            submat(i, j) = fullmat(permuted_indices[index + i], j);
          } else if (dimension == 1) {
            submat(i, j) = fullmat(i, permuted_indices[index + j]);
          } else {
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
          }
        });
    } else {
      // Add support for larger dimension values on an as-needed basis.
      std::cerr << "narrow(): Sorry, this dimension value not yet supported. ";
      std::cerr << "Ask Brian." << std::endl;
      exit(1);
    }
  }


  /*
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
      std::cerr << "reverse_narrow(): Invalid dimension." << std::endl;
      exit(1);
    }
    const int order = fullmat.order();
    if (0 == order) {
      std::cerr << "reverse_narrow(): order too small." << std::endl;
      exit(1);
    } else if (1 == order) {
      std::cerr << "reverse_narrow(): order too small." << std::endl;
      exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
          }
        });
    } else if (order == 2) {
      looper_2_dim(submat.get_extents(), [&] (int i, int j) {
          if (dimension == 0) {
            fullmat(index + i, j) = submat(i, j);
          } else if (dimension == 1) {
            fullmat(i, index + j) = submat(i, j);
          } else {
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
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
            std::cerr << "Bad dim size" << std::endl;
            exit(1);
          }
        });
    } else {
      // Add support for larger dimension values on an as-needed basis.
      std::cerr << "narrow(): Sorry, this dimension value not yet supported. ";
      std::cerr << "Ask Brian." << std::endl;
      exit(1);
    }
  }


  /*
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

  /*
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

  /*
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
      std::cerr << "map1(): wrong matrix size." << std::endl;
      exit(1);
    }
#pragma omp parallel for
    for (int i = 0; i < outmat.size(); i++) {
      outmat[i] = func(inmat[i]);
    }
  }

  /*
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
   * The number of elements in all supplied matrices must be the same but the dimensions may
   * be different.
   */
  template <typename T, typename Func>
    void map2(Matrix<T>& outmat, const Matrix<T>& inmat1, const Matrix<T>& inmat2, Func func) {
    if ((outmat.size() != inmat1.size()) || (outmat.size() != inmat2.size())) {
      error_exit("map2(): wrong matrix size.");
    }
#pragma omp parallel for
    for (int i = 0; i < outmat.size(); i++) {
      outmat[i] = func(inmat1[i], inmat2[i]);
    }
  }

  /*
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
      std::cerr << "map3(): wrong matrix size." << std::endl;
      exit(1);
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
      std::cerr << "copy_column_to_matrix(): Inconsistent parameter values." << std::endl;
      exit(1);
    }
    for (int i = 0; i != B.size(); ++i) {
      B[i] = A(i, column);
    }
  }

  /*
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

  /*
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

  /*
   * Return the maximum value.
   */
  template <typename T>
    T max_value(const Matrix<T>& A) {
    return max_value(A.get_backing_vector());
  }

  /*
   * Return the minimum value.
   */
  template <typename T>
    T min_value(const Matrix<T>& A) {
    return min_value(A.get_backing_vector());
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
      resized();
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
      resized();
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
      resized();
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
      resized();
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
      resized();
    }
    map1(A, B, [] (T b) {
        return b*b;
      });
  }

  /**
   * Compute the sum of all elements in <i>A</i> and return it.
   *
   * @param A The input matrix.
   * @return The sum of all elements in A
   */
  template <typename T>
    T sum(Matrix<T>& A) {
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
    const int rowsA = A.extent(0);
    const int colsA = A.extent(1);
    if ((rowsA != colsB) || (colsA != rowsB)) {
      A.resize(colsB, rowsB);
      resized();
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
  template <typename T>
    void set_value(Matrix<T>& A, T value) {
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
   * Add the scalar value b to each element of matrix A.
   *
   */
  template <typename T>
    void add_scalar(Matrix<T>& A, T b) {
    apply(A, [=] (T a) {
        return a + b;
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
   *
   */
  template <typename T>
    void scale(Matrix<T>& A, const Matrix<T> &B, T scale_factor) {
    if (A.size() != B.size()) {
      A.resize(B.get_extents());
      resized();
    }
    map1(A, B, [=] (T b) {
        return b*scale_factor;
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
      resized();
    }
    map2(A, B, C, [] (T b, T c) {
        return b + c;
      });
  }


  /**
   * Copy the contents of matrix B into matrix A.
   *
   * This performs the same operation as the copy assignment operator (=).
   *
   * @param A The result is returned in this matrixx, which will be resized to the same dimensions as
   * matrix B if it does not already have the same number of elements.
   * @param B Input matrix which is not modified.
   */
  template <typename T>
    void copy_matrix(Matrix<T>& A, const Matrix<T>& B) {
    if (A.size() != B.size()) {
      A.resize(B.get_extents());
      resized();
    }
    // Copy contents of B into A.
#pragma omp parallel for
    for (int i = 0; i < A.size(); i++) {
      A[i] = B[i];
    }
  }

  /*
   * Check that the supplied matrices have dimensions that are compatible with the factorization:
   *
   * X approx= W * H.
   *
   * If they are not compatible, exit with an error.
   */
  void check_matrix_factorization_dimensions(const MatrixF& X, const MatrixF& W, const MatrixF& H);


  /*
   * Check that both matrices have the same dimensions. If they differ, exit with an error.
   */
  template <typename T1, typename T2>
    void check_dimensions(const Matrix<T1>& A, const Matrix<T2>& B) {
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
  bool check_dimensions_a_eq_b_tran_times_c(const MatrixF& A, const MatrixF& B, const MatrixF& C);

  /*
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
   * Copy the elements from a list of Matrix into a single flat Matrix.
   *
   * Since it can be inconvinient to determine to determine the total number of elements in the matrix
   * list before calling this function, it is not necessary to supply a "flat_mat" of the correct size.
   * The supplied "flat_mat" will be resized to match the total number of elements.
   *
   * @param mat_list The list of matrices that will be copied from.
   * @param flat_mat The matrix that will be copied into. This matrix will be resized to the same total
   * number of elements in the matrix list if necessary.
   */
  template <typename T>
    void copy_list_to_flat_matrix(const std::vector<Matrix<T>>& mat_list, Matrix<T>& flat_mat) {
    // Do an initial pass through all matrices to determine the total number of elements.
    int total_size = 0;
    for (size_t i = 0; i < mat_list.size(); i++) {
      const Matrix<T>& temp = mat_list[i];
      total_size += temp.size();
    }
    // If the element count is different the size of the current flat_mat, then reinitialize.
    if (total_size != flat_mat.size()) {
      // Create 1-dim matrix of size total_size.
      std::cout << "Resizing flat_mat to size = " << total_size << std::endl;
      flat_mat.resize(total_size);
    }
    int cur_pos = 0;
    for (size_t i = 0; i < mat_list.size(); i++) {
      const Matrix<T>& temp = mat_list[i];
      for (int backing_index = 0; backing_index < temp.size(); ++backing_index) {
        flat_mat[cur_pos + backing_index] = temp[backing_index];
      }
      cur_pos += temp.size();
    }
  }

  /**
   * Copy the elements from a flat Matrix into a list of matrices.
   *
   * The size of "flat_mat" must be equal to the total number of elements in the matrix list. Otherwise,
   * this function will exit with an error.
   *
   * @param mat_list The list of matrices that will be copied to.
   * @param flat_mat The matrix that will be copied from.
   */
  template <typename T>
    void copy_flat_matrix_to_list(std::vector<Matrix<T>>& mat_list, const Matrix<T>& flat_mat) {
    // Do an initial pass through all matrices to determine the total number of elements.
    int total_size = 0;
    for (size_t i = 0; i < mat_list.size(); i++) {
      Matrix<T>& temp = mat_list[i];
      total_size += temp.size();
    }
    // If the element count is different the size of the current flat_mat, then exit with error.
    if (total_size != flat_mat.size()) {
      error_exit("copy_flat_matrix_to_list(): Supplied matrix list has different element count than supplied flat matrix.");
    }
    int cur_pos = 0;
    for (size_t i = 0; i < mat_list.size(); i++) {
      MatrixF& temp = mat_list[i];
      for (int backing_index = 0; backing_index < temp.size(); ++backing_index) {
        temp[backing_index] = flat_mat[cur_pos + backing_index];
      }
      cur_pos += temp.size();
    }
  }


  ////////////////////////////////////////////////////////////////////////////////////////////
  // Network/layer utilities

  /*
   * Return the number of errors in the supplied network_output matrix, given the target labels
   * in target_labels.
   * For each column n of network_output, compute the row corresponding to the maximum value and
   * then check if target_labels(n) contains the same row index. If so, the network output at column
   * n is considered correct. Otherwise, it is considered an error.
   *
   * network_output: An M x N matrix. M is the number of class labels and N is the number of
   *                 output cases. Ideally, exactly one output class should be chosen and therefore
   *                 for each column, the correct row should have value = 1 and all other rows should
   *                 be equal to 0.
   *
   * target_labels: A 1-D matrix (array) of class labels of length N. For each element, the value is an integer in
   *              the range [0, N).
   */
  int error_count(const MatrixF& network_output, const Matrix<int> target_labels);

  /*
   * Compute the maxout of "input" and return the result in "output" and return the corresponding
   * indices of the maximum elements in "state".
   *
   * For each column of "input", the maximum value is taken of each consecutive K rows and the result
   * is written into each consecutive row of "output." Therefore, it is required that "input" have
   * size M x P and "output" have size N x P where K = M/N is an integer. That is, N divides evenly
   * into M. The size of "state" must be the same as "maxout_values_mat".
   *
   * Return the result in "output." If the matrices have inconsistent sizes, exit with an error.
   */
  void compute_forward_maxout(const MatrixF& input, MatrixF& output, Matrix<int>& state);

  /*
   * For each element in "output_backward", update the corresponding
   * value in "input_backward."
   *
   * In this version, all elements of "input_backward" are updated. The elements of "input_backward" that were chosen as a "max value" are
   * updated with the corresponding max value. All other elements of "input_backward" are set to 0.
   */
  void compute_reverse_maxout(MatrixF& input_backward, const MatrixF& output_backward, const Matrix<int>& state, bool accumulate=true);

  /*
   * Parameters:
   *
   * decay_val: The penalty for unused weights. Must be in the range [0, 1] where 0 imposes no penalty and 1 imposes the maximum penalty.
   */
  void compute_reverse_maxout_decay_unused(MatrixF& input_backward, const MatrixF& input, const MatrixF& output_backward,
                                           const Matrix<int>& state, float decay_val);


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
  void compute_reverse_kmax_decay_unused(MatrixF& input_backward, const MatrixF& inputs, const MatrixF& output_backward, const Matrix<int>& state,
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
   * Compute the forward-direction ReLU (Rectified Linear Unit) activation function on the input matrix "input".
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
   * @param output An N-dimensional matrix of the same size as in_vals.
   *
   * @param state An N-dimensional matrix of the same size as in_vals. This is used for storing
   *              state information that will be needed later by the reverse-direction relu function.
   *              This does not need to be initialized to
   *              any particular values since calling this function will overwrite its contents anyway.
   */
  void compute_forward_relu(const MatrixF& input, MatrixF& output, Matrix<int>& state);

  /*
   * Same as compute_forward_relu(), except this is the "leaky" version with hard-coded leakyness
   * parameter (see code for value).
   */
  void compute_forward_leaky_relu(const MatrixF& input, MatrixF& output, Matrix<int>& state);

  /**
   * Compute the forward-direction tanh activation function
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

  /*
   * Update the weights matrix W using the gradient W_grad and the learning rate alpha.
   *
   * W = W - alpha*W_grad
   *
   * is computed element-wise.
   *
   */
  void update_weights_from_gradient(MatrixF& W, const MatrixF& W_grad, float alpha);

  /*
   * Set weight/activatin decay to the specified value.
   *
   * The weight decay for each weight or activation w_i is set according to:
   *
   * w_i = w_i - decay_val*w_i
   */
  void update_weights_from_decay(MatrixF& W, float decay_val);

  /*
   * Update the weights matrix W using the gradient W_grad and the learning rate alpha and
   * weight decay lambda.
   *
   * W = W - alpha*W_grad -alpha*lambda*W
   *
   * is computed element-wise.
   *
   */
  void update_weights_from_gradient(MatrixF& W, const MatrixF& W_grad, float alpha, float lambda);

  /*
   * Update the weights matrix W using the gradient W_grad and the learning rate alpha.
   *
   * W = W - alpha*W_grad + ...
   *
   * is computed element-wise.
   *
   */
  void update_weights_from_gradient(MatrixF& W, const MatrixF& W_grad, float alpha, float lambda,
                                    float sparsity_param, bool force_nonnegative);


  void update_weights_from_gradient_rmsprop_v3(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_mean_square,
                                               float alpha, float rho);


  /*
   * Rmsprop with momentum.
   */
  void update_weights_from_gradient_rmsprop_momentum(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_mean_square,
                                                     MatrixF W_momentum, float alpha, float rho, float momentum);


  void update_weights_from_gradient_adagrad(MatrixF& W, const MatrixF& W_grad, MatrixF& W_grad_sum_square,
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



  /*
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
  void convolve_3d_filter_with_bias_minibatch_optimized(MatrixF& Z2, const MatrixF& W, const MatrixF& bias, const MatrixF& A1, MatrixF& temp_Z2, MatrixF& temp_A1, MatrixF& temp_W);





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
  void compute_convolutive_deltas_minibatch(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2);

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
  void compute_3d_convolutive_deltas_minibatch(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2);


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
  void compute_3d_convolutive_deltas_minibatch_optimized(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2,
                                                         MatrixF& temp_deltas_Z2, MatrixF& temp_deltas_A1, MatrixF& temp_W);

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
  void compute_convolutive_deltas_minibatch_optimized(MatrixF& deltas_A1, const MatrixF& W, const MatrixF& deltas_Z2,
                                                      MatrixF& temp_deltas_Z2, MatrixF& temp_deltas_A1, MatrixF& temp_W);



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


  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Convolution operations for factor models/deconvolutional networks
  //
  // Not included in the release version :)



  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Tensor operations and math functions


  /*
   * Use method from: http://cs231n.github.io/neural-networks-3/
   *
   * |A - B|
   * ---------
   * max(|A|, |B|)
   *
   * Note: do not use this method if we expect both matrices to be nearly 0.
   */
  float relative_error(const MatrixF& A, const MatrixF& B);

  /*
   * For each element x_i of X, set x_i = max(x_i, min_val).
   */
  void threshold_lower(MatrixF& X, float min_val);

  /*
   * For each element x_i of X, set x_i = min(x_i, min_val).
   */
  void threshold_upper(MatrixF& X, float max_val);


  /*
   * Print all elements in the vector to std out.
   */
  template <typename T>
    void print_vector(std::vector<T> vec) {
    for_each(begin(vec), end(vec), [] (T val) {
        std::cout << val << " ";
      });
    std::cout << std::endl;
  }



}

#endif  /* _UTILITIES_H */
