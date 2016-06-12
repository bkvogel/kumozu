#ifndef _MATRIX_REF_VECTOR_H
#define _MATRIX_REF_VECTOR_H
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
#include <string>
#include <iostream>
#include <functional>
#include <map>
#include "Constants.h"
#include "Utilities.h"
#include <vector>

namespace kumozu {

  /**
   * Represents a vector of Matrix references.
   *
   * This class is a container that holds a list of Matrix references. This class intends to
   * be useful when it is desired to have a single container that refers to a set of related
   * matrices (which in general may have different sizes and dimensions). An example use case
   * would be a "get_paramaters()" function for a neural network the returns in instance of
   * this class that contains each parameter matrix in the network. This instance could
   * be passed into an "Updater" instance that will then update the corresponding matrices using
   * an optimization algorithm.
   * 
   * Note also that the API provided by this clsss simply wraps a small subset of the 
   * API for std::vector, although the function signatures may be slightly different.
   */
  template <typename T>
  class MatrixRefVector {

  public:

    /**
     * Create an empty list.
     */
    MatrixRefVector() {

    }

    Matrix<T>& at(int index) {
      if (m_mode == 1) {
	error_exit("at(): Error: Cannot get non-const Matrix reference from a const reference.");
      }
      return m_mat_list.at(index).get();
    }

    const Matrix<T>& at(int index) const {
      if (m_mode == 0) {
	return m_mat_list.at(index).get();
      } else {
	return m_const_mat_list.at(index).get();
      }
    }

    const int size() const {
      if (m_mode == 0) {
	return static_cast<int>(m_mat_list.size());
      } else {
	return static_cast<int>(m_const_mat_list.size());
      }
    }

    void push_back(Matrix<T>& mat) {
      if (m_mode == -1) {
	m_mode = 0;
      }
      if (m_mode != 0) {
	error_exit("push_back(): Error: Attempt to push non-const reference but the const list is not empty.");
      }
      m_mat_list.push_back(std::ref(mat));
    }

    void push_back(const Matrix<T>& mat) {
      if (m_mode == -1) {
	m_mode = 1;
      }
      if (m_mode != 1) {
	error_exit("push_back(): Error: Attempt to push const reference but the non-const list is not empty.");
      }
      m_const_mat_list.push_back(std::cref(mat));
    }

    void clear() {
      m_mat_list.clear();
      m_const_mat_list.clear();
    }

  private:

    std::vector<std::reference_wrapper<Matrix<T>>> m_mat_list;
    std::vector<std::reference_wrapper<const Matrix<T>>> m_const_mat_list;
    int m_mode {-1};

  };

  using MatrixRefVectorF = MatrixRefVector<float>;

  /**
   * Copy the matrix list from in_list into out_list. 
   *
   * Both supplied lists must have the same size. However, it is not necessary for the
   * individual matrices in out_list to have the same extents as the corresponing matrices
   * in in_list. If the matrix size differs, the matrix in out_list will be resized to match the
   * corresponding matrix in in_list.
   */
  template <typename T>
    void copy(MatrixRefVector<T>& out_list, const MatrixRefVector<T>& in_list) {
    if (out_list.size() != in_list.size()) {
      error_exit("copy(): Error: out_list does not have same size as in_list.");
    }
    for (int i = 0; i < in_list.size(); ++i) {
      const auto& in_mat = in_list.at(i);
      auto& out_mat = out_list.at(i);
      copy_matrix(out_mat, in_mat);
    }
  }
  
}

#endif  /* _UTILITIES_H */
