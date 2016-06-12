#ifndef _MATRIX_H
#define _MATRIX_H
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
#include <string>
#include <vector>

#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include <ctime>
namespace kumozu {

  /* An instance of this class represents a dense N-dimensional matrix of type T, sometimes also
   * called a "tensor" or "multi-dimensional array."
   *
   * This matrix is backed by a 1-D array of templated type T that uses C-style
   * order (as opposed to Fortran) where T must be a numeric type such as float. For a 2D matrix,
   * this corresponds to row-major ordering.
   *
   * Currently, 1 to 6-dimensional matrices
   * are supported and it is striaghtforward to add support for higher dimensional matrices.
   * The matrix is initialized
   * to all zeros.
   *
   * Views are not currently supported. That is, it is not currently possible for two distinct Matrix
   * objects to share the same underlying backing array.
   *
   * The following convinience typedefs for common numerical types are available:
   *
   * float type:  MatrixF is same as Matrix<float>
   * double type: MatrixD is same as Matrix<double>
   * int type:    MatrixI is same as Matrix<int>
   *
   * To choose a specific number of dimensions N, create an instance of this class using the constructor
   * corresponding to the desired number of dimensions, which will create a new N-dimensional matrix,
   * initialized to all 0's.
   *
   * It is also possible to use the default constructor to get a new Matrix instance of size 0. This matrix (as
   * well as an existing nonzero size matrix) can later be given a different size and/or number of dimensions by
   * calling the resize() member function.
   *
   * It is important to note that once an N-dimensional matrix is created, it is still possible to call the (incorrect) functions that
   * are intended for an M-dimensional matrix where N != M. Although these functions may still be called without rasing an error,
   * they are typically not what is intended and will likely lead to undesired and/or undefined behavior.
   * For simplicity and performance, no error checking is performed to prevent such incorrect usage by default. However, such
   * checks are performed when debug-mode is enabled. While debugging, it is recommended to enable
   * bounds-checking and checking that the access function
   * called matches the actual dimension. Once the code runs correctly and does not
   * throw any out-of-bounds errors, the code can then be rebuilt without bounds-checking for increased performance. To enable
   * the debugging bounds-checking and dimension-checking, define KUMOZU_DEBUG as a compiler flag.
   *
   * If it is actually desired to interpret a N-dim matrix as a N-dim matrix (where M != N), then the resize() function should
   * be used to change the number of dimensions. The underlying storage will only be reallocated if the new size is different than
   * the old size.
   */
  template <typename T>
    class Matrix{

  private:


    std::vector<T> m_values;
    // Number of dimensions.
    int m_order;
    // m_extents[i] is the size of the i'th dimension (0-based).
    std::vector<int> m_extents; // size is m_order

    // m_strides[i] is the number of elements in the underlying 1d array that are needed to go from
    // element to element in dimension i. The last element of m_strides will always have value 1 since
    // elements in the last (that is, highest) dimension are always contiguous in memory.
    std::vector<int> m_strides; // size is m_order

    // If in debug mode, turn on bounds checking.
#ifdef KUMOZU_DEBUG
    static constexpr bool m_bounds_check = true;
#else
    static constexpr bool m_bounds_check = false;
#endif

    // Used by constructor that takes extents as parameter.
    static int extents_to_size(const std::vector<int>& extents) {
      int elem_count = 1;
      for (size_t i = 0; i < extents.size(); ++i) {
        elem_count *= extents[i];
      }
      if (elem_count == 0) {
        std::cerr << "extents_to_size(): 0-valued extents are not allowed. Exiting." << std::endl;
        exit(1);
      }
      return elem_count;
    }

    // Given the vector of extents, make the vector of strides.
    static void extents_to_strides(std::vector<int>& strides, const std::vector<int>& extents) {
      for (int n = static_cast<int>(extents.size())-1; n >=0; --n) {
        if (n == static_cast<int>(extents.size())-1) {
          strides[n] = 1;
        } else {
          strides[n] = strides[n+1]*extents[n+1];
        }
      }
    }

  public:

    /*
     * Create a new 1D matrix of order 0 (0-dimensional).
     * This matrix does not contain any data. If this constructor is used, the
     * resize() function should be called to give the matrix a nonzero size.
     */
    Matrix();

    /*
     * Create a new matrix from the supplied extents. Note that extents = dimensions.
     *
     * This will create an N-dim matrix where N = extents.size().
     * The i'th element of extents specifies the i'th extent. Note that
     * the i'th extent is the size of the i'th dimension.
     */
    Matrix(const std::vector<int>& extents);

    /*
     * Create a new 1D matrix with dimension e0, initialized to
     * all zeros.
     */
    Matrix(int e0);

    /*
     * Create a new 1D matrix with dimensions e0 x e1, initialized to
     * all zeros.
     */
    Matrix(int e0, int e1);

    /*
     * Create a new 3D matrix with dimensions e0 x e1 x e2, initialized to
     * all zeros.
     */
    Matrix(int e0, int e1, int e2);

    /*
     * Create a new 4D matrix with dimensions e0 x e1 x e2 x e3, initialized to
     * all zeros.
     */
    Matrix(int e0, int e1, int e2, int e3);

    /*
     * Create a new 5D matrix with dimensions e0 x e1 x e2 x e3 x e4, initialized to
     * all zeros.
     */
    Matrix(int e0, int e1, int e2, int e3, int e4);

    /*
     * Create a new 6D matrix with dimensions e0 x e1 x e2 x e3 x e4 x e5, initialized to
     * all zeros.
     */
    Matrix(int e0, int e1, int e2, int e3, int e4, int e5);

    virtual ~Matrix() {};


    // Resizing functions. These are used to change the size and/or number of dimensions of a Matrix.

    /*
     * Resize the matrix to the supplied extents. Note that extents = dimensions.
     *
     * This will resize to an N-dim matrix where N = extents.size().
     * The i'th element of extents specifies the i'th extent. Note that
     * the i'th extent is the size of the i'th dimension.
     *
     * If resized to the same size or smaller, the remaining element values are not changed, but
     * any newly added values are initialized to 0.
     */
    void resize(const std::vector<int>& extents) {
      m_values.resize(extents_to_size(extents));
      m_order = static_cast<int>(extents.size());
      m_extents = extents;
      m_strides.resize(m_order);
      extents_to_strides(m_strides, m_extents);
    }

    /*
     * Resize to a 1D matrix with dimension e0.
     *
     * If resized to the same size or smaller, the remaining element values are not changed, but
     * any newly added values are initialized to 0.
     */
    void resize(int e0) {
      m_values.resize(e0);
      m_order = 1;
      m_extents.resize(1);
      m_strides.resize(1);
      m_extents[0] = e0;
      extents_to_strides(m_strides, m_extents);
    }

    /*
     * Resize to a 1D matrix with dimension (e0 x e1).
     *
     * If resized to the same size or smaller, the remaining element values are not changed, but
     * any newly added values are initialized to 0.
     */
    void resize(int e0, int e1) {
      m_values.resize(e0*e1);
      m_order = 2;
      m_extents.resize(2);
      m_strides.resize(2);
      m_extents[0] = e0;
      m_extents[1] = e1;
      extents_to_strides(m_strides, m_extents);
    }

    /*
     * Resize to a 1D matrix with dimension (e0 x e1 x e2).
     *
     * If resized to the same size or smaller, the remaining element values are not changed, but
     * any newly added values are initialized to 0.
     */
    void resize(int e0, int e1, int e2) {
      m_values.resize(e0*e1*e2);
      m_order = 3;
      m_extents.resize(3);
      m_strides.resize(3);
      m_extents[0] = e0;
      m_extents[1] = e1;
      m_extents[2] = e2;
      extents_to_strides(m_strides, m_extents);
    }

    /*
     * Resize to a 1D matrix with dimension (e0 x e1 x e2 x e3).
     *
     * If resized to the same size or smaller, the remaining element values are not changed, but
     * any newly added values are initialized to 0.
     */
    void resize(int e0, int e1, int e2, int e3) {
      m_values.resize(e0*e1*e2*e3);
      m_order = 4;
      m_extents.resize(4);
      m_strides.resize(4);
      m_extents[0] = e0;
      m_extents[1] = e1;
      m_extents[2] = e2;
      m_extents[3] = e3;
      extents_to_strides(m_strides, m_extents);
    }

    /*
     * Resize to a 1D matrix with dimension (e0 x e1 x e2 x e3 x e4).
     *
     * If resized to the same size or smaller, the remaining element values are not changed, but
     * any newly added values are initialized to 0.
     */
    void resize(int e0, int e1, int e2, int e3, int e4) {
      m_values.resize(e0*e1*e2*e3*e4);
      m_order = 5;
      m_extents.resize(5);
      m_strides.resize(5);
      m_extents[0] = e0;
      m_extents[1] = e1;
      m_extents[2] = e2;
      m_extents[3] = e3;
      m_extents[4] = e4;
      extents_to_strides(m_strides, m_extents);
    }

    /*
     * Resize to a 1D matrix with dimension (e0 x e1 x e2 x e3 x e4 x e5).
     *
     * If resized to the same size or smaller, the remaining element values are not changed, but
     * any newly added values are initialized to 0.
     */
    void resize(int e0, int e1, int e2, int e3, int e4, int e5) {
      m_values.resize(e0*e1*e2*e3*e4*e5);
      m_order = 6;
      m_extents.resize(6);
      m_strides.resize(6);
      m_extents[0] = e0;
      m_extents[1] = e1;
      m_extents[2] = e2;
      m_extents[3] = e3;
      m_extents[4] = e4;
      m_extents[5] = e5;
      extents_to_strides(m_strides, m_extents);
    }

    /*
     * Return a reference to the value at position "index" in the backing array
     * for this matrix. This array has size equal to "size()."
     */
    T& operator[](int index) {
      if (m_bounds_check) {
        if ((index < 0) || (index >= size())) {
          std::cerr << "operator[](int index)" << std::endl;
          std::cerr << "1-dim index out of bounds: " << index << std::endl;
          std::cerr << "extent(0) = " << m_extents[0] << std::endl;
          exit(1);
        }
      }
      return m_values[index];
    }

    /*
     * Return a reference to the value at position "index" in the backing array
     * for this matrix. This array has size equal to "size()."
     */
    const T& operator[](int index) const {
      if (m_bounds_check) {
        if ((index < 0) || (index >= size())) {
          std::cerr << "operator[](int index)" << std::endl;
          std::cerr << "1-dim index out of bounds: " << index << std::endl;
          std::cerr << "extent(0) = " << m_extents[0] << std::endl;
          exit(1);
        }
      }
      return m_values[index];
    }

    /*
     * Get the element at the specified location using 1-dimensional indexing.
     * Note: This is the equivalent to indexing directly into the backing array.
     */
    T& operator()(int i0) {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 1) {
          std::cerr << "1-dim indexing called but order is not 1." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "operator()(int i0)" << std::endl;
          std::cerr << "1-dim index out of bounds: " << i0 << std::endl;
          std::cerr << "extent(0) = " << m_extents[0] << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0)" << std::endl;
          exit(1);
        }
      }
      return m_values[i0];
    }

    /*
     * Get the element at the specified location using 1-dimensional indexing.
     * Note: This is the equivalent to indexing directly into the backing array.
     */
    const T& operator()(int i0) const {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 1) {
          std::cerr << "1-dim indexing called but order is not 1." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "operator()(int i0)" << std::endl;
          std::cerr << "1-dim index out of bounds: " << i0 << std::endl;
          std::cerr << "extent(0) = " << m_extents[0] << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0) const" << std::endl;
          exit(1);
        }
      }
      return m_values[i0];
    }

    /*
     * Get the element at the specified location using 2-dimensional indexing.
     *
     */
    T& operator()(int i0, int i1) {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 2) {
          std::cerr << "2-dim indexing called but order is not 2." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "2-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "2-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1)" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1];
    }

    /*
     * Get the element at the specified location using 2-dimensional indexing.
     *
     */
    const T& operator()(int i0, int i1) const {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 2) {
          std::cerr << "2-dim indexing called but order is not 2." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "2-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "2-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1) const" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1];
    }

    /*
     * Get the element at the specified location using 3-dimensional indexing.
     *
     */
    T& operator()(int i0, int i1, int i2) {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 3) {
          std::cerr << "3-dim indexing called but order is not 3." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "3-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "3-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "3-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2)" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] + i2];
    }

    /*
     * Get the element at the specified location using 3-dimensional indexing.
     *
     */
    const T& operator()(int i0, int i1, int i2) const {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 3) {
          std::cerr << "3-dim indexing called but order is not 3." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "3-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "3-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "3-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2) const" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] + i2];
    }

    /*
     * Get the element at the specified location using 4-dimensional indexing.
     *
     */
    T& operator()(int i0, int i1, int i2, int i3) {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 4) {
          std::cerr << "4-dim indexing called but order is not 4." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "4-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "4-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "4-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if ((i3 < 0) || (i3 >= m_extents[3])) {
          std::cerr << "4-dim index i3 out of bounds: " << i3 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2, int i3)" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] +  i2*m_strides[2] +i3];
    }

    /*
     * Get the element at the specified location using 4-dimensional indexing.
     *
     */
    const T& operator()(int i0, int i1, int i2, int i3) const {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 4) {
          std::cerr << "4-dim indexing called but order is not 4." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "4-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "4-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "4-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if ((i3 < 0) || (i3 >= m_extents[3])) {
          std::cerr << "4-dim index i3 out of bounds: " << i3 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2, int i3) const" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] +  i2*m_strides[2] +i3];
    }

    /*
     * Get the element at the specified location using 5-dimensional indexing.
     *
     */
    T& operator()(int i0, int i1, int i2, int i3, int i4) {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 5) {
          std::cerr << "5-dim indexing called but order is not 5." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "5-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "5-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "5-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if ((i3 < 0) || (i3 >= m_extents[3])) {
          std::cerr << "5-dim index i3 out of bounds: " << i3 << std::endl;
          good = false;
        }
        if ((i4 < 0) || (i4 >= m_extents[4])) {
          std::cerr << "5-dim index i4 out of bounds: " << i4 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2, int i3, int i4)" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] + i2*m_strides[2] + i3*m_strides[3] + i4];
    }

    /*
     * Get the element at the specified location using 5-dimensional indexing.
     *
     */
    const T& operator()(int i0, int i1, int i2, int i3, int i4) const {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 5) {
          std::cerr << "5-dim indexing called but order is not 5." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "5-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "5-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "5-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if ((i3 < 0) || (i3 >= m_extents[3])) {
          std::cerr << "5-dim index i3 out of bounds: " << i3 << std::endl;
          good = false;
        }
        if ((i4 < 0) || (i4 >= m_extents[4])) {
          std::cerr << "5-dim index i4 out of bounds: " << i4 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2, int i3, int i4) const" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] + i2*m_strides[2] + i3*m_strides[3] + i4];
    }

    /*
     * Get the element at the specified location using 6-dimensional indexing.
     *
     */
    T& operator()(int i0, int i1, int i2, int i3, int i4, int i5) {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 6) {
          std::cerr << "6-dim indexing called but order is not 6." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "6-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "6-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "6-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if ((i3 < 0) || (i3 >= m_extents[3])) {
          std::cerr << "6-dim index i3 out of bounds: " << i3 << std::endl;
          good = false;
        }
        if ((i4 < 0) || (i4 >= m_extents[4])) {
          std::cerr << "6-dim index i4 out of bounds: " << i4 << std::endl;
          good = false;
        }
        if ((i5 < 0) || (i5 >= m_extents[5])) {
          std::cerr << "6-dim index i5 out of bounds: " << i5 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2, int i3, int i4, int i5)" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] + i2*m_strides[2] + i3*m_strides[3] + i4*m_strides[4] + i5];
    }

    /*
     * Get the element at the specified location using 6-dimensional indexing.
     *
     */
    const T& operator()(int i0, int i1, int i2, int i3, int i4, int i5) const {
      if (m_bounds_check) {
        bool good = true;
        if (m_order != 6) {
          std::cerr << "6-dim indexing called but order is not 6." << std::endl;
          good = false;
        }
        if ((i0 < 0) || (i0 >= m_extents[0])) {
          std::cerr << "6-dim index i0 out of bounds: " << i0 << std::endl;
          good = false;
        }
        if ((i1 < 0) || (i1 >= m_extents[1])) {
          std::cerr << "6-dim index i1 out of bounds: " << i1 << std::endl;
          good = false;
        }
        if ((i2 < 0) || (i2 >= m_extents[2])) {
          std::cerr << "6-dim index i2 out of bounds: " << i2 << std::endl;
          good = false;
        }
        if ((i3 < 0) || (i3 >= m_extents[3])) {
          std::cerr << "6-dim index i3 out of bounds: " << i3 << std::endl;
          good = false;
        }
        if ((i4 < 0) || (i4 >= m_extents[4])) {
          std::cerr << "6-dim index i4 out of bounds: " << i4 << std::endl;
          good = false;
        }
        if ((i5 < 0) || (i5 >= m_extents[5])) {
          std::cerr << "6-dim index i5 out of bounds: " << i5 << std::endl;
          good = false;
        }
        if (!good) {
          std::cerr << "operator()(int i0, int i1, int i2, int i3, int i4, int i5) const" << std::endl;
          exit(1);
        }
      }
      return m_values[i0*m_strides[0] + i1*m_strides[1] + i2*m_strides[2] + i3*m_strides[3] + i4*m_strides[4] + i5];
    }

    /*
     * Return the total number of elements in the Matrix. This value is the product of the dimension sizes that
     * were supplied to the constructor.
     */
    int size() const {
      return static_cast<int>(m_values.size());
    }

    /*
     * Return the size of the i'th dimension.
     */
    int extent(int i) const {
      return m_extents[i];
    }

    /*
     * Return a vector of extents for this matrix.
     *
     * The i'th component of the returned array contains the size
     * of the i'th dimension.
     *
     */
    const std::vector<int>& get_extents() const {
      return m_extents;
    }


    /*
     * Return the number of dimensions in this matrix.
     */
    int order() const {
      return m_order;
    }

    /*
     * Get pointer to underlying backing array. Be careful!
     */
    T* get_backing_data() {
      return m_values.data();
    }

    /*
     * Get pointer to underlying backing array. Be careful!
     */
    const T* get_backing_data() const {
      return m_values.data();
    }

    /*
     * Get vector of underlying backing array.
     */
    std::vector<T>& get_backing_vector() {
      return m_values;
    }

    /*
     * Get vector of underlying backing array.
     */
    const std::vector<T>& get_backing_vector() const {
      return m_values;
    }

    /*
     * Convert a Matrix to a vector<T>. In order for this to make sense,
     * the Matrix must conceptually
     * correspond to a 1-dimensional array. That is, it must satisfy
     * one of the following two conditions:
     *
     * The size of the Maatrix is N x 1 where N >= 1.
     *
     * The size of the Matrix is 1 x M where M >= 1.
     *
     * If neither of these conditions is satisfied, the program will exit with
     * an error. This function is only supported on 2-dimensional matrices.
     */
    operator std::vector<T>() const;

    /*
     * Return true if the supplied 1-dimensional index is in a valid range. Otherwise return false.
     *
     * This function does not check that matrix is actually 1-dimensional since that check is already
     * performed if KUMOZU_DEBUG is defined.
     */
    bool is_valid_range(int i0) const {
      if ((i0 < 0) || (i0 >= m_extents[0])) {
        return false;
      } else {
        return true;
      }
    }

    /*
     * Return true if the supplied 2-dimensional index is in a valid range. Otherwise return false.
     *
     * This function does not check that matrix is actually 2-dimensional since that check is already
     * performed if KUMOZU_DEBUG is defined.
     */
    bool is_valid_range(int i0, int i1) const {
      if ((i0 < 0) || (i0 >= m_extents[0])) {
        return false;
      }
      if ((i1 < 0) || (i1 >= m_extents[1])) {
        return false;
      }
      return true;
    }

    /*
     * Return true if the supplied 3-dimensional index is in a valid range. Otherwise return false.
     *
     * This function does not check that matrix is actually 3-dimensional since that check is already
     * performed if KUMOZU_DEBUG is defined.
     */
    bool is_valid_range(int i0, int i1, int i2) const {
      if ((i0 < 0) || (i0 >= m_extents[0])) {
        return false;
      }
      if ((i1 < 0) || (i1 >= m_extents[1])) {
        return false;
      }
      if ((i2 < 0) || (i2 >= m_extents[2])) {
        return false;
      }
      return true;
    }

    /*
     * Return true if the supplied 4-dimensional index is in a valid range. Otherwise return false.
     *
     * This function does not check that matrix is actually 4-dimensional since that check is already
     * performed if KUMOZU_DEBUG is defined.
     */
    bool is_valid_range(int i0, int i1, int i2, int i3) const {
      if ((i0 < 0) || (i0 >= m_extents[0])) {
        return false;
      }
      if ((i1 < 0) || (i1 >= m_extents[1])) {
        return false;
      }
      if ((i2 < 0) || (i2 >= m_extents[2])) {
        return false;
      }
      if ((i3 < 0) || (i3 >= m_extents[3])) {
        return false;
      }
      return true;
    }

    /*
     * Return true if the supplied 5-dimensional index is in a valid range. Otherwise return false.
     *
     * This function does not check that matrix is actually 5-dimensional since that check is already
     * performed if KUMOZU_DEBUG is defined.
     */
    bool is_valid_range(int i0, int i1, int i2, int i3, int i4) const {
      if ((i1 < 0) || (i1 >= m_extents[1])) {
        return false;
      }
      if ((i2 < 0) || (i2 >= m_extents[2])) {
        return false;
      }
      if ((i3 < 0) || (i3 >= m_extents[3])) {
        return false;
      }
      if ((i4 < 0) || (i4 >= m_extents[4])) {
        return false;
      }
      return true;
    }

    /*
     * Return true if the supplied 6-dimensional index is in a valid range. Otherwise return false.
     *
     * This function does not check that matrix is actually 6-dimensional since that check is already
     * performed if KUMOZU_DEBUG is defined.
     */
    bool is_valid_range(int i0, int i1, int i2, int i3, int i4, int i5) const {
      if ((i0 < 0) || (i0 >= m_extents[0])) {
        return false;
      }
      if ((i1 < 0) || (i1 >= m_extents[1])) {
        return false;
      }
      if ((i2 < 0) || (i2 >= m_extents[2])) {
        return false;
      }
      if ((i3 < 0) || (i3 >= m_extents[3])) {
        return false;
      }
      if ((i4 < 0) || (i4 >= m_extents[4])) {
        return false;
      }
      if ((i5 < 0) || (i5 >= m_extents[5])) {
        return false;
      }
      return true;
    }

  };


  // Send contents of Matrix to ostream.
  template <typename T>
    std::ostream& operator<<(std::ostream& os, const Matrix<T>& m);

  // Typedefs for common types.
  //typedef Matrix<float> MatrixF;
  //typedef Matrix<double> MatrixD;
  //typedef Matrix<int> MatrixI;

  using MatrixF = Matrix<float>;
  using MatrixD = Matrix<double>;
  using MatrixI = Matrix<int>;

  /////////////////////////////////////////////////////////////////////////////////////////
  // Implementation below: Needs to be in header file or else we get linker errors.

  // 0-dim matrix:
  // Don't try to put any data in this :)
  template <typename T>
    Matrix<T>::Matrix()
    : m_values(), m_order { 0 }, m_extents(), m_strides() {
  }

  template <typename T>
    Matrix<T>::Matrix(const std::vector<int>& extents)
    : m_values(extents_to_size(extents)),
    m_order {static_cast<int>(extents.size())},
    m_extents(extents),
      m_strides(m_order) {
        extents_to_strides(m_strides, m_extents);
      }

    // 1-dim matrix:
    template <typename T>
      Matrix<T>::Matrix(int e0)
      : m_values(e0), m_order { 1 }, m_extents(1), m_strides(1) {
      m_extents[0] = e0;
      extents_to_strides(m_strides, m_extents);
    }

    // 2-dim matrix:
    template <typename T>
      Matrix<T>::Matrix(int e0, int e1)
      : m_values(e0*e1), m_order { 2 }, m_extents(2), m_strides(2) {
      m_extents[0] = e0;
      m_extents[1] = e1;
      extents_to_strides(m_strides, m_extents);
    }


    // 3-dim matrix:
    template <typename T>
      Matrix<T>::Matrix(int e0, int e1, int e2)
      : m_values(e0*e1*e2), m_order {3},
      m_extents(3), m_strides(3) {
        m_extents[0] = e0;
        m_extents[1] = e1;
        m_extents[2] = e2;
        extents_to_strides(m_strides, m_extents);
      }

      // 4-dim matrix:
      template <typename T>
        Matrix<T>::Matrix(int e0, int e1, int e2, int e3)
        : m_values(e0*e1*e2*e3), m_order {4},
        m_extents(4), m_strides(4) {
          m_extents[0] = e0;
          m_extents[1] = e1;
          m_extents[2] = e2;
          m_extents[3] = e3;
          extents_to_strides(m_strides, m_extents);

        }

        // 5-dim matrix:
        template <typename T>
          Matrix<T>::Matrix(int e0, int e1, int e2, int e3, int e4)
          : m_values(e0*e1*e2*e3*e4), m_order {5},
          m_extents(5), m_strides(5) {
            m_extents[0] = e0;
            m_extents[1] = e1;
            m_extents[2] = e2;
            m_extents[3] = e3;
            m_extents[4] = e4;
            extents_to_strides(m_strides, m_extents);
          }

          // 6-dim matrix:
          template <typename T>
            Matrix<T>::Matrix(int e0, int e1, int e2, int e3, int e4, int e5)
            : m_values(e0*e1*e2*e3*e4*e5), m_order {6},
            m_extents(6), m_strides(6) {
              m_extents[0] = e0;
              m_extents[1] = e1;
              m_extents[2] = e2;
              m_extents[3] = e3;
              m_extents[4] = e4;
              m_extents[5] = e5;
              extents_to_strides(m_strides, m_extents);
            }

            template <typename T>
              Matrix<T>::operator std::vector<T>() const {
              if (m_order != 1) {
                std::cerr << "Matrix is not 1D. Exiting." << std::endl;
                std::cerr << "order = " << m_order << std::endl;
                exit(1);
              }
              std::vector<T> out(extent(0));
              for (int i = 0; i < extent(0); ++i) {
                //out.at(i) = get(i);
                out.at(i) = m_values[i];
              }
              return out;
            }


            template <typename T>
              std::ostream& operator<<(std::ostream& os, const Matrix<T>& m) {
              if (m.order() == 1) {
                for (int i = 0; i < m.extent(0); i++) {
                  os << m(i) << "  ";
                }
              } else if (m.order() == 2) {
                for (int i = 0; i < m.extent(0); i++) {
                  for (int j = 0; j < m.extent(1); j++) {
                    os << m(i, j) << "  ";
                  }
                  os << std::endl;
                }
              } else if (m.order() == 3) {
                for (int i = 0; i < m.extent(0); ++i) {
                  for (int j = 0; j < m.extent(1); ++j) {
                    for (int k = 0; k < m.extent(2); ++k) {
                      os << m(i, j, k) << "  ";
                    }
                    os << std::endl;
                  }
                  os << std::endl;
                }
              } else if (m.order() == 4) {
                for (int i = 0; i < m.extent(0); ++i) {
                  for (int j = 0; j < m.extent(1); ++j) {
                    for (int k = 0; k < m.extent(2); ++k) {
                      for (int l = 0; l < m.extent(3); ++l) {
                        os << m(i, j, k, l) << "  ";
                      }
                      os << std::endl;
                    }
                    os << std::endl;
                  }
                  os << std::endl;
                }
              } else if (m.order() == 5) {
                for (int i = 0; i < m.extent(0); ++i) {
                  for (int j = 0; j < m.extent(1); ++j) {
                    for (int k = 0; k < m.extent(2); ++k) {
                      for (int l = 0; l < m.extent(3); ++l) {
                        for (int n = 0; n < m.extent(4); ++n) {
                          os << m(i, j, k, l, n) << "  ";
                        }
                        os << std::endl;
                      }
                      os << std::endl;
                    }
                    os << std::endl;
                  }
                  os << std::endl;
                }
              } else if (m.order() == 6) {
                for (int i = 0; i < m.extent(0); ++i) {
                  for (int j = 0; j < m.extent(1); ++j) {
                    for (int k = 0; k < m.extent(2); ++k) {
                      for (int l = 0; l < m.extent(3); ++l) {
                        for (int n = 0; n < m.extent(4); ++n) {
                          for (int p = 0; p < m.extent(5); ++p) {
                            os << m(i, j, k, l, n, p) << "  ";
                          }
                          os << std::endl;
                        }
                        os << std::endl;
                      }
                      os << std::endl;
                    }
                    os << std::endl;
                  }
                  os << std::endl;
                }
              } else {
                os << "Not supported." << std::endl;
              }
              return os;
            }


}

#endif  /* _MATRIX_H */
