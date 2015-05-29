#ifndef _MATRIXT_H
#define	_MATRIXT_H
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

	/* An instance of this class represents a simple dense N-dimensional matrix of T in C-style
	 * order (as opposed to Fortran) where T must be a numeric type such as float. For a 2D matrix,
	 * this corresponds to row-major ordering.
	 * Currently, 1 to 6-dimensional matrices
	 * are supported and it is striaghtforward to add support for higher dimensional matrices. 
	 * This matrix is backed by a 1-D array of T. The matrix is initialized
	 * to all zeros.
	 *
	 * A typdef is made so that "Matrix" refers to a MatrixT<float> for convinience since this is the
	 * most common usage.
	 *
	 * To choose a specific number of dimensions N, create an instance of this class using the constructor
	 * corresponding to the desired number of dimensions, which will create a new N-dimensional matrix,
	 * initialized to all 0's.
	 *
	 * Note that once an N-dimensional matrix is created, it is still possible to use the functions that
	 * are intended for an M-dimensional matrix where N != M. Although these functions may still be called,
	 * they are typically not what is intended are will likely lead to undesired or undefined behavior.
	 * For simplicity and performance, no error checking is performed to prevent this. Be careful!
	 *
	 */
	template <typename T>
	class MatrixT{

	private:
		
		MatrixT() {}
		std::vector<T> m_values;
		// Number of dimensions.
		const int m_order; 
		// m_extents[i] is the size of the i'th dimension (0-based).
		std::vector<int> m_extents;

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

	public:

		/*
		 * Create a new 1D matrix of order 0 (0-dimensional).
		 * This matrix does not contain any data. It is only intended to be used
		 * to allow default initialization in a class, such as when it would be inconvinient
		 * to initialize the MatrixT in the constructor initialization list. For example, if
		 * it requires some computations to compute the sizes of the various dimensions, it may
		 * be more convinient to use the assignment operator to "initialize" the MatrixT in
		 * the constructor body.
		 */
		//MatrixT();

		/*
		 * Create a new matrix from the supplied extents.
		 *
		 * This will create an N-dim matrix where N = extents.size().
		 * The i'th element of extents specifies the i'th extent.
		 */
		MatrixT(const std::vector<int>& extents);

		/*
		 * Create a new 1D matrix with dimension e0, initialized to
		 * all zeros.
		 */
		MatrixT(int e0);

		/*
		 * Create a new 1D matrix with dimensions e0 x e1, initialized to
		 * all zeros.
		 */
		MatrixT(int e0, int e1);

		/*
		 * Create a new 3D matrix with dimensions e0 x e1 x e2, initialized to
		 * all zeros.
		 */
		MatrixT(int e0, int e1, int e2);

		/*
		 * Create a new 4D matrix with dimensions e0 x e1 x e2 x e3, initialized to
		 * all zeros.
		 */
		MatrixT(int e0, int e1, int e2, int e3);

		/*
		 * Create a new 5D matrix with dimensions e0 x e1 x e2 x e3 x e4, initialized to
		 * all zeros.
		 */
		MatrixT(int e0, int e1, int e2, int e3, int e4);

		/*
		 * Create a new 6D matrix with dimensions e0 x e1 x e2 x e3 x e4 x e5, initialized to
		 * all zeros.
		 */
		MatrixT(int e0, int e1, int e2, int e3, int e4, int e5);

		virtual ~MatrixT() {};

		// deprecated
		//int height; // Same as extent(0)
		// deprecated
		//int width; // Same as extent(1)

		/*
		 * Place <i>val</i> into the element at row = <i>row</i> and column = <i>column</i>.
		 *
		 * @param row The row index to set.
		 * @param column The column index to set.
		 * @param val The value to set.
		 */
		// deprecated
		void set(int row, int column, T val) {
			m_values[get_backing_index(row, column)] = val;
		}

		/*
		 * Add the supplied value to the existing value.
		 */
		// deprecated.
		void accumulate(int row, int column, T val) {
			m_values[get_backing_index(row, column)] += val;
		}

		/*
		 * Return a reference to the value at position "index" in the backing array
		 * for this matrix. This array has size equal to "order."
		 */
		T& operator[](int index) {
			return m_values[index];
		}

		/*
		 * Return a reference to the value at position "index" in the backing array
		 * for this matrix. This array has size equal to "order."
		 */
		const T& operator[](int index) const {
			return m_values[index];
		}

		/*
		 * Get the element at the specified row and column.
		 *
		 */
		// deprecated
		T get(int i0, int i1) const {
			return m_values[get_backing_index(i0, i1)]; // row-major
		}

		/*
		 * Get the element at the specified location using 1-dimensional indexing.
		 * Note: This is the equivalent to indexing directly into the backing array.
		 */
		T& operator()(int i0) {
			return m_values[i0];
		}

		/*
		 * Get the element at the specified location using 1-dimensional indexing.
		 * Note: This is the equivalent to indexing directly into the backing array.
		 */
		const T& operator()(int i0) const {
			return m_values[i0];
		}

		/*
		 * Get the element at the specified location using 2-dimensional indexing.
		 *
		 */
		T& operator()(int i0, int i1) {
			return m_values[get_backing_index(i0, i1)];
		}

		/*
		 * Get the element at the specified location using 2-dimensional indexing.
		 *
		 */
		const T& operator()(int i0, int i1) const {
			return m_values[get_backing_index(i0, i1)];
		}

		/*
		 * Get the element at the specified location using 3-dimensional indexing.
		 *
		 */
		T& operator()(int i0, int i1, int i2) {
			return m_values[get_backing_index(i0, i1, i2)];
		}

		/*
		 * Get the element at the specified location using 3-dimensional indexing.
		 *
		 */
		const T& operator()(int i0, int i1, int i2) const {
			return m_values[get_backing_index(i0, i1, i2)];
		}

		/*
		 * Get the element at the specified location using 4-dimensional indexing.
		 *
		 */
		T& operator()(int i0, int i1, int i2, int i3) {
			return m_values[get_backing_index(i0, i1, i2, i3)];
		}

		/*
		 * Get the element at the specified location using 4-dimensional indexing.
		 *
		 */
		const T& operator()(int i0, int i1, int i2, int i3) const {
			return m_values[get_backing_index(i0, i1, i2, i3)];
		}

		/*
		 * Get the element at the specified location using 5-dimensional indexing.
		 *
		 */
		T& operator()(int i0, int i1, int i2, int i3, int i4) {
			return m_values[get_backing_index(i0, i1, i2, i3, i4)];
		}

		/*
		 * Get the element at the specified location using 5-dimensional indexing.
		 *
		 */
		const T& operator()(int i0, int i1, int i2, int i3, int i4) const {
			return m_values[get_backing_index(i0, i1, i2, i3, i4)];
		}

		/*
		 * Get the element at the specified location using 6-dimensional indexing.
		 *
		 */
		T& operator()(int i0, int i1, int i2, int i3, int i4, int i5) {
			return m_values[get_backing_index(i0, i1, i2, i3, i4, i5)];
		}

		/*
		 * Get the element at the specified location using 6-dimensional indexing.
		 *
		 */
		const T& operator()(int i0, int i1, int i2, int i3, int i4, int i5) const {
			return m_values[get_backing_index(i0, i1, i2, i3, i4, i5)];
		}

		/*
		 * Return the index into the backing array using 2-dimensional indexing.
		 */
		int get_backing_index(int i0, int i1) const {
			return i0*m_extents[1] + i1; // row-major
		}

		/*
		 * Return the index into the backing array using 3-dimensional indexing.
		 */
		int get_backing_index(int i0, int i1, int i2) const {
			return i0*m_extents[1]*m_extents[2] + i1*m_extents[2] + i2;
		}

		/*
		 * Return the index into the backing array using 4-dimensional indexing.
		 */
		int get_backing_index(int i0, int i1, int i2, int i3) const {
			return i0*m_extents[1]*m_extents[2]*m_extents[3] + i1*m_extents[2]*m_extents[3] + 
				i2*m_extents[3] +i3;
		}

		/*
		 * Return the index into the backing array using 5-dimensional indexing.
		 */
		int get_backing_index(int i0, int i1, int i2, int i3, int i4) const {
			return i0*m_extents[1]*m_extents[2]*m_extents[3]*m_extents[4] + 
				i1*m_extents[2]*m_extents[3]*m_extents[4] + i2*m_extents[3]*m_extents[4] + 
				i3*m_extents[4] + i4;
		}

		/*
		 * Return the index into the backing array using 6-dimensional indexing.
		 */
		int get_backing_index(int i0, int i1, int i2, int i3, int i4, int i5) const {
			return i0*m_extents[1]*m_extents[2]*m_extents[3]*m_extents[4]*m_extents[5] + 
				i1*m_extents[2]*m_extents[3]*m_extents[4]*m_extents[5] + i2*m_extents[3]*m_extents[4]*m_extents[5] + 
				i3*m_extents[4]*m_extents[5] + i4*m_extents[5] + i5;
		}
		
		/*
		 * Set all values to be uniformly disstributed random values in [min, max).
		 */
		// deprecated: Use Utilities.h function instead.
		void randomize_uniform(T min, T max);


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
		  std::vector<int> get_extents() const {
			return m_extents;
		}
		 */

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

	};


	// Send contents of Matrix to ostream.
	template <typename T>
	std::ostream& operator<<(std::ostream& os, const MatrixT<T>& m);

	// Since in most cases, we just want a Matrix of floats, make a typedef.
	typedef MatrixT<float> Matrix;


	/////////////////////////////////////////////////////////////////////////////////////////
	// Implementation below: Needs to be in header file or else we get linker errors.

	// 0-dim matrix:
	//template <typename T>
	//	MatrixT<T>::MatrixT()
	//	: m_values(), m_order { 0 }, m_extents() {
	//}

	template <typename T>
	MatrixT<T>::MatrixT(const std::vector<int>& extents)
		: m_values(extents_to_size(extents)), m_order {static_cast<int>(extents.size())}, m_extents(extents) {
		
	}

	// 1-dim matrix:
	template <typename T>
		MatrixT<T>::MatrixT(int e0)
		: m_values(e0), m_order { 1 }, m_extents(1) {
		m_extents[0] = e0;
	}

	// 2-dim matrix:
	template <typename T>
		MatrixT<T>::MatrixT(int e0, int e1)
		: m_values(e0*e1), m_order { 2 }, m_extents(2) {
		m_extents[0] = e0;
		m_extents[1] = e1;
	}

	
	// 3-dim matrix:
	template <typename T>
		MatrixT<T>::MatrixT(int e0, int e1, int e2)
		: m_values(e0*e1*e2), m_order {3}, 
		m_extents(3) {
			m_extents[0] = e0;
			m_extents[1] = e1;
			m_extents[2] = e2;
		}
	
	// 4-dim matrix:
	template <typename T>
		MatrixT<T>::MatrixT(int e0, int e1, int e2, int e3)
		: m_values(e0*e1*e2*e3), m_order {4}, 
		m_extents(4) {
			m_extents[0] = e0;
			m_extents[1] = e1;
			m_extents[2] = e2;
			m_extents[3] = e3;

			
		}
		
	// 5-dim matrix:
	template <typename T>
		MatrixT<T>::MatrixT(int e0, int e1, int e2, int e3, int e4)
		: m_values(e0*e1*e2*e3*e4), m_order {5}, 
		m_extents(5) {
			m_extents[0] = e0;
			m_extents[1] = e1;
			m_extents[2] = e2;
			m_extents[3] = e3;
			m_extents[4] = e4;
	}

	// 6-dim matrix:
	template <typename T>
		MatrixT<T>::MatrixT(int e0, int e1, int e2, int e3, int e4, int e5)
		: m_values(e0*e1*e2*e3*e4*e5), m_order {6}, 
		m_extents(6) {
			m_extents[0] = e0;
			m_extents[1] = e1;
			m_extents[2] = e2;
			m_extents[3] = e3;
			m_extents[4] = e4;
			m_extents[5] = e5;
	}
	
	template <typename T>
	void MatrixT<T>::randomize_uniform(T min, T max) {
	  std::cerr << "Sorry, not implemented for non-floats yet." << std::endl;
		exit(1);
	}
	

	template <>
	inline void MatrixT<float>::randomize_uniform(float min, float max) {
		std::vector<float>& backingArrayA  = get_backing_vector();
		static std::mt19937 mersenne_twister_engine;
		mersenne_twister_engine.seed(static_cast<unsigned long>(time(NULL)));
		std::uniform_real_distribution<float> uni(min, max);
		for (size_t i = 0; i < backingArrayA.size(); i++) {
			backingArrayA[i] = uni(mersenne_twister_engine);
		}
	}
	

	/*
	template <typename T>
	T MatrixT<T>::maxValue() const {
		const T* backingArrayA = m_values.data();
		float max = backingArrayA[0];
		for (int i = 1; i < height * width; i++) {
			if (backingArrayA[i] > max) {
				max = backingArrayA[i];
			}
		}
		return max;
	}
	*/
	/*
	template <typename T>
	T MatrixT<T>::minValue() const {
		const T* backingArrayA = m_values.data();
		float min = backingArrayA[0];
		for (int i = 1; i < height * width; i++) {
			if (backingArrayA[i] < min) {
				min = backingArrayA[i];
			}
		}
		return min;
	}
	*/

	/*
	template <typename T>
	MatrixT<T>::operator std::vector<T>() const {
		if (m_order != 2) {
			std::cerr << "Matrix is not 2D. Exiting." << std::endl;
			std::cerr << "order = " << m_order << std::endl;
			exit(1);
		}
		if (extent(0) == 1) {
			// Matrix has 1 row. OK.
			std::vector<T> out(extent(1));
			for (int i = 0; i < extent(1); ++i) {
				out.at(i) = get(0, i);
			}
			return out;
		}
		else if (extent(1) == 1) {
			// Matrix has 1 column. OK.
		  std::vector<T> out(extent(0));
			for (int i = 0; i < extent(0); ++i) {
				out.at(i) = get(i, 0);
			}
			return out;
		}
		else {
		  std::cerr << "Matrix  of size " << extent(0) << " rows x " << extent(1) << " columns cannot be converted to an array! Exiting." << std::endl;
			exit(1);
		}
	}
	*/


	template <typename T>
	MatrixT<T>::operator std::vector<T>() const {
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
	std::ostream& operator<<(std::ostream& os, const MatrixT<T>& m) {
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

#endif	/* _MATRIXT_H */

