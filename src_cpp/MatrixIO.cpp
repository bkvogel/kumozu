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
#include "MatrixIO.h"
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;

// todo: replace C-style file IO code with C++ code.

namespace kumozu {

	void save_matrix(const Matrix& A, std::string filename) {
		ofstream outFile(filename.c_str(), ios::out | ios::binary);
		if (!outFile) {
			std::cerr << "Cannot open output file." << std::endl;
			exit(1);
		}
		// Write the number of dimensions of the matrix to the file.
		int order = A.order();
		outFile.write((char *)&order, sizeof (int));

		for (int n = 0; n != order; ++n) {
			// Write the n'th extent to the file.
			int extent = A.extent(n);
			outFile.write((char *)&extent, sizeof (int));
		}
		// Now write the matrix elements to the file in the same order as they appear in 
		// the 1D backing array.
		float tempVal;
		char * tempCharArray = (char *) &tempVal;
		for (int i = 0; i != A.size(); ++i) {
			tempVal = A[i];
			outFile.write(tempCharArray, sizeof(float));
		}
		outFile.close();
	}

	void save_vector(const std::vector<float> &A, std::string filename) {
		ofstream outFile(filename.c_str(), ios::out | ios::binary);
		if (!outFile) {
			std::cerr << "Cannot open output file." << std::endl;
			exit(1);
		}
		// Write the number of dimensions of the vector to the file.
		int order = 1;
		outFile.write((char *)&order, sizeof (int));

		int extent = static_cast<int>(A.size());
		outFile.write((char *)&extent, sizeof (int));
		
		// Now write the elements to the file in the same order as they appear in 
		// the 1D backing array.
		float tempVal;
		char * tempCharArray = (char *) &tempVal;
		for (size_t i = 0; i != A.size(); ++i) {
			tempVal = A[i];
			outFile.write(tempCharArray, sizeof(float));
		}
		outFile.close();
	}


	Matrix load_matrix(std::string filename) {
		ifstream inFile(filename.c_str(), ios::in | ios::binary);
		if (!inFile) {
			std::cerr << "Cannot open input file." << std::endl;
			exit(1);
		}

		int order;
		// Read in the first int, which specifies the dimension of the matrix.
		inFile.read((char *) &order, sizeof(int));
		if (order == 1) {
			int e0;
			inFile.read((char *) &e0, sizeof(int));
			Matrix A(e0);
			// Now read in the elements
			float tempVal;
			for (int i = 0; i != A.size(); ++i) {
				inFile.read((char *) &tempVal, sizeof(float));
				A[i] = tempVal;
			}
			inFile.close();
			return A;
		} else if (order == 2) {
			int e0;
			int e1;
			inFile.read((char *) &e0, sizeof(int));
			inFile.read((char *) &e1, sizeof(int));
			Matrix A(e0, e1);
			// Now read in the elements
			float tempVal;
			for (int i = 0; i != A.size(); ++i) {
				inFile.read((char *) &tempVal, sizeof(float));
				A[i] = tempVal;
			}
			inFile.close();
			return A;
		} else if (order == 3) {
			int e0;
			int e1;
			int e2;
			inFile.read((char *) &e0, sizeof(int));
			inFile.read((char *) &e1, sizeof(int));
			inFile.read((char *) &e2, sizeof(int));
			Matrix A(e0, e1, e2);
			// Now read in the elements
			float tempVal;
			for (int i = 0; i != A.size(); ++i) {
				inFile.read((char *) &tempVal, sizeof(float));
				A[i] = tempVal;
			}
			inFile.close();
			return A;
		} else if (order == 4) {
			int e0;
			int e1;
			int e2;
			int e3;
			inFile.read((char *) &e0, sizeof(int));
			inFile.read((char *) &e1, sizeof(int));
			inFile.read((char *) &e2, sizeof(int));
			inFile.read((char *) &e3, sizeof(int));
			Matrix A(e0, e1, e2, e3);
			// Now read in the elements
			float tempVal;
			for (int i = 0; i != A.size(); ++i) {
				inFile.read((char *) &tempVal, sizeof(float));
				A[i] = tempVal;
			}
			inFile.close();
			return A;
		} else if (order == 5) {
			int e0;
			int e1;
			int e2;
			int e3;
			int e4;
			inFile.read((char *) &e0, sizeof(int));
			inFile.read((char *) &e1, sizeof(int));
			inFile.read((char *) &e2, sizeof(int));
			inFile.read((char *) &e3, sizeof(int));
			inFile.read((char *) &e4, sizeof(int));
			Matrix A(e0, e1, e2, e3, e4);
			// Now read in the elements
			float tempVal;
			for (int i = 0; i != A.size(); ++i) {
				inFile.read((char *) &tempVal, sizeof(float));
				A[i] = tempVal;
			}
			inFile.close();
			return A;
		} else if (order == 6) {
			int e0;
			int e1;
			int e2;
			int e3;
			int e4;
			int e5;
			inFile.read((char *) &e0, sizeof(int));
			inFile.read((char *) &e1, sizeof(int));
			inFile.read((char *) &e2, sizeof(int));
			inFile.read((char *) &e3, sizeof(int));
			inFile.read((char *) &e4, sizeof(int));
			inFile.read((char *) &e5, sizeof(int));
			Matrix A(e0, e1, e2, e3, e4, e5);
			// Now read in the elements
			float tempVal;
			for (int i = 0; i != A.size(); ++i) {
				inFile.read((char *) &tempVal, sizeof(float));
				A[i] = tempVal;
			}
			inFile.close();
			return A;
		} else {
			std::cerr << "Sorry, only 1 to 6-dimensional matrices are currently supported." << std::endl;
			exit(1);
		}
		
		
	}



}
