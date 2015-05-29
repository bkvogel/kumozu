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
#ifndef _MATRIXIO_H
#define	_MATRIXIO_H

#include "MatrixT.h"
#include <string>
#include <vector>

/*
 * This class provides methods for loading and saving an N-dimensional matrix or vector to a file.
 *
 * Currently, only float-valued matrices are supported. Todo: add support for other numeric data types.
 *
 */
namespace kumozu {

    /*
     * Save the contents of the supplied Matrix to a file. The data is saved as
     * float type.
     *
     * @param A The matrix to write to a file.
     * @param filename The name of the file. The contents of A will be written to
     * this file. Any existing file with the same name will be overwritten.
     */
    void save_matrix(const Matrix &A, std::string filename);

	/*
     * Save the contents of the supplied vector to a file. The data is saved as
     * float type.
     *
     * @param A The vector to write to a file.
     * @param filename The name of the file. The contents of A will be written to
     * this file. Any existing file with the same name will be overwritten.
     */
void save_vector(const std::vector<float> &A, std::string filename);

	/*
	* Load a matrix from a file. The data in the file is represented as float
	* type.
	*
	* @param filename The name of the file containing the matrix data.
	* @return a new matrix containing the contents of file "filename."
	*
	*/
	Matrix load_matrix(std::string filename);


}

#endif	/* _MATRIXIO_H */

