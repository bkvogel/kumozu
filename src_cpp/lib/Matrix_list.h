#ifndef _MATRIXLIST_H
#define _MATRIXLIST_H
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
#include <string>

namespace kumozu {

/**
 * Represents a single element in a matrix of floats at the
 * specified row and column.
 */
struct Matrix_element {
    int row;
    int col;
    float val;
};

/**
 * An instance of this class represents a sparse matrix of floats.
 * The matrix supports fast iteration through its elements in the same order
 * that they were added.
 *
 * This matrix is backed by a vector of Matrix_element, which is publicly
 * ascesable.
 */
class Matrix_list {

public:

    /**
     * Construct a Matrix_list from the supplied dense matrix.
     */
    Matrix_list(const MatrixF& A);

    Matrix_list(int row_count, int col_count);


    // Number of rows (not all of them necesarily used)
    int rows;
    // Number of columns (not all of them necesarily used)
    int columns;

    // List of the observed elements in the matrix.
    std::vector<Matrix_element> element_list;

private:
    Matrix_list() {}

};

std::ostream& operator<<(std::ostream& os, const Matrix_list& m);

}
#endif
