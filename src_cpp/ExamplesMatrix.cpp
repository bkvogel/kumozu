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

#include "ExamplesMatrix.h"
#include "Matrix.h"
#include "MatrixIO.h"
#include <cstdlib>
#include "Utilities.h"

// Uncomment following line to disable assertion checking.
//#define NDEBUG
// Uncomment to enable assertion checking.
#undef NDEBUG
#include <assert.h>

using namespace std;

namespace kumozu {

  void example_matrix_utilities() {
    // Run this and look at stdout to see examples using the multi-dimensional matrix.
    // Note: MatrixF is a typedef for Matrix<float>
    cout << "Matrix utility examples." << endl;

    cout << "// 3 x 4 matrix, initialized to 0." << endl;
    cout << "MatrixF X(3,4);" << endl;
    MatrixF X(3,4);

    cout << "cout << X << endl;" << endl;
    cout << X << endl;

    cout << "// Create range of values in X." << endl;

    cout << "float i = 0;" << endl;
    float i = 0;
    cout << "// Fill up X." << endl;
    cout << R"(apply_sequential(X, [&] (float a) {
        // Ignore the value of a.
        i += 0.1f;
        return i;
      });)" << endl;
    apply_sequential(X, [&] (float a) {
        // Ignore the value of a.
        i += 0.1f;
        return i;
      });

    cout << "cout << X << endl;" << endl;
    cout << X << endl;

    cout << "// Apply sqrt() to all elements in X." << endl;
    cout << R"(apply(X, [] (float a) {
        return std::sqrt(a);
      });)" << endl;

    apply(X, [] (float a) {
        return std::sqrt(a);
      });

    cout << "cout << X << endl;" << endl;
    cout << X << endl;

    cout << "MatrixF Y(4,5);" << endl;
    MatrixF Y(4,5);

    cout << "cout << Y << endl;" << endl;
    cout << Y << endl;

    cout << "i = 0;" << endl;
    i = 0;

    cout << R"(apply_sequential(Y, [&] (float a) {
        // Ignore the value of a.
        i += 1.0f;
        return i;
      });)" << endl;
    apply_sequential(Y, [&] (float a) {
        // Ignore the value of a.
        i += 1.0f;
        return i;
      });

    cout << endl << "// map2() example:" << endl;

    cout << "cout << Y << endl;" << endl;
    cout << Y << endl;

    cout << "MatrixF A = Y;" << endl;
    MatrixF A = Y;


    cout << "cout << A << endl;" << endl;
    cout << A << endl;

    cout << "MatrixF B = Y;" << endl;
    MatrixF B = Y;

    cout << "cout << B << endl;" << endl;
    cout << B << endl;

    cout << "// Apply map2(): " << endl;
    cout << R"(map2(Y, A, B, [] (float a, float b) {
        return a + 2*b;
      });)" << endl << endl;

    map2(Y, A, B, [] (float a, float b) {
        return a + 2*b;
      });

    cout << "cout << Y << endl;" << endl;
    cout << Y << endl;

    cout << "// narrow() example:" << endl;

    i = 0;

    cout << R"(apply_sequential(Y, [&] (float a) {
        // Ignore the value of a.
        i += 1.0f;
        return i;
      });)" << endl;
    apply_sequential(Y, [&] (float a) {
        // Ignore the value of a.
        i += 1.0f;
        return i;
      });

    cout << "cout << Y << endl;" << endl;
    cout << Y << endl;

    cout << "MatrixF D = narrow(Y, 1, 1, 2);" << endl;
    MatrixF D = narrow(Y, 1, 1, 2);

    cout << "cout << D << endl;" << endl;
    cout << D << endl;

    cout << "// Now randomize D:" << endl;
    cout << "randomize_normal(D, 1.0f, 1.0f);" << endl;
    randomize_normal(D, 1.0f, 1.0f);

    cout << "cout << D << endl;" << endl;
    cout << D << endl;

    cout << "// Now copy data from D back into same locations in Y:" << endl;
    cout << "reverse_narrow(D, Y, 1, 1, 2);" << endl;
    reverse_narrow(D, Y, 1, 1, 2);

    cout << "cout << Y << endl;" << endl;
    cout << Y << endl;

    cout << "// Matrix multilication example:" << endl;
    cout << "MatrixF U(3,4);" << endl;
    MatrixF U(3,4);
    cout << "cout << U << endl;" << endl;
    cout << U << endl;
    cout << "randomize_uniform(U, -1.0f, 1.0f);" << endl;
    randomize_uniform(U, -1.0f, 1.0f);
    cout << "cout << U << endl;" << endl;
    cout << U << endl;
    cout << "MatrixF R(4,5);" << endl;
    MatrixF R(4,5);
    cout << "cout << R << endl;" << endl;
    cout << R << endl;
    cout << "set_value(R, 1.0f);" << endl;
    set_value(R, 1.0f);
    cout << "cout << R << endl;" << endl;
    cout << R << endl;    
    cout << "// Compute C = U*R:" << endl;
    cout << "MatrixF C;" << endl;
    MatrixF C;
    cout << "// Note: C has not been initialized to the required output dimensions but will be " << endl;
    cout << "// resized to the correct dimensions inside the matrix multiplication function." << endl;
    cout << "// Many of the matrix utility functions work like this (auto re-sizing of result)." << endl;
    cout << "mat_multiply(C, U, R);" << endl;
    mat_multiply(C, U, R);
    cout << "cout << C << endl;" << endl;
    cout << C << endl;

  }


}
