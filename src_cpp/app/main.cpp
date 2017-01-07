/*
 * Copyright (c) 2005-2016, Brian K. Vogel
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
#include <cstdlib>
#include <string.h>
#include <iostream>
#include <stdexcept>
#include <omp.h>

#include "UnitTests.h"
#include "ExamplesConvNet.h"
#include "ExamplesRecSystem.h"
#include "ExamplesPlot.h"
#include "ExamplesMatrix.h"
#include "ExamplesRNN.h" 
#include "ExamplesNMF.h"

using namespace std;
using namespace kumozu;

void run_unit_tests();

/*
 * Supply one argument that specifies the name of the example to run.
 */
int main(int argc, char* argv[]) {

    try {
        // Set number of threads for OpenMP and OpenBlas.
        //omp_set_num_threads(8); //  8-core
        //omp_set_num_threads(4); // 4-core
        //omp_set_num_threads(2); // 2-core
        // Show first NaN exception in gdb. Only works on Linux.
        //feenableexcept(FE_INVALID | FE_OVERFLOW);

        if (argc == 2) {
            std::string methodName = argv[1];
            if (methodName == "example_plots") {
                example_plots();
                example_image_plots();
                example_image_plots2();
                example_image_plots_rgb();
                example_multiplot();
            }
            if (methodName == "matrix_mult_benchmark") {
                // Run to make sure BLAS library is set up correctly and gives reasonable performance.
                benchmark_mat_mult();
            }
            else if (methodName == "test") {
                // Unit tests
                run_unit_tests();
            }
            else if (methodName == "matrix_examples") {
                // Matrix utility examples:
                example_matrix_utilities();
            }
            else if (methodName == "cifar10_example_1") {
                cifar10_example_1();
            }
            else if (methodName == "mnist_example_1") {
                mnist_example_1();
            }
            else if (methodName == "netflix_prize_example_1") {
                netflix_prize_example_1();
            }
            else if (methodName == "lstm_example") {
                // LSTM RNN example.
                lstm_example();
            }
            //
            else if (methodName == "nmf_example_1") {
                nmf_example_1();
            }
            else if (methodName == "nmf_example_2") {
                //nmf_mnist_example_1();
            }
            else if (methodName == "debug") {
                //call debug function here.
            }
        }
    }
    catch (out_of_range) {
        cerr << "Out of range error." << endl;
    }
    catch (...) {
        cerr << "Unkown exception!" << endl;
    }


    return EXIT_SUCCESS;
}

void run_unit_tests() {

    //test_mat_mult();
    //benchmark_mat_mult();
    //stress_test_forward_prop();

    // Run all unit tests:
    run_all_tests();
}
