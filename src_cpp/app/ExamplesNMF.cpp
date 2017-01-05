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

#include "ExamplesNMF.h"
#include <cstdlib>
#include <string.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <ctime>
#include <omp.h>
#include <algorithm>
#include <memory>

#include "Matrix.h"
#include "MatrixIO.h"
#include "UnitTests.h"
#include "PlotUtilities.h"
#include "Utilities.h"
#include "NmfUpdaterLeeSeung.h"


using namespace std;

namespace kumozu {


void nmf_example_1() {
    cout << "nmf_example_1()..." << endl;
    // Load spectrogram image that was generated in Python.
    MatrixF spectrogram = load_matrix("X.dat");
    const int rows_X = spectrogram.extent(0);
    const int cols_X = spectrogram.extent(1);
    const int basis_vector_count = 6;
    MatrixF W(rows_X, basis_vector_count);
    randomize_uniform(W, 0.0f, 0.1f);
    MatrixF H(basis_vector_count, cols_X);
    randomize_uniform(H, 0.0f, 0.1f);

    NmfUpdaterLeeSeung nmf_updater("X = W*H");

    const int max_iterations = 200;
    const int plot_every = 20;
    Gnuplot x_eq_w_h_plot;
    for (int i = 0; i < max_iterations; ++i) {
        cout << "iterations: " << i << endl;
        if ((i % plot_every) == 0) {
            x_eq_w_h_plot << "set multiplot layout 1,3 title 'X eq W x H'" << endl;
            const vector<string> common_options = {"set palette rgb 21,22,23", "set colorbox",
                                                   "set tics font \", 5\""}; // hot
            plot_image_greyscale(x_eq_w_h_plot, spectrogram, "X", common_options);
            plot_image_greyscale(x_eq_w_h_plot, W, "W", common_options);
            plot_image_greyscale(x_eq_w_h_plot, H, "H", common_options);
            x_eq_w_h_plot << "unset multiplot" << endl;
        }
        nmf_updater.right_update(spectrogram, W, H);
        nmf_updater.left_update(spectrogram, W, H);
        nmf_updater.print_rmse(spectrogram, W, H);
    }

}


}
