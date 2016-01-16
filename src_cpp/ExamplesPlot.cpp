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

#include "ExamplesPlot.h"
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

#include "Matrix.h"
#include "Utilities.h"
#include "Constants.h"
#include "PlotUtilities.h"

// Uncomment following line to disable assertion checking.
//#define NDEBUG
// Uncomment to enable assertion checking.
#undef NDEBUG
#include <assert.h>

using namespace std;

namespace kumozu {

  void example_plots() {
    cout << "example_plots()..." << endl;

    // plot vector
    vector<float> a;
    for (int i=0; i != 100; ++i) {
      a.push_back(sqrt(i) - 2.5f);
    }
    plot(a, "A plot with a title.");
    plot(a, "A plot with axis labels.", {"set xlabel 'elapsed time, minutes'", "set ylabel 'hotdogs per minute'"});

  }

  void example_image_plots() {
    cout << "example_image_plots()..." << endl;
    const int height = 5;
    const int width = 8;
    MatrixF image(height, width);
    image(0,0) = 1.0f;
    image(3,2) = 0.5f;
    image(4,7) = 0.2f;

    plot_image_greyscale(image, "A greyscale plot of 2D matrix.");

  }

  void example_image_plots2() {
    cout << "example_image_plots2()..." << endl;

    const int height = 240;
    const int width = 320;
    MatrixF image(height, width);
    for (int i = 0; i != height; ++i) {
      for (int j = 0; j != width; ++j) {
        float x = (i-0.5*height)/5.0f;
        float y = (j - 0.5*width)/5.0f;
        float z = std::cos(sqrt(x*x + y*y));
        image(i,j) = z;
      }
    }

    //plot_image_greyscale(image, "Some function.", {"set palette model XYZ rgbformulae 7,5,15", "set colorbox", "set tics"});
    plot_image_greyscale(image, "Some function.", {"set palette color", "set colorbox", "set tics"});

  }

  void example_image_plots_rgb() {
    cout << "example_image_plots_rgb()..." << endl;
    const int depth = 3; // For RGB, must be 3.
    const int height = 5;
    const int width = 8;
    MatrixF image(depth, height, width);
    image(0,0,0) = 255.0f; // red
    image(1,2,3) = 255.0f; // green
    image(2,3,4) = 255.0f; // blue

    // White corner
    image(0,4,7) = 255.0f; // red
    image(1,4,7) = 255.0f; // green
    image(2,4,7) = 255.0f; // blue

    // Grey corner
    image(0,4,0) = 128.0f; // red
    image(1,4,0) = 128.0f; // green
    image(2,4,0) = 128.0f; // blue
    plot_image_rgb(image, "RGB test plot.");
  }

  void example_multiplot() {
    cout << "example_multiplot" << endl;

    // Make data for plot #1
    vector<float> a;
    for (int i=0; i != 100; ++i) {
      a.push_back(sqrt(i) - 2.5f);
    }

    // Make data for plot #2
    const int height = 5;
    const int width = 8;
    MatrixF image(height, width);
    image(0,0) = 1.0f;
    image(3,2) = 0.5f;
    image(4,7) = 0.2f;

    // Make data for plot #3
    const int depth = 3; // For RGB, must be 3.

    MatrixF imageb(depth, height, width);
    imageb(0,0,0) = 255.0f; // red
    imageb(1,2,3) = 255.0f; // green
    imageb(2,3,4) = 255.0f; // blue

    // White corner
    imageb(0,4,7) = 255.0f; // red
    imageb(1,4,7) = 255.0f; // green
    imageb(2,4,7) = 255.0f; // blue

    // Grey corner
    imageb(0,4,0) = 128.0f; // red
    imageb(1,4,0) = 128.0f; // green
    imageb(2,4,0) = 128.0f; // blue


    // Now get a plot window instance and make the plots.
    Gnuplot gp;
    //gp << "set multiplot layout 1,3 title \"Three plots in a figure\"" << endl;
    gp << "set multiplot layout 1,3 title 'Three plots in a figure'" << endl;

    // plot #1
    plot(gp, a, "A plot with axis labels.", {"set xlabel 'elapsed time, minutes'", "set ylabel 'hotdogs per minute'"});

    // plot #2
    plot_image_greyscale(gp, image, "A greyscale plot of 2D matrix.");

    // plot #3
    plot_image_rgb(gp, imageb, "RGB test plot.");

    gp << "unset multiplot" << endl;
  }


}
