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

#include "CostFunction.h"
#include "Utilities.h"


using namespace std;

namespace kumozu {


  void CostFunction::check_gradients(std::vector<int> input_extents) {
    // Create random input activations for the layer.
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    MatrixF input_deltas(input_extents);
    randomize_uniform(input_deltas, 0.0f, 1.0f);
    MatrixF target_activations(input_extents);
    //randomize_uniform(target_activations, 0.0f, 1.0f);
    randomize_uniform(target_activations, 0.0f, 1.0f);
    float cost = forward(input_activations, target_activations);
    back_propagate(input_deltas, input_activations, target_activations);
    cout << "Cost = " << cost << endl;

    MatrixF gradients_numerical = input_deltas; // Temp matrix to hold the numerical gradients.
    set_value(gradients_numerical, 0.0f);
    for (int n = 0; n != input_activations.size(); ++n) {
      float orig = input_activations[n]; // backup
      input_activations[n] += m_epsilon;
      // Now compute J(theta_plus)
      float J_plus = forward(input_activations, target_activations);
      // Now compute J(theta_minus)
      input_activations[n] = orig - m_epsilon;
      float J_minus = forward(input_activations, target_activations);
      // Put back original value.
      input_activations[n] = orig;
      gradients_numerical[n] = (J_plus - J_minus)/(2*m_epsilon);
    }
    const float relative_error_score = relative_error(input_deltas, gradients_numerical);
    std::cout << "numerical-back-prop gradients relative error = " << relative_error_score << std::endl;
    std::cout << "input_deltas = " << std::endl << input_deltas << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "gradients_numerical = " << std::endl << gradients_numerical << std::endl;
    assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);

  }



}
