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

#include "CrossEntropyCostFunction.h"
#include "Utilities.h"


using namespace std;

namespace kumozu {

  void CrossEntropyCostFunction::reinitialize(std::vector<int> input_extents) {
    m_minibatch_size =  input_extents.at(1);

    m_temp_input_error.resize(input_extents);
    m_exp_input.resize(input_extents);
    m_mu.resize(input_extents);
    m_col_sums.resize(m_minibatch_size);
  }

  float CrossEntropyCostFunction::forward_propagate(const MatrixF& input_activations, const Matrix<int>& target_activations) {
    copy_matrix(m_exp_input, input_activations);
    const int unit_count = input_activations.extent(0);
    // Begin Optional: Compute max value alpha for each column independently and subtact. This helps prevent overflow:
    // http://deeplearning.stanford.edu/wiki/index.php/Exercise:Softmax_Regression
#pragma omp parallel for
    for (int b = 0; b < m_minibatch_size; ++b) {
      float max_val = 0;
      for (int n = 0; n < unit_count; ++n) {
        if (m_exp_input(n, b) > max_val) {
          max_val = m_exp_input(n, b);
        }
      }
      for (int n = 0; n < unit_count; ++n) {
        m_exp_input(n, b) -= max_val;
      }
    }
    // End Optional

    apply(m_exp_input, [] (float a) {
        return std::exp(a);
      });

    set_value(m_col_sums, 0.0f);
#pragma omp parallel for
    for (int b = 0; b < m_minibatch_size; ++b) {
      for (int n = 0; n < unit_count; ++n) {
        m_col_sums(b) += m_exp_input(n, b);
      }
    }
#pragma omp parallel for
    for (int b = 0; b < m_minibatch_size; ++b) {
      for (int n = 0; n < unit_count; ++n) {
        m_mu(n, b) = m_exp_input(n, b)/m_col_sums(b);
      }
    }
    
#pragma omp parallel for collapse(2)
    for (int m = 0; m < m_minibatch_size; ++m) {
      for (int n = 0; n < unit_count; ++n) {
        if (target_activations(m) == n) {
          m_temp_input_error(n, m) = m_mu(n, m) - 1;
        } else {
          m_temp_input_error(n, m) = m_mu(n, m);
        }
      }
    }
    float cost = 0.0f;
    for (int m = 0; m < m_minibatch_size; ++m) {
      int n = target_activations(m);
      cost -= std::log(m_mu(n, m));
    }
    return cost;
  }


  void CrossEntropyCostFunction::back_propagate(MatrixF& input_error, const MatrixF& input_activations,
                                                const Matrix<int>& target_activations) {
    copy_matrix(input_error, m_temp_input_error);
  }


  void CrossEntropyCostFunction::check_gradients(std::vector<int> input_extents) {
    // Create random input activations for the layer.
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    MatrixF input_deltas(input_extents);
    randomize_uniform(input_deltas, 0.0f, 1.0f);
    const int minibatch_count = input_extents.at(1);
    Matrix<int> target_activations(minibatch_count);
    MatrixF temp_rand(minibatch_count);
    randomize_uniform(temp_rand, 0.0f, input_extents.at(0));

    for (int b = 0; b < minibatch_count; ++b) {
      // Use random class lables.
      target_activations(b) = static_cast<int>(temp_rand(b));
    }
    //randomize_uniform(target_activations, 0.0f, 1.0f);
    //randomize_uniform(target_activations, 0.0f, 1.0f);
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
