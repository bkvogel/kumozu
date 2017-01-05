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

void CrossEntropyCostFunction::reinitialize() {
    std::vector<int> input_extents = get_input_port_data().get_extents();
    m_minibatch_size =  input_extents.at(1);

    m_temp_input_error.resize(input_extents);
    m_exp_input.resize(input_extents);
    m_mu.resize(input_extents);
    m_col_sums.resize(m_minibatch_size);

    // The output activations will only contain 1 value: the cost
    m_output_forward.resize(1);
    m_output_backward.resize(1);
}

void CrossEntropyCostFunction::forward_propagate() {
    if (!m_has_target_activations) {
        error_exit("forward_propagate(): Error: set_target_activations() has not been called yet!");
    }
    const MatrixF& input_activations = get_input_port_data();
    const Matrix<int>& target_activations = m_target_activations;
    //copy_matrix(m_exp_input, input_activations);
    m_exp_input = input_activations;
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
    m_output_forward[0] = cost;
    // Only the gradient-checking functions should ever modify the output_backward activations, so
    // this is probably safe.
    set_value(m_output_backward, 1.0f);
}


void CrossEntropyCostFunction::back_propagate_activation_gradients() {
    //copy_matrix(get_input_port_backward(), m_temp_input_error);
    get_input_port_grad() = m_temp_input_error;
    const float out_back = m_output_backward[0];
    scale(get_input_port_grad(), get_input_port_grad(), out_back);
}

}
