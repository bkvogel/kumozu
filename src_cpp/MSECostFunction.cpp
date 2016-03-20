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

#include "MSECostFunction.h"
#include "Utilities.h"


using namespace std;

namespace kumozu {

  void MSECostFunction::reinitialize() {
    std::vector<int> input_extents = get_input_port_forward().get_extents();
    m_minibatch_size =  input_extents.at(1);

    m_temp_input_error.resize(input_extents);
    m_temp_size_input.resize(input_extents);

    // The output activations will only contain 1 value: the cost
    m_output_forward.resize(1);
    m_output_backward.resize(1);
  }

  void MSECostFunction::forward_propagate() {
    if (!m_has_target_activations) {
      error_exit("forward_propagate(): Error: set_target_activations() has not been called yet!");
    }
    const MatrixF& input_activations = get_input_port_forward();
    const MatrixF& target_activations = m_target_activations;
    element_wise_difference(m_temp_input_error, input_activations, target_activations);
    copy_matrix(m_temp_size_input, m_temp_input_error);
    apply(m_temp_size_input, [] (float a) {
        return a*a;
      });
    m_output_forward[0] = 0.5*sum(m_temp_size_input);
    // Only the gradient-checking functions should ever modify the output_backward activations, so
    // this is probably safe.
    set_value(m_output_backward, 1.0f);
  }

  void MSECostFunction::back_propagate_deltas() {
    copy_matrix(get_input_port_backward(), m_temp_input_error); // old way
    const float out_back = m_output_backward[0];
    scale(get_input_port_backward(), get_input_port_backward(), out_back);
  }

}
