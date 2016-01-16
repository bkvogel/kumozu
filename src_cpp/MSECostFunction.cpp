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

  void MSECostFunction::reinitialize(std::vector<int> input_extents) {
    m_minibatch_size =  input_extents.at(1);

    m_temp_input_error = MatrixF(input_extents);
    m_temp_size_input = MatrixF(input_extents);
  }

  float MSECostFunction::forward_propagate(const MatrixF& input_activations, const MatrixF& target_activations) {
    element_wise_difference(m_temp_input_error, input_activations, target_activations);
    copy_matrix(m_temp_size_input, m_temp_input_error);
    apply(m_temp_size_input, [] (float a) {
        return a*a;
      });
    return 0.5*sum(m_temp_size_input);
  }

  void MSECostFunction::back_propagate(MatrixF& input_error, const MatrixF& input_activations,
                                       const MatrixF& true_output_activations) {
    copy_matrix(input_error, m_temp_input_error);
  }

}
