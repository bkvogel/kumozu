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
#include "LinearLayer.h"
#include "Utilities.h"

using namespace std;

namespace kumozu {

  void LinearLayer::reinitialize(std::vector<int> input_extents) {
    string indent = "    ";
    m_input_layer_units = input_extents.at(0);
    m_output_layer_units = m_dim_output;
    m_minibatch_size = input_extents.at(1);
    m_output_forward.resize(m_dim_output, m_minibatch_size);
    m_output_backward.resize(m_dim_output, m_minibatch_size);
    const std::vector<int> new_W_extents = {m_dim_output, m_input_layer_units};
    if (new_W_extents != get_weights().get_extents()) {
      // Only reiniitalize W if its size has changed. Note that simply changing the mini-batch
      // size should not cause W to change size.
      get_weights().resize(new_W_extents);
      m_temp_size_W.resize(new_W_extents);
      get_weight_gradient().resize(new_W_extents);
      // If we reinitialize W, must reinitialize bias as well.
      get_bias().resize(m_dim_output);
      m_temp_size_bias.resize(m_dim_output);
      get_bias_gradient().resize(m_dim_output);

      const float std_dev_init = 1.0f/std::sqrt(m_input_layer_units); // default
      randomize_uniform(get_weights(), -std_dev_init, std_dev_init); // default
      //const float std_dev_init = 2.0f*std::sqrt(2.0f)/std::sqrt(m_input_layer_units);
      //randomize_normal(m_W, 0.0f, std_dev_init);
      m_W_fixed_random = get_weights();
      if (VERBOSE_MODE) {
        std::cout << indent << "Initialized weights with std dev = " << std_dev_init << std::endl;
        std::cout << indent << "Input layer units = " << m_input_layer_units << std::endl;
        std::cout << indent << "Output layer units = " << m_output_layer_units << std::endl;
      }
    }
    m_output_forward_indices = Matrix<int>(m_dim_output, m_minibatch_size);
  }


  void LinearLayer::forward_propagate(const MatrixF& input_forward) {
    if (!is_enable_bias()) {
      set_value(get_bias(), 0.0f);
    }

    // m_Z = m_W * m_input_forward + m_bias
    do_product_update(m_output_forward, get_weights(), input_forward, get_bias());
  }

  void LinearLayer::back_propagate_paramater_gradients(const MatrixF& input_forward) {
    // Compute grad_W = m_Z_error*m_input_forward^T
    compute_weight_grad_sgd_minibatch(m_output_backward, get_weight_gradient(), input_forward);
    compute_bias_grad_sgd_minibatch(m_output_backward, get_bias_gradient());
  }

  void LinearLayer::back_propagate_deltas(MatrixF& input_backward, const MatrixF& input_forward) {
    if (m_use_fixed_random_back_prop) {
      do_backprop_update_sgd_minibatch(m_output_backward, m_W_fixed_random, input_backward); // fixed-random backpropagation
    } else {
      do_backprop_update_sgd_minibatch(m_output_backward, get_weights(), input_backward); // Usual backpropagation
    }
  }

}
