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
#include "ColumnActivationFunction.h"

#include "Utilities.h"
using namespace std;

namespace kumozu {

  void ColumnActivationFunction::reinitialize(std::vector<int> input_extents) {
    if (input_extents == m_input_extents) {
      // Extents already match, so nothing to do.
      return;
    }
    std::cout << "Initializing " << m_layer_name << " ..." << std::endl;
    m_input_extents = input_extents;
    //Layer::reinitialize(input_extents);
    // Note: input_extents.at(1) is mini-batch size.
    // input_extents.at(1) is dim_input.

    std::cout << "ColumnActivationFunction:" << std::endl;
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
      std::cout << "Using ReLU activation:" << std::endl;
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
      std::cout << "Using leakyReLU activation:" << std::endl;
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
      std::cout << "Using linear activation:" << std::endl;
    } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
      std::cout << "Using maxout activation:" << std::endl;
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
      std::cout << "Using kmax activation:" << std::endl;
    }

    const int dim_output = input_extents.at(0)/m_maxout_factor;
    m_output_activations.resize(dim_output, input_extents.at(1));
    m_output_error.resize(dim_output, input_extents.at(1));
    m_state.resize(dim_output, input_extents.at(1));

    std::cout << "dim_input = " << input_extents.at(0) << std::endl;
    std::cout << "mini-batch size = " << input_extents.at(1) << std::endl;
    std::cout << "dim_output = " << dim_output << std::endl;
  }

  void ColumnActivationFunction::forward_propagate(const MatrixF& input_activations) {
    //reinitialize(input_activations.get_extents());
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
      compute_forward_relu(input_activations, m_output_activations, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU_decay_unused) {
      // ReLU and ReLU_decay_unused use excatly the same forward activation.
      compute_forward_relu(input_activations, m_output_activations, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
      compute_forward_leaky_relu(input_activations, m_output_activations, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
      compute_forward_identity_activation(input_activations, m_output_activations, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
      compute_forward_maxout(input_activations, m_output_activations, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout_decay_unused) {
      // maxout and maxout_decay_unused use excatly the same forward activation.
      compute_forward_maxout(input_activations, m_output_activations, m_state);
    }  else if (m_activation_type == ACTIVATION_TYPE::kmax) {
      compute_forward_kmax_v2(input_activations, m_output_activations, m_state, m_partition_count, m_k);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax_decay_unused) {
      // kmax and kmax_decay_unused use excatly the same forward activation.
      compute_forward_kmax_v2(input_activations, m_output_activations, m_state, m_partition_count, m_k);
    } else {
      cerr << "forward_propagate(): Unrecognized activation type!" << endl;
    }
  }

  void ColumnActivationFunction::back_propagate_deltas(MatrixF& input_error) {
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
      compute_reverse_relu(input_error, m_output_error, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
      compute_reverse_leaky_relu(input_error, m_output_error, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
      compute_reverse_identity_activation(input_error, m_output_error, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
      compute_reverse_maxout(input_error, m_output_error, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
      compute_reverse_kmax_v2(input_error, m_output_error, m_state, m_partition_count, m_k);
    } else {
      cerr << "back_propagate_deltas(): Unrecognized activation type!" << endl;
    }
  }

  // To use this version, need to modify class to save the input acitvations on the forward pass so they can be used here.
  /*
    void ColumnActivationFunction::reverse_activation(MatrixF& input_deltas, const MatrixF& input) {
    if (m_activation_type == ACTIVATION_TYPE::kmax_decay_unused) {
    compute_reverse_kmax_decay_unused(input_deltas, input, m_output_deltas, m_state,
    m_partition_count, m_k, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU_decay_unused) {
    compute_reverse_relu_decay_unused(input_deltas, input, m_output_deltas, m_state, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout_decay_unused) {
    compute_reverse_maxout_decay_unused(input_deltas, input, m_output_deltas, m_state, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU) {
    compute_reverse_relu(input_deltas, m_output_deltas, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
    compute_reverse_leaky_relu(input_deltas, m_output_deltas, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
    compute_reverse_identity_activation(input_deltas, m_output_deltas, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
    compute_reverse_maxout(input_deltas, m_output_deltas, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
    compute_reverse_kmax_v2(input_deltas, m_output_deltas, m_state, m_partition_count, m_k);
    } else {
    cerr << "reverse_activation(): Unrecognized activation type!" << endl;
    }
    }
  */


}
