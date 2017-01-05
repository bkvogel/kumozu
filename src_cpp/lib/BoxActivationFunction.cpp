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
#include "BoxActivationFunction.h"

#include "Utilities.h"
using namespace std;

namespace kumozu {

  void ImageActivationFunction::reinitialize(std::vector<int> input_extents) {
    string indent = "    ";
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
      std::cout << indent << "Using ReLU activation:" << std::endl;
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
      std::cout << indent << "Using leakyReLU activation:" << std::endl;
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
      std::cout << indent << "Using linear activation:" << std::endl;
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
      std::cout << indent << "Using kmax activation:" << std::endl;
    }
    m_minibatch_size = input_extents.at(0);
    m_depth = input_extents.at(1);
    m_height = input_extents.at(2);
    m_width = input_extents.at(3);
    m_output_data.resize(m_minibatch_size, m_depth, m_height, m_width);
    m_output_grad.resize(m_minibatch_size, m_depth, m_height, m_width);
    m_state.resize(m_minibatch_size, m_depth, m_height, m_width);
  }

  void ImageActivationFunction::forward_propagate(const MatrixF& input_activations) {
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
      compute_forward_relu(input_activations, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU_decay_unused) {
      compute_forward_relu(input_activations, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
      compute_forward_leaky_relu(input_activations, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
      compute_forward_identity_activation(input_activations, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
      forward_3d_kmax(input_activations, m_output_data, m_state, m_box_depth, m_box_height, m_box_width, m_k);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax_decay_unused) {
      forward_3d_kmax(input_activations, m_output_data, m_state, m_box_depth, m_box_height, m_box_width, m_k);
    }
  }

  void ImageActivationFunction::back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) {
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
      compute_reverse_relu(input_backward, m_output_grad, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
      compute_reverse_leaky_relu(input_backward, m_output_grad, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
      compute_reverse_identity_activation(input_backward, m_output_grad, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
      reverse_3d_kmax(input_backward, m_output_grad, m_state);
    }
  }

  // To use this version, need to modify class to save the input acitvations on the forward pass so they can be used here.
  /*
    void BoxActivationFunction::reverse_activation(MatrixF& input_backward, const MatrixF& input) {
    if (m_activation_type == ACTIVATION_TYPE::kmax_decay_unused) {
    reverse_3d_kmax_decay_unused(input_backward, input, m_output_backward, m_state, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU_decay_unused) {
    compute_reverse_relu_decay_unused(input_backward, input, m_output_backward, m_state, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU) {
    compute_reverse_relu(input_backward, m_output_backward, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
    compute_reverse_leaky_relu(input_backward, m_output_backward, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::linear) {
    compute_reverse_identity_activation(input_backward, m_output_backward, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
    reverse_3d_kmax(input_backward, m_output_backward, m_state);
    } else {
    cerr << "reverse_activation(): Unrecognized activation type!" << endl;
    }
    }
  */

}
