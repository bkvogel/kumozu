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
    if (VERBOSE_MODE) {
        std::cout << "ColumnActivationFunction:" << std::endl;
        if (m_activation_type == ACTIVATION_TYPE::ReLU) {
            std::cout << "Using ReLU activation:" << std::endl;
        } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
            std::cout << "Using leakyReLU activation:" << std::endl;
        } else if (m_activation_type == ACTIVATION_TYPE::identity) {
            std::cout << "Using identity activation:" << std::endl;
        } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
            std::cout << "Using maxout activation:" << std::endl;
        } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
            std::cout << "Using kmax activation:" << std::endl;
        } else if (m_activation_type == ACTIVATION_TYPE::tanh) {
            std::cout << "Using tanh activation:" << std::endl;
        } else if (m_activation_type == ACTIVATION_TYPE::sigmoid) {
            std::cout << "Using sigmoid activation:" << std::endl;
        }
    }
    const int dim_output = input_extents.at(0)/m_maxout_factor;
    m_output_data.resize(dim_output, input_extents.at(1));
    m_output_grad.resize(dim_output, input_extents.at(1));
    m_state.resize(dim_output, input_extents.at(1));
    if (VERBOSE_MODE) {
        std::cout << "dim_input = " << input_extents.at(0) << std::endl;
        std::cout << "mini-batch size = " << input_extents.at(1) << std::endl;
        std::cout << "dim_output = " << dim_output << std::endl;
    }
}

void ColumnActivationFunction::forward_propagate(const MatrixF& input_forward) {
    //reinitialize(input_forward.get_extents());
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
        compute_forward_relu(input_forward, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU_decay_unused) {
        // ReLU and ReLU_decay_unused use excatly the same forward activation.
        compute_forward_relu(input_forward, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
        compute_forward_leaky_relu(input_forward, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::identity) {
        compute_forward_identity_activation(input_forward, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
        forward_maxout(input_forward, m_output_data, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout_decay_unused) {
        // maxout and maxout_decay_unused use excatly the same forward activation.
        forward_maxout(input_forward, m_output_data, m_state);
    }  else if (m_activation_type == ACTIVATION_TYPE::kmax) {
        compute_forward_kmax_v2(input_forward, m_output_data, m_state, m_partition_count, m_k);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax_decay_unused) {
        // kmax and kmax_decay_unused use excatly the same forward activation.
        compute_forward_kmax_v2(input_forward, m_output_data, m_state, m_partition_count, m_k);
    } else if (m_activation_type == ACTIVATION_TYPE::tanh) {
        compute_forward_tanh(input_forward, m_output_data);
    } else if (m_activation_type == ACTIVATION_TYPE::sigmoid) {
        compute_forward_sigmoid(input_forward, m_output_data);
    } else {
        error_exit("forward_propagate(): Unrecognized activation type!");
    }
}

void ColumnActivationFunction::back_propagate_activation_gradients(MatrixF& input_grad, const MatrixF& input_data) {
    if (m_activation_type == ACTIVATION_TYPE::ReLU) {
        compute_reverse_relu(input_grad, m_output_grad, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
        compute_reverse_leaky_relu(input_grad, m_output_grad, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::identity) {
        compute_reverse_identity_activation(input_grad, m_output_grad, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
        compute_reverse_maxout(input_grad, m_output_grad, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout_decay_unused) {
        compute_reverse_maxout_decay_unused(input_grad, input_data, m_output_grad,
                                            m_state, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
        compute_reverse_kmax_v2(input_grad, m_output_grad, m_state, m_partition_count, m_k);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax_decay_unused) {
        // kmax and kmax_decay_unused use excatly the same forward activation.
        compute_reverse_kmax_decay_unused(input_grad, input_data, m_output_grad,
                                          m_state, m_partition_count, m_k, m_decay_unused_penalty);
    }  else if (m_activation_type == ACTIVATION_TYPE::tanh) {
        compute_reverse_tanh(input_grad, m_output_data, m_output_grad);
    } else if (m_activation_type == ACTIVATION_TYPE::sigmoid) {
        compute_reverse_sigmoid(input_grad, m_output_data, m_output_grad);
    } else {
        error_exit("back_propagate_deltas(): Unrecognized activation type!");
    }
}

// To use this version, need to modify class to save the input acitvations on the forward pass so they can be used here.
/*
    void ColumnActivationFunction::reverse_activation(MatrixF& input_backward, const MatrixF& input) {
    if (m_activation_type == ACTIVATION_TYPE::kmax_decay_unused) {
    compute_reverse_kmax_decay_unused(input_backward, input, m_output_backward, m_state,
    m_partition_count, m_k, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU_decay_unused) {
    compute_reverse_relu_decay_unused(input_backward, input, m_output_backward, m_state, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout_decay_unused) {
    compute_reverse_maxout_decay_unused(input_backward, input, m_output_backward, m_state, m_decay_unused_penalty);
    } else if (m_activation_type == ACTIVATION_TYPE::ReLU) {
    compute_reverse_relu(input_backward, m_output_backward, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::leakyReLU) {
    compute_reverse_leaky_relu(input_backward, m_output_backward, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::identity) {
    compute_reverse_identity_activation(input_backward, m_output_backward, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::maxout) {
    compute_reverse_maxout(input_backward, m_output_backward, m_state);
    } else if (m_activation_type == ACTIVATION_TYPE::kmax) {
    compute_reverse_kmax_v2(input_backward, m_output_backward, m_state, m_partition_count, m_k);
    } else {
    cerr << "reverse_activation(): Unrecognized activation type!" << endl;
    }
    }
  */


}
