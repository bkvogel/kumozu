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
#include "ConvLayer2D.h"
#include "Utilities.h"
#include "MatrixIO.h"

using namespace std;

namespace kumozu {

void ConvLayer2D::reinitialize(std::vector<int> input_extents) {
    cout << get_name() << endl;
    string indent = "    ";
    m_minibatch_size = input_extents.at(0);
    m_image_depth = input_extents.at(1);
    m_image_height = input_extents.at(2);
    m_image_width = input_extents.at(3);
    m_output_data.resize(m_minibatch_size, m_filter_count, m_image_height, m_image_width);
    m_output_grad.resize(m_minibatch_size, m_filter_count, m_image_height, m_image_width);
    const std::vector<int> new_W_extents = {m_filter_count, m_image_depth, m_conv_filter_height, m_conv_filter_width};
    auto& W = get_param("W");
    auto& bias = get_param("bias");
    if (new_W_extents != W.get_extents()) {
        W.resize(new_W_extents);
        m_temp_size_W.resize(new_W_extents);
        //get_weight_gradient().resize(new_W_extents);
        bias.resize(m_filter_count);
        m_temp_size_bias.resize(m_filter_count);
        //get_bias_gradient().resize(m_filter_count);

        const float std_dev_init = 1.0f/std::sqrt(static_cast<float>(m_image_depth*m_conv_filter_height*m_conv_filter_width)); // default
        randomize_uniform(W.data, -std_dev_init, std_dev_init); // default
        //randomize_uniform(m_bias, -std_dev_init, std_dev_init);
        //const float std_dev_init = 2.0f*std::sqrt(2.0f)/std::sqrt(static_cast<float>(m_image_depth*m_conv_filter_height*m_conv_filter_width));
        //randomize_normal(m_W, 0.0f, std_dev_init);

        m_W_fixed_random = W.data;
        if (VERBOSE_MODE) {
            std::cout << indent << "Initialized weights with std dev = " << std_dev_init << std::endl;
        }
    }
    m_temp_Z2.resize(m_image_height*m_image_width*m_minibatch_size, m_filter_count);
    m_temp_A1.resize(m_image_height*m_image_width*m_minibatch_size, m_image_depth*m_conv_filter_height*m_conv_filter_width + 1);
    m_temp_W.resize(m_image_depth*m_conv_filter_height*m_conv_filter_width + 1, m_filter_count);
    if (VERBOSE_MODE) {
        std::cout << indent << "Image height = " << m_image_height << std::endl;
        std::cout << indent << "Image width = " << m_image_width << std::endl;
        std::cout << indent << "Input image channels = " << m_image_depth << std::endl;
        std::cout << indent << "Output image channels = " << m_filter_count << std::endl;
    }
}



void ConvLayer2D::forward_propagate(const MatrixF& input_activations) {
    auto& W = get_param("W");
    auto& bias = get_param("bias");
    //std::cout << get_name() << ": W: " << W.data << std::endl;
    if (!m_enable_bias) {
        set_value(bias.data, 0.0f);
    }
    // perform convolution on one mini-batch.
    // Compute m_Z = m_W (convolve with) m_input_activations + m_bias:

    // naive version
    //convolve_3d_filter_with_bias_minibatch(m_output_forward, get_weights(), get_bias(), input_activations);

    // optimized version
    convolve_3d_filter_with_bias_minibatch_optimized(m_output_data, W.data, bias.data, input_activations, m_temp_Z2, m_temp_A1, m_temp_W);

}

void ConvLayer2D::back_propagate_paramater_gradients(const MatrixF& input_activations) {
    auto& W = get_param("W");
    auto& bias = get_param("bias");
    // naive version
    //compute_3d_weight_grad_convolutive_minibatch(get_weight_gradient(), m_output_backward, input_activations);

    // optimized version
    compute_3d_weight_grad_convolutive_minibatch_optimized(W.grad, m_output_grad, input_activations, m_temp_Z2, m_temp_A1, m_temp_W);

    compute_bias_grad_convolutive_minibatch(bias.grad, m_output_grad);
}




void ConvLayer2D::back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) {
    // Update m_input_backward.
    if (m_use_fixed_random_back_prop) {
        // naive version
        //compute_3d_convolutive_deltas_minibatch(input_backward, m_W_fixed_random, m_output_backward);

        // optimized version
        compute_3d_convolutive_deltas_minibatch_optimized(input_backward, m_W_fixed_random, m_output_grad, m_temp_Z2, m_temp_A1, m_temp_W);
    } else {
        auto& W = get_param("W");
        // naive version
        //compute_3d_convolutive_deltas_minibatch(input_backward, get_weights(), m_output_backward);

        // optimized version
        compute_3d_convolutive_deltas_minibatch_optimized(input_backward, W.data, m_output_grad, m_temp_Z2, m_temp_A1, m_temp_W);
    }
}

}
