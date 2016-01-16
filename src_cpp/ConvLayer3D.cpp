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
#include "ConvLayer3D.h"
#include "Utilities.h"
#include "MatrixIO.h"

using namespace std;

namespace kumozu {

  void ConvLayer3D::reinitialize(std::vector<int> input_extents) {
    string indent = "    ";
    m_input_extents = input_extents;
    m_minibatch_size = input_extents.at(0);
    m_image_depth = input_extents.at(1);
    m_image_height = input_extents.at(2);
    m_image_width = input_extents.at(3);
    m_output_activations = MatrixF(m_minibatch_size, m_filter_count, m_image_height, m_image_width);
    m_output_error = MatrixF(m_minibatch_size, m_filter_count, m_image_height, m_image_width);
    const std::vector<int> new_W_extents = {m_filter_count, m_image_depth, m_conv_filter_height, m_conv_filter_width};
    if (new_W_extents != m_W.get_extents()) {
      m_W = MatrixF(new_W_extents);
      m_temp_size_W = MatrixF(new_W_extents);
      m_W_grad = MatrixF(new_W_extents);
      m_bias = MatrixF(m_filter_count);
      m_temp_size_bias = MatrixF(m_filter_count);
      m_bias_grad = MatrixF(m_filter_count);

      const float std_dev_init = 1.0f/std::sqrt(static_cast<float>(m_image_depth*m_conv_filter_height*m_conv_filter_width)); // default
      randomize_uniform(m_W, -std_dev_init, std_dev_init); // default
      //randomize_uniform(m_bias, -std_dev_init, std_dev_init);
      //const float std_dev_init = 2.0f*std::sqrt(2.0f)/std::sqrt(static_cast<float>(m_image_depth*m_conv_filter_height*m_conv_filter_width));
      //randomize_normal(m_W, 0.0f, std_dev_init);

      m_W_fixed_random = m_W;
      std::cout << indent << "Initialized weights with std dev = " << std_dev_init << std::endl;
    }
    m_temp_Z2 = MatrixF(m_image_height*m_image_width*m_minibatch_size, m_filter_count);
    m_temp_A1 = MatrixF(m_image_height*m_image_width*m_minibatch_size, m_image_depth*m_conv_filter_height*m_conv_filter_width + 1);
    m_temp_W = MatrixF(m_image_depth*m_conv_filter_height*m_conv_filter_width + 1, m_filter_count);

    std::cout << indent << "Image height = " << m_image_height << std::endl;
    std::cout << indent << "Image width = " << m_image_width << std::endl;
    std::cout << indent << "Image depth = " << m_image_depth << std::endl;
    std::cout << indent << "Number of convolutional filters = " << m_filter_count << std::endl;

  }



  void ConvLayer3D::forward_propagate(const MatrixF& input_activations) {
    if (!m_enable_bias) {
      set_value(m_bias, 0.0f);
    }
    // perform convolution on one mini-batch.
    // Compute m_Z = m_W (convolve with) m_input_activations + m_bias:

    // naive version
    //convolve_3d_filter_with_bias_minibatch(m_output_activations, m_W, m_bias, input_activations);

    // optimized version
    convolve_3d_filter_with_bias_minibatch_optimized(m_output_activations, m_W, m_bias, input_activations, m_temp_Z2, m_temp_A1, m_temp_W);
  }

  void ConvLayer3D::back_propagate_paramater_gradients(const MatrixF& input_activations) {
    // naive version
    //compute_3d_weight_grad_convolutive_minibatch(m_W_grad, m_output_error, input_activations);

    // optimized version
    compute_3d_weight_grad_convolutive_minibatch_optimized(m_W_grad, m_output_error, input_activations, m_temp_Z2, m_temp_A1, m_temp_W);

    compute_bias_grad_convolutive_minibatch(m_bias_grad, m_output_error);
  }




  void ConvLayer3D::back_propagate_deltas(MatrixF& input_error) {
    // Update m_input_error.
    if (m_use_fixed_random_back_prop) {
      // naive version
      //compute_3d_convolutive_deltas_minibatch(input_error, m_W_fixed_random, m_output_error);

      // optimized version
      compute_3d_convolutive_deltas_minibatch_optimized(input_error, m_W_fixed_random, m_output_error, m_temp_Z2, m_temp_A1, m_temp_W);
    } else {
      // naive version
      //compute_3d_convolutive_deltas_minibatch(input_error, m_W, m_output_error);

      // optimized version
      compute_3d_convolutive_deltas_minibatch_optimized(input_error, m_W, m_output_error, m_temp_Z2, m_temp_A1, m_temp_W);
    }
  }





}
