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
#include "BatchNormalization3D.h"
#include "Utilities.h"

using namespace std;

namespace kumozu {

  void BatchNormalization3D::reinitialize(std::vector<int> input_extents) {
    // Input extents should be: (minibatch_size, image_depth, image_height, image_width)
    string indent = "    ";
    m_minibatch_size = input_extents.at(0);
    m_image_depth = input_extents.at(1);
    m_image_height = input_extents.at(2);
    m_image_width = input_extents.at(3);
    m_output_forward.resize(input_extents);
    m_output_backward.resize(input_extents);
    m_input_sized_normalize_offset.resize(input_extents);
    m_input_sized_normalize_scale.resize(input_extents);
    m_temp_size_input.resize(input_extents);
    m_W_expanded.resize(input_extents);
    m_bias_expanded.resize(input_extents);
    m_x_hat.resize(input_extents);
    m_centered_input.resize(input_extents);
    m_temp_size_input_1d.resize(m_image_depth);
    if (m_enable_gamma_beta) {
      const std::vector<int> new_W_extents = {m_image_depth};
      //if (new_W_extents != m_W.get_extents()) {
      if (new_W_extents != get_weights().get_extents()) {
        get_weights().resize(m_image_depth); // This is "gamma" in the paper.
        get_weight_gradient().resize(m_image_depth);
        get_bias().resize(m_image_depth); // this is "beta" in the paper.
        get_bias_gradient().resize(m_image_depth);
        const float std_dev_init = 1.0f;
        randomize_uniform(get_weights(), -std_dev_init, std_dev_init);
        randomize_uniform(get_bias(), -std_dev_init, std_dev_init);
        cout << indent << "weights std dev = " << std_dev_init << endl;
      }
    }
    m_mean_cur_batch.resize(m_image_depth);
    m_mean_to_use.resize(m_image_depth);
    m_mean_running_avg.resize(m_image_depth);
    m_var_cur_batch.resize(m_image_depth);
    m_var_to_use.resize(m_image_depth);
    m_var_running_avg.resize(m_image_depth);
    set_value(m_var_running_avg, 1.0f);
    m_var_deltas.resize(m_var_running_avg.get_extents());
    m_mean_deltas.resize(m_mean_running_avg.get_extents());
    m_xhat_deltas.resize(input_extents);
  }




  void BatchNormalization3D::forward_propagate(const MatrixF& input_activations) {
    // Compute mini-batch mean:
    const float minibatch_scale = 1/static_cast<float>(m_minibatch_size*m_image_height*m_image_width);
#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      float sum = 0.0f;
      for (int j=0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            sum += input_activations(j,i,k,l);
          }
        }
      }
      m_mean_cur_batch(i) = sum*minibatch_scale;
    }

    //cout << "m_mean_cur_batch = " << endl << m_mean_cur_batch << endl;

    // Compute running average mean:
    if (is_train_mode()) {
      map2(m_mean_running_avg, m_mean_running_avg, m_mean_cur_batch, [=](float old_avg, float new_avg) {
          return old_avg*(1 - m_momentum) + new_avg*m_momentum;
        });
      copy_matrix(m_mean_to_use, m_mean_cur_batch);
    } else {
      // test mode
      if (m_bypass_running_average) {
        copy_matrix(m_mean_to_use, m_mean_cur_batch);
      } else {
        copy_matrix(m_mean_to_use, m_mean_running_avg);
      }
    }

    // Expand mean to same size as mini-batch and negate:
#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      for (int j=0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            m_temp_size_input(j,i,k,l) = -m_mean_to_use(i);
          }
        }
      }
    }

    // Compute mini-batch variance:

    element_wise_sum(m_centered_input, input_activations, m_temp_size_input);
    element_wise_square(m_temp_size_input, m_centered_input);

#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            sum += m_temp_size_input(j,i,k,l);
          }
        }
      }
      m_var_cur_batch(i) = sum;
    }

    scale(m_var_cur_batch, m_var_cur_batch, minibatch_scale); // variance for current mini-batch.

    if (is_train_mode()) {
      map2(m_var_running_avg, m_var_running_avg, m_var_cur_batch, [=](float old_avg, float new_avg) {
          return old_avg*(1 - m_momentum) + new_avg*m_momentum;
        });
      copy_matrix(m_var_to_use, m_var_cur_batch);
    } else {
      // test mode
      if (m_bypass_running_average) {
        copy_matrix(m_var_to_use, m_var_cur_batch);
      } else {
        copy_matrix(m_var_to_use, m_var_running_avg);
      }
    }

    map1(m_temp_size_input_1d, m_var_to_use, [=] (float var) {
        return 1/std::sqrt(var + m_normalization_epsilon);
      });

    // Expand variance to same size as mini-batch
#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      for (int j = 0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            m_input_sized_normalize_scale(j,i,k,l) = m_temp_size_input_1d(i);
          }
        }
      }
    }

    copy_matrix(m_x_hat, m_centered_input);
    element_wise_multiply(m_x_hat, m_x_hat, m_input_sized_normalize_scale);
    copy_matrix(m_output_forward, m_x_hat);
    if (m_enable_gamma_beta) {
      // Perform the "scale and shift" using learned gamma and beta parameters.

      MatrixF& W = get_weights();

      // Expand W to same size as input mini-batch:
#pragma omp parallel for
      for (int i = 0; i < m_image_depth; ++i) {
        for (int j = 0; j < m_minibatch_size; ++j) {
          for (int k = 0; k < m_image_height; ++k) {
            for (int l = 0; l < m_image_width; ++l) {
              m_W_expanded(j,i,k,l) = W(i);
            }
          }
        }
      }

      // Perform scale: Note: W is gamma in the paper.
      element_wise_multiply(m_output_forward, m_output_forward, m_W_expanded);

      // Expand bias to same size as input mini-batch:
      const MatrixF& bias = get_bias();
#pragma omp parallel for
      for (int i = 0; i < m_image_depth; ++i) {
        for (int j = 0; j < m_minibatch_size; ++j) {
          for (int k = 0; k < m_image_height; ++k) {
            for (int l = 0; l < m_image_width; ++l) {
              m_bias_expanded(j,i,k,l) = bias(i);
            }
          }
        }
      }

      // Perform shift: Note: m_bias is beta in the paper.
      element_wise_sum(m_output_forward, m_output_forward, m_bias_expanded);
    }


  }

  void BatchNormalization3D::back_propagate_paramater_gradients(const MatrixF& input_activations) {
    MatrixF& W_grad = get_weight_gradient();
    if (m_enable_gamma_beta) {
      // Update gamma:
#pragma omp parallel for
      for (int i = 0; i < m_image_depth; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < m_minibatch_size; ++j) {
          for (int k = 0; k < m_image_height; ++k) {
            for (int l = 0; l < m_image_width; ++l) {
              sum += m_output_backward(j,i,k,l)*m_x_hat(j,i,k,l);
            }
          }
        }
        W_grad(i) = sum;
      }

      // Update beta:
      MatrixF& bias_grad = get_bias_gradient();
#pragma omp parallel for
      for (int i = 0; i < m_image_depth; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < m_minibatch_size; ++j) {
          for (int k = 0; k < m_image_height; ++k) {
            for (int l = 0; l < m_image_width; ++l) {
              sum += m_output_backward(j,i,k,l);
            }
          }
        }
        bias_grad(i) = sum;
      }
    }
  }


  void BatchNormalization3D::back_propagate_deltas(MatrixF& input_backward, const MatrixF& input_forward) {
    MatrixF& W = get_weights();
    // Compute del loss/ del x hat
    if (m_enable_gamma_beta) {
      // Expand W to same size as input mini-batch:
#pragma omp parallel for
      for (int i = 0; i < m_image_depth; ++i) {
        for (int j = 0; j < m_minibatch_size; ++j) {
          for (int k = 0; k < m_image_height; ++k) {
            for (int l = 0; l < m_image_width; ++l) {
              m_W_expanded(j,i,k,l) = W(i);
            }
          }
        }
      }
      element_wise_multiply(m_xhat_deltas, m_output_backward, m_W_expanded);
    } else {
      copy_matrix(m_xhat_deltas, m_output_backward);
    }
    map1(m_temp_size_input_1d, m_var_to_use, [=] (float var) {
        return -0.5f*std::pow(var + m_normalization_epsilon, -1.5f);
      });

    // Compute del loss/del variance
#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            sum += m_xhat_deltas(j,i,k,l)*m_centered_input(j,i,k,l)*m_temp_size_input_1d(i);
          }
        }
      }
      m_var_deltas(i) = sum;
    }

    // Compute del loss/del mean
    // part 1 (before plus sign in paper):
#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            sum -= m_xhat_deltas(j,i,k,l)*m_input_sized_normalize_scale(j,i,k,l);
          }
        }
      }
      m_mean_deltas(i) = sum;
    }
    const float minibatch_scale = 1/static_cast<float>(m_minibatch_size*m_image_height*m_image_width);
    //const float minibatch_scale = 1/static_cast<float>(m_minibatch_size);
    // part 2 (after plus sign in paper):
#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            sum -= 2.0f*minibatch_scale*m_centered_input(j,i,k,l);
          }
        }
      }
      m_temp_size_input_1d(i) = sum;
    }
    element_wise_multiply(m_temp_size_input_1d, m_temp_size_input_1d, m_var_deltas);
    element_wise_sum(m_mean_deltas, m_mean_deltas, m_temp_size_input_1d);



    // part 1 of 3 for input deltas (before first plus sign in paper):
    copy_matrix(input_backward, m_xhat_deltas);
    element_wise_multiply(input_backward, input_backward, m_input_sized_normalize_scale);

    // part 2 and 3 of 3 for input deltas (aftere first plus sign in paper):
#pragma omp parallel for
    for (int i = 0; i < m_image_depth; ++i) {
      for (int j = 0; j < m_minibatch_size; ++j) {
        for (int k = 0; k < m_image_height; ++k) {
          for (int l = 0; l < m_image_width; ++l) {
            const float second_term = m_var_deltas(i)*2.0f*minibatch_scale*m_centered_input(j,i,k,l);
            const float third_term = m_mean_deltas(i)*minibatch_scale;
            input_backward(j,i,k,l) += second_term + third_term;
          }
        }
      }
    }

  }


}
