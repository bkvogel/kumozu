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
#include "BatchNormalization1D.h"
#include "Utilities.h"

using namespace std;

namespace kumozu {

void BatchNormalization1D::reinitialize(std::vector<int> input_extents) {
    string indent = "    ";
    m_dim_input = input_extents.at(0);
    m_minibatch_size = input_extents.at(1);
    m_output_var.resize(input_extents);
    //m_output_grad.resize(input_extents);
    m_input_sized_normalize_offset.resize(input_extents);
    m_input_sized_normalize_scale.resize(input_extents);
    m_temp_size_input.resize(input_extents);
    m_W_expanded.resize(input_extents);
    m_bias_expanded.resize(input_extents);
    m_x_hat.resize(input_extents);
    m_centered_input.resize(input_extents);
    m_temp_size_input_1d.resize(m_dim_input);
    auto& W = get_param("W");
    auto& bias = get_param("bias");
    if (m_enable_gamma_beta) {
        const std::vector<int> new_W_extents = {m_dim_input};
        if (new_W_extents != W.get_extents()) {
            W.resize(m_dim_input); // This is "gamma" in the paper.
            bias.resize(m_dim_input); // this is "beta" in the paper.
            //const float std_dev_init = 1.0f;
            //randomize_uniform(get_weights(), 0.0f, std_dev_init);
            //randomize_uniform(get_weights(), 0.99f, 1.01f);

            //randomize_uniform(get_bias(), -std_dev_init, std_dev_init);
            //cout << indent << "weights std dev = " << std_dev_init << endl;
            set_value(W.data, 1.0f);
            set_value(bias.data, 0.0f);
        }
    }
    m_mean_cur_batch.resize(m_dim_input);
    m_mean_to_use.resize(m_dim_input);
    m_mean_running_avg.resize(m_dim_input);
    m_var_cur_batch.resize(m_dim_input);
    m_var_to_use.resize(m_dim_input);
    m_var_running_avg.resize(m_dim_input);
    //set_value(m_var_running_avg, 1.0f);
    m_var_deltas.resize(m_var_running_avg.get_extents());
    m_mean_deltas.resize(m_mean_running_avg.get_extents());
    m_xhat_deltas.resize(input_extents);
}




void BatchNormalization1D::forward_propagate(const MatrixF& input_activations) {



    // If we are in training mode, update both the batch estimates and the running
    // average estimates of mean and variacne:
    if (is_train_mode()) {
        //cout << "bn: train" << endl;
        // Compute mini-batch mean:
        const float minibatch_scale = 1/static_cast<float>(m_minibatch_size);
#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            float sum = 0.0f;
            for (int j=0; j < m_minibatch_size; ++j) {
                sum += input_activations(i,j);
            }
            m_mean_cur_batch(i) = sum*minibatch_scale;
        }

        if (m_is_first_batch) {
            //cout << "1st batch" << endl;
            // Initialize running average to the mini-batch statistics.
            m_mean_running_avg = m_mean_cur_batch;
        }

        // Running average mean:
        map2(m_mean_running_avg, m_mean_running_avg, m_mean_cur_batch, [=](float old_part, float new_part) {
            return old_part*(1.0f - m_momentum) + new_part*m_momentum;
        });

        // Expand mean to same size as mini-batch and negate:
#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            for (int j=0; j < m_minibatch_size; ++j) {
                m_temp_size_input(i,j) = -m_mean_cur_batch(i);
            }
        }
        // Compute mini-batch variance:
        element_wise_sum(m_centered_input, input_activations, m_temp_size_input);
        element_wise_square(m_temp_size_input, m_centered_input);

#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < m_minibatch_size; ++j) {
                sum += m_temp_size_input(i, j);
            }
            m_var_cur_batch(i) = sum;
        }
        // variance for current mini-batch.
        scale(m_var_cur_batch, m_var_cur_batch, minibatch_scale);

        if (m_is_first_batch) {
            // Initialize running average to the mini-batch statistics.
            m_var_running_avg = m_var_cur_batch;
        }

        // Running average variance:
        map2(m_var_running_avg, m_var_running_avg, m_var_cur_batch, [=](float old_part, float new_part) {
            return old_part*(1.0f - m_momentum) + new_part*m_momentum;
        });

        m_mean_to_use = m_mean_cur_batch;
        m_var_to_use = m_var_cur_batch;
    } else {
        //cout << "bn: TEST" << endl;
        assertion(!m_is_first_batch, "This layer must be called in train mode at least once before it is called in test mode.");
        m_mean_to_use = m_mean_running_avg;
        m_var_to_use = m_var_running_avg;

        //m_mean_to_use = m_mean_cur_batch;
        //m_var_to_use = m_var_cur_batch;
        //cout << "bn 1D: test mode:" << endl;
        //cout << "running avg var:" << endl;
        //cout << m_var_running_avg << endl;
        //cout << "batch var: " << endl;
        //cout << m_var_cur_batch << endl;
        //cout << "---------------" << endl;

        //cout << "bn 1D: test mode:" << endl;
        //cout << "running avg mean:" << endl;
        //cout << m_mean_running_avg << endl;
        //cout << "batch mean: " << endl;
        //cout << m_mean_cur_batch << endl;
        //cout << "---------------" << endl;
    }

    // Center the input:
    // Expand mean to same size as mini-batch and negate:
#pragma omp parallel for
    for (int i = 0; i < m_dim_input; ++i) {
        for (int j=0; j < m_minibatch_size; ++j) {
            m_temp_size_input(i,j) = -m_mean_to_use(i);
        }
    }
    element_wise_sum(m_centered_input, input_activations, m_temp_size_input);


    map1(m_temp_size_input_1d, m_var_to_use, [=] (float var) {
        return 1/std::sqrt(var + m_normalization_epsilon);
    });
    // Expand to same size as mini-batch
#pragma omp parallel for
    for (int i = 0; i < m_dim_input; ++i) {
        for (int j = 0; j < m_minibatch_size; ++j) {
            m_input_sized_normalize_scale(i, j) = m_temp_size_input_1d(i);
        }
    }

    m_x_hat = m_centered_input;
    element_wise_multiply(m_x_hat, m_x_hat, m_input_sized_normalize_scale);
    m_output_var.data = m_x_hat;
    if (m_enable_gamma_beta) {
        // Perform the "scale and shift" using learned gamma and beta parameters.

        //MatrixF& W = get_weights();
        // Expand W to same size as input mini-batch:
        auto& W = get_param("W");
#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            for (int j = 0; j < m_minibatch_size; ++j) {
                m_W_expanded(i, j) = W.data(i); // fixme
            }
        }

        // Perform scale: Note: W is gamma in the paper.
        element_wise_multiply(m_output_var.data, m_output_var.data, m_W_expanded);
        //const MatrixF& bias = get_bias();
        // Expand bias to same size as input mini-batch:
        auto& bias = get_param("bias");
#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            for (int j = 0; j < m_minibatch_size; ++j) {
                m_bias_expanded(i, j) = bias.data(i);
            }
        }

        // Perform shift: Note: m_bias is beta in the paper.
        element_wise_sum(m_output_var.data, m_output_var.data, m_bias_expanded);
    }

    m_is_first_batch = false;
}

void BatchNormalization1D::back_propagate_paramater_gradients(const MatrixF& input_activations) {
    //MatrixF& W_grad = get_weight_gradient();
    auto& W = get_param("W");
    if (m_enable_gamma_beta) {
        // Update gamma:
        auto& output_grad = m_output_var.grad;
#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < m_minibatch_size; ++j) {
                sum += output_grad(i, j)*m_x_hat(i, j);
            }
            W.grad(i) = sum;
        }

        // Update beta:
        //MatrixF& bias_grad = get_bias_gradient();
        auto& bias = get_param("bias");
#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < m_minibatch_size; ++j) {
                sum += output_grad(i, j);
            }
            bias.grad(i) = sum;
        }
    }

}


void BatchNormalization1D::back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) {
    //MatrixF& W = get_weights();
    // Compute del loss/ del x hat
    auto& W = get_param("W");
    if (m_enable_gamma_beta) {
        // Expand W to same size as input mini-batch:
#pragma omp parallel for
        for (int i = 0; i < m_dim_input; ++i) {
            for (int j = 0; j < m_minibatch_size; ++j) {
                m_W_expanded(i, j) = W.data(i);
            }
        }
        element_wise_multiply(m_xhat_deltas, m_output_var.grad, m_W_expanded);
    } else {
        //copy_matrix(m_xhat_deltas, m_output_backward);
        m_xhat_deltas = m_output_var.grad;
    }
    map1(m_temp_size_input_1d, m_var_to_use, [=] (float var) {
        return -0.5f*std::pow(var + m_normalization_epsilon, -1.5f);
    });

    // Compute del loss/del variance
#pragma omp parallel for
    for (int i = 0; i < m_dim_input; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < m_minibatch_size; ++j) {
            sum += m_xhat_deltas(i, j)*m_centered_input(i,j)*m_temp_size_input_1d(i);
        }
        m_var_deltas(i) = sum;
    }

    // Compute del loss/del mean
    // part 1 (before plus sign in paper):
#pragma omp parallel for
    for (int i = 0; i < m_dim_input; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < m_minibatch_size; ++j) {
            sum -= m_xhat_deltas(i, j)*m_input_sized_normalize_scale(i,j);
        }
        m_mean_deltas(i) = sum;
    }
    const float minibatch_scale = 1/static_cast<float>(m_minibatch_size);
    // part 2 (after plus sign in paper):
#pragma omp parallel for
    for (int i = 0; i < m_dim_input; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < m_minibatch_size; ++j) {
            sum -= 2.0f*minibatch_scale*m_centered_input(i,j);
        }
        m_temp_size_input_1d(i) = sum;
    }
    element_wise_multiply(m_temp_size_input_1d, m_temp_size_input_1d, m_var_deltas);
    element_wise_sum(m_mean_deltas, m_mean_deltas, m_temp_size_input_1d);

    // part 1 of 3 for input deltas (before first plus sign in paper):
    //copy_matrix(input_backward, m_xhat_deltas);
    input_backward = m_xhat_deltas;
    element_wise_multiply(input_backward, input_backward, m_input_sized_normalize_scale);

    // part 2 and 3 of 3 for input deltas (aftere first plus sign in paper):
#pragma omp parallel for
    for (int i = 0; i < m_dim_input; ++i) {
        for (int j = 0; j < m_minibatch_size; ++j) {
            const float second_term = m_var_deltas(i)*2.0f*minibatch_scale*m_centered_input(i,j);
            const float third_term = m_mean_deltas(i)*minibatch_scale;
            input_backward(i,j) += second_term + third_term;
        }
    }


}




}
