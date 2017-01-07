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
    cout << get_name() << endl;
    m_minibatch_size = input_extents.at(0);
    //cout << "BatchNormalization3D: reinitialize() called. minibatch size = " << m_minibatch_size << endl;
    m_image_depth = input_extents.at(1);
    m_image_height = input_extents.at(2);
    m_image_width = input_extents.at(3);
    m_output_var.resize(input_extents);
    //m_output_grad.resize(input_extents);
    m_input_sized_normalize_offset.resize(input_extents);
    m_input_sized_normalize_scale.resize(input_extents);
    m_temp_size_input.resize(input_extents);
    m_W_expanded.resize(input_extents);
    m_bias_expanded.resize(input_extents);
    m_x_hat.resize(input_extents);
    m_centered_input.resize(input_extents);
    m_temp_size_input_1d.resize(m_image_depth);
    auto& W = get_param("W");
    auto& bias = get_param("bias");
    if (m_enable_gamma_beta) {
        const std::vector<int> new_W_extents = {m_image_depth};
        //if (new_W_extents != m_W.get_extents()) {
        if (new_W_extents != W.get_extents()) {
            W.resize(m_image_depth); // This is "gamma" in the paper.
            bias.resize(m_image_depth); // this is "beta" in the paper.
            //const float std_dev_init = 1.0f;
            //randomize_uniform(get_weights(), 0.0f, std_dev_init);
            //randomize_uniform(get_weights(), 0.99f, 1.01f);
            //randomize_uniform(get_bias(), -std_dev_init, std_dev_init);
            //cout << indent << "weights std dev = " << std_dev_init << endl;
            set_value(W.data, 1.0f);
            set_value(bias.data, 0.0f);
        }
    }
    m_mean_cur_batch.resize(m_image_depth);
    m_mean_to_use.resize(m_image_depth);
    m_mean_running_avg.resize(m_image_depth);
    m_var_cur_batch.resize(m_image_depth);
    m_var_to_use.resize(m_image_depth);
    m_var_running_avg.resize(m_image_depth);
    //if (m_is_first_batch) {
    //set_value(m_var_running_avg, 1.0f); // fixme 1
    //}
    m_var_deltas.resize(m_var_running_avg.get_extents());
    m_mean_deltas.resize(m_mean_running_avg.get_extents());
    m_xhat_deltas.resize(input_extents);
}




void BatchNormalization3D::forward_propagate(const MatrixF& input_activations) {
    // If we are in training mode, update both the batch estimates and the running
    // average estimates of mean and variacne:
    //assertion(m_var_running_avg(0) != 1, "oopsy1");
    if (is_train_mode()) {
        //cout << "bn3D: train" << endl;
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

        if (m_is_first_batch) {
            //cout << "bn3D: 1st batch" << endl;
            // Initialize running average to the mini-batch statistics.
            m_mean_running_avg = m_mean_cur_batch;
        }

        // Running average mean:
        map2(m_mean_running_avg, m_mean_running_avg, m_mean_cur_batch, [=](float old_avg, float new_avg) {
            return old_avg*(1.0f - m_momentum) + new_avg*m_momentum;
        });

        // Expand mean to same size as mini-batch and negate:
#pragma omp parallel for
        for (int i = 0; i < m_image_depth; ++i) {
            for (int j=0; j < m_minibatch_size; ++j) {
                for (int k = 0; k < m_image_height; ++k) {
                    for (int l = 0; l < m_image_width; ++l) {
                        m_temp_size_input(j,i,k,l) = -m_mean_cur_batch(i);
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

        // variance for current mini-batch.
        scale(m_var_cur_batch, m_var_cur_batch, minibatch_scale);

        if (m_is_first_batch) {
            // Initialize running average to the mini-batch statistics.
            m_var_running_avg = m_var_cur_batch;
        }

        // Running average variance:
        map2(m_var_running_avg, m_var_running_avg, m_var_cur_batch, [=](float old_avg, float new_avg) {
            return old_avg*(1.0f - m_momentum) + new_avg*m_momentum;
        });

        m_mean_to_use = m_mean_cur_batch;
        m_var_to_use = m_var_cur_batch;
        //cout << "bn3D, train, m_var_to_use:" << endl;
        //cout << m_var_to_use << endl;
        //cout << "bn3D, train, m_var_running_avg:" << endl;
        //cout << m_var_running_avg << endl;
    } else {
        //cout << "bn3D: TEST" << endl;
        assertion(!m_is_first_batch, "This layer must be called in train mode at least once before it is called in test mode.");
        m_mean_to_use = m_mean_running_avg;
        m_var_to_use = m_var_running_avg;



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

    //assertion(m_var_running_avg(0) != 1, "oopsy");

    // Center the input:
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


    map1(m_temp_size_input_1d, m_var_to_use, [=] (float var) {
        return 1/std::sqrt(var + m_normalization_epsilon);
    });
    //assertion(m_var_running_avg(0) != 1, "oopsy2");
    // Expand to same size as mini-batch
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

    m_x_hat = m_centered_input;
    element_wise_multiply(m_x_hat, m_x_hat, m_input_sized_normalize_scale);
    //copy_matrix(m_output_forward, m_x_hat);
    m_output_var.data = m_x_hat;
    if (m_enable_gamma_beta) {
        // Perform the "scale and shift" using learned gamma and beta parameters.

        //MatrixF& W = get_weights();

        // Expand W to same size as input mini-batch:
        auto& W = get_param("W");
#pragma omp parallel for
        for (int i = 0; i < m_image_depth; ++i) {
            for (int j = 0; j < m_minibatch_size; ++j) {
                for (int k = 0; k < m_image_height; ++k) {
                    for (int l = 0; l < m_image_width; ++l) {
                        m_W_expanded(j,i,k,l) = W.data(i);
                    }
                }
            }
        }
        //assertion(m_var_running_avg(0) != 1, "oopsy3");
        // Perform scale: Note: W is gamma in the paper.
        element_wise_multiply(m_output_var.data, m_output_var.data, m_W_expanded);

        // Expand bias to same size as input mini-batch:
        //const MatrixF& bias = get_bias();
        auto& bias = get_param("bias");
#pragma omp parallel for
        for (int i = 0; i < m_image_depth; ++i) {
            for (int j = 0; j < m_minibatch_size; ++j) {
                for (int k = 0; k < m_image_height; ++k) {
                    for (int l = 0; l < m_image_width; ++l) {
                        m_bias_expanded(j,i,k,l) = bias.data(i);
                    }
                }
            }
        }

        // Perform shift: Note: m_bias is beta in the paper.
        element_wise_sum(m_output_var.data, m_output_var.data, m_bias_expanded);
    }

    m_is_first_batch = false;
    //assertion(m_var_running_avg(0) != 1, "oopsy4");
}

void BatchNormalization3D::back_propagate_paramater_gradients(const MatrixF& input_activations) {
    //MatrixF& W_grad = get_weight_gradient();

    if (m_enable_gamma_beta) {
        auto& W = get_param("W");
        auto& output_grad = m_output_var.grad;
        // Update gamma:
#pragma omp parallel for
        for (int i = 0; i < m_image_depth; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < m_minibatch_size; ++j) {
                for (int k = 0; k < m_image_height; ++k) {
                    for (int l = 0; l < m_image_width; ++l) {
                        sum += output_grad(j,i,k,l)*m_x_hat(j,i,k,l);
                    }
                }
            }
            W.grad(i) = sum;
        }

        // Update beta:
        //MatrixF& bias_grad = get_bias_gradient();
        auto& bias = get_param("bias");
#pragma omp parallel for
        for (int i = 0; i < m_image_depth; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < m_minibatch_size; ++j) {
                for (int k = 0; k < m_image_height; ++k) {
                    for (int l = 0; l < m_image_width; ++l) {
                        sum += output_grad(j,i,k,l);
                    }
                }
            }
            bias.grad(i) = sum;
        }
    }
}


void BatchNormalization3D::back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) {
    //MatrixF& W = get_weights();
    // Compute del loss/ del x hat
    if (m_enable_gamma_beta) {
        auto& W = get_param("W");
        // Expand W to same size as input mini-batch:
#pragma omp parallel for
        for (int i = 0; i < m_image_depth; ++i) {
            for (int j = 0; j < m_minibatch_size; ++j) {
                for (int k = 0; k < m_image_height; ++k) {
                    for (int l = 0; l < m_image_width; ++l) {
                        m_W_expanded(j,i,k,l) = W.data(i);
                    }
                }
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
    //copy_matrix(input_backward, m_xhat_deltas);
    input_backward = m_xhat_deltas;
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
