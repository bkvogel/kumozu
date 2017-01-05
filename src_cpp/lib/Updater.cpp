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

#include "Updater.h"
#include "Utilities.h"

namespace kumozu {

void Updater::update() {

    const int list_size = m_parameters_list.size();
    if (static_cast<int>(m_momentum_list.size()) != list_size) {
        // Resize
        m_momentum_list.clear();
        m_sum_square_grad_list.clear();
        for (int i = 0; i < list_size; ++i) {
            m_momentum_list.push_back(std::make_unique<MatrixF>(m_parameters_list.at(i)->get_extents()));
            m_sum_square_grad_list.push_back(std::make_unique<MatrixF>(m_parameters_list.at(i)->get_extents()));
        }
    }
    if (m_prevent_zero_epsilon > 0) {
        if (static_cast<int>(m_prevent_zero_list.size()) != list_size) {
            m_prevent_zero_list.clear();
            for (int i = 0; i < list_size; ++i) {
                m_prevent_zero_list.push_back(std::make_unique<MatrixF>(m_parameters_list.at(i)->get_extents()));
            }
            for (int i = 0; i < list_size; ++i) {
                MatrixF& small_random = *m_prevent_zero_list.at(i);
                if (m_force_nonnegative) {
                    randomize_uniform(small_random, 0.0f, m_prevent_zero_epsilon);
                } else {
                    randomize_uniform(small_random, -m_prevent_zero_epsilon, m_prevent_zero_epsilon);
                }
            }
        } else if (m_always_new_random) {
            for (int i = 0; i < list_size; ++i) {
                MatrixF& small_random = *m_prevent_zero_list.at(i);
                if (m_force_nonnegative) {
                    randomize_uniform(small_random, 0.0f, m_prevent_zero_epsilon);
                } else {
                    randomize_uniform(small_random, -m_prevent_zero_epsilon, m_prevent_zero_epsilon);
                }
            }
        }
    }
    for (int n=0; n < list_size; ++n) {
        MatrixF& W = m_parameters_list.at(n)->data;
        const MatrixF& grad_W = m_parameters_list.at(n)->grad;
        MatrixF& sum_square_grad_W = *m_sum_square_grad_list.at(n);
        MatrixF& momentum_W = *m_momentum_list.at(n);
        if (W.size() != grad_W.size()) {
            error_exit("update(): Error: Inconsistent matrix dimensions found in list.");
        }


        if (0 == m_current_mode) {
            // Constant learning rate
            update_weights_from_gradient(W, grad_W, m_learning_rate);
        } else if (1 == m_current_mode) {
            // Constant learning rate.
            //update_weights_from_gradient(W, grad_W, m_learning_rate, m_weight_decay);
        } else if (2 == m_current_mode) {
            // Rmsprop
            update_weights_from_gradient_rmsprop_v3(W, grad_W, sum_square_grad_W, m_rmsprop_learning_rate, m_rho);
        } else if (3 == m_current_mode) {
            // Rmsprop with momentum
            update_weights_from_gradient_rmsprop_momentum(W, grad_W, sum_square_grad_W, momentum_W, m_rmsprop_learning_rate, m_rho, m_momentum);
        }

        // flags:
        if (m_force_nonnegative) {
            threshold_lower(W, 0.0f); // force nonnegative.
        }
        if (m_enable_weight_decay) {
            // Enable weight/activation decay.
            update_weights_from_decay(W, m_weight_decay);
        }
        if (m_prevent_zero_epsilon > 0) {
            MatrixF& small_random = *m_prevent_zero_list.at(n);
            if (m_force_nonnegative) {
                map2(W, W, small_random, [=] (float w, float r) {
                    if (w < m_prevent_zero_epsilon) {
                        return r;
                    } else {
                        return w;
                    }
                });
            } else {
                map2(W, W, small_random, [=] (float w, float r) {
                    if (std::abs(w) < m_prevent_zero_epsilon) {
                        return r;
                    } else {
                        return w;
                    }
                });
            }
        }
    }
}




void Updater::set_mode_constant_learning_rate(float learning_rate) {
    m_current_mode = 0;
    m_learning_rate = learning_rate;
}



void Updater::set_mode_rmsprop(float rmsprop_learning_rate, float rho) {
    m_current_mode = 2;
    m_rmsprop_learning_rate = rmsprop_learning_rate;
    m_rho = rho;
}

void Updater::set_mode_rmsprop_momentum(float rmsprop_learning_rate, float rho, float momentum) {
    m_current_mode = 3;
    m_rmsprop_learning_rate = rmsprop_learning_rate;
    m_rho = rho;
    m_momentum = momentum;
}

void Updater::set_flag_force_nonnegative(bool force_nonnegative) {
    if (force_nonnegative) {
        m_force_nonnegative = true;
    } else {
        m_force_nonnegative = false;
    }
}

void Updater::set_flag_weight_decay(float decay_val, bool enable_weight_decay) {
    m_weight_decay = decay_val;
    m_enable_weight_decay = enable_weight_decay;
}

void Updater::set_flag_prevent_zero(float epsilon, bool always_new_random) {
    m_prevent_zero_epsilon = epsilon;
    m_always_new_random = always_new_random;
}


}
