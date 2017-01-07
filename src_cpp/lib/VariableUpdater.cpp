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

#include "VariableUpdater.h"
#include "Utilities.h"

namespace kumozu {

void VariableUpdater::update(MatrixF& W, const MatrixF& grad_W) {
    if (0 == m_current_mode) {
        // Constant learning rate
        update_parameters_sgd(W, grad_W, m_learning_rate);
    } else if (1 == m_current_mode) {
        // Constant learning rate.
        //update_weights_from_gradient(W, grad_W, m_learning_rate, m_weight_decay);
    } else if (2 == m_current_mode) {
        // Rmsprop
        update_parameters_rmsprop(W, grad_W, m_sum_square_grad_W, m_rmsprop_learning_rate, m_rho);
    } else if (3 == m_current_mode) {
        // Rmsprop with momentum
        update_parameters_rmsprop_momentum(W, grad_W, m_sum_square_grad_W, m_momentum_W, m_rmsprop_learning_rate, m_rho, m_momentum);
    }

    // flags:
    if (m_force_nonnegative) {
        threshold_lower(W, 0.0f); // force nonnegative.
    }
    if (m_enable_weight_decay) {
        // Enable weight/activation decay.
        update_parameters_from_decay(W, m_weight_decay);
    }
}

void VariableUpdater::update(VariableF& var) {
    update(var.data, var.grad);
}

void VariableUpdater::set_mode_constant_learning_rate(float learning_rate) {
    m_current_mode = 0;
    m_learning_rate = learning_rate;
}



void VariableUpdater::set_mode_rmsprop(float rmsprop_learning_rate, float rho) {
    m_current_mode = 2;
    m_rmsprop_learning_rate = rmsprop_learning_rate;
    m_rho = rho;
}

void VariableUpdater::set_mode_rmsprop_momentum(float rmsprop_learning_rate, float rho, float momentum) {
    m_current_mode = 3;
    m_rmsprop_learning_rate = rmsprop_learning_rate;
    m_rho = rho;
    m_momentum = momentum;
}

void VariableUpdater::set_flag_force_nonnegative(bool force_nonnegative) {
    if (force_nonnegative) {
        m_force_nonnegative = true;
    } else {
        m_force_nonnegative = false;
    }
}

void VariableUpdater::set_flag_weight_decay(float decay_val, bool enable_weight_decay) {
    m_weight_decay = decay_val;
    m_enable_weight_decay = enable_weight_decay;
}



}
