#ifndef _VARIABLE_UPDATER_H
#define _VARIABLE_UPDATER_H
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


#include "Matrix.h"
#include <string>
#include <iostream>
#include "Constants.h"
#include "Utilities.h"
#include "Variable.h"


namespace kumozu {

/**
 * fixme
   * An instance of this class is associated with a pair of matrices: W and W_grad. An "update()"
   * function is provided that can then be used to update the values in W from the gradients in W_grad.
   *
   * This class supports several update modes and options that together specify the algorithm for updating
   * W from W_grad. These modes can be changed at any time using the provided mode-setting methods.
   *
   * Usage:
   *
   * This class is generally intended to be used to update either the weights W in a layer using the corresponding
   * gradients W_grad or the activations X in a layer using the corresponding gradients (deltas). Note that
   * since the bias vector is represented using the same MatrixF structure as the weights, this class can also be
   * used to updated the bias vector b from the corresponding gradients in grad_b.
   *
   * First create a new instance of this class, supplying the size of W to the constructor.
   *
   * Then optionally call one of the mode-setting functions to configure the update algorithm.
   *
   * Then, whenver the gradients in W_grad get updated, we can then call the update() method to update
   * W.
   *
   * Implementation note: Although another option would be to have a distinct subclass for each update algorithm,
   * we choose here to make this class support all available update algorithms.
   */
// todo: move specific optimization algrithms into a subclass for each algorithm.
class VariableUpdater {

public:

    VariableUpdater() {}

    /*
     * If no "mode-setting" methods are called, update will apply a
     * default learning rate of 0.01.
     *
     * Parameters:
     *
     *
     */
    VariableUpdater(std::string name):
        m_name {name},
        m_learning_rate {0.01f}, // default learning rate.
        m_weight_decay {0.0f},
        m_force_nonnegative {false},
        m_enable_weight_decay {false},
        m_current_mode {0} // default mode.
    {

    }

    /*
     * Update the elements in the supplied MatrixF W using the gradients in
     * the supplied MatrixF grad_W.
     *
     * Depending on the current update mode, it is possible that the updated values may
     * also be influenced by the history of past gradients supplied to this method
     * up until now.
     *
     * Parameters:
     *
     * W: The matrix to be updated.
     *
     * grad_W: The matrix containing the gradients, which must be the same size as W.
     */
    // deprecated. Use the function that takes a Variable instead.
    void update(MatrixF& W, const MatrixF& grad_W);

    /**
     * Update the parameters in the variable from the gradients.
     *
     * @param var
     */
    void update(VariableF& var);


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Set update mode.
    //
    // Only one mode can be active at a time.

    /*
                   * Set a constant learning rate.
                   *
                   * This is the default mode, with default values: learning_rate = 0.01.
                   *
                   */
    void set_mode_constant_learning_rate(float learning_rate);


    void set_mode_rmsprop(float rmsprop_learning_rate, float rho);

    /*
                   * Try:
                   * rmsprop_learning_rate = 1e-2 to 1e-4
                   * rho = 0.9
                   * momentum = 0.9
                   */
    void set_mode_rmsprop_momentum(float rmsprop_learning_rate, float rho, float momentum);


    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Update flags
    //
    // These can be used with any update mode.

    /*
                   * Force all values to be nonnegative after the gradient update if the supplied parameter is true.
                   *
                   * This flag is compatible with all update modes.
                   *
                   * The default value is false so that negative values are allowed.
                   */
    void set_flag_force_nonnegative(bool force_nonnegative);

    /*
                   * Set weight/activatin decay to the specified value.
                   *
                   * The weight decay for each weight or activation w_i is set according to:
                   *
                   * w_i = w_i - decay_val*w_i
                   *
                   * The valid range for decay_val is [0, 1], but very small values such as 1e-5
                   * are often used. Note that a value of 0 leaves w_i unchanged while a value of
                   * 1 will cause w_i to be decayed to 0 in one step.
                   *
                   * Parameters:
                   *
                   * decay_val: The decay value, which must be in the range [0,1].
                   *
                   * enable_weight_decay: Set to "true" to enable or "false" to disable. The
                   *                      default value is "false."
                   */
    void set_flag_weight_decay(float decay_val, bool enable_weight_decay);



private:

    MatrixF m_momentum_W;
    MatrixF m_sum_square_grad_W;
    std::string m_name;
    float m_rmsprop_learning_rate; // 1.0e-4f
    float m_learning_rate;
    float m_weight_decay;
    bool m_force_nonnegative;
    bool m_enable_weight_decay;
    float m_rms_counter = 1.0f;
    float m_rho;
    float m_momentum;
    int m_current_mode;
};

}

#endif /* _VARIABLE_UPDATER_H */
