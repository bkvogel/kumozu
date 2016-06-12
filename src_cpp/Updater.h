#ifndef _UPDATER_H
#define _UPDATER_H
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
#include <functional>
#include <memory>
#include "Constants.h"
#include "Utilities.h"
#include "MatrixRefVector.h"

namespace kumozu {

  /**
   * Apply an optimization algorithm to update parameter values using the corresponding gradients.
   *
   * An instance of this class is associated with two lists of matrices that are supplied to the constructor:
   * - A list of parameter matrices that that the optimization algorithm will modify.
   * - A list of read-only matrices that contain the corresponding gradients.
   *
   * Usage:
   *
   * First create a new instance of this class, supplying the paramer and gradient lists to the constructor.
   *
   * Then optionally call one of the mode-setting functions to configure the update algorithm.
   *
   * Then, whenver new gradients are available, call update() to use the gradients to update the corresponding parameter values.
   *
   * Typically, on Updater instance will be created for the weights parameters and another Updater for the bias parameters.
   * Note also the Node contains functions get*list() than can be used to obtain the matrix lists that need to be supplied
   * to the constructor.
   *
   */
  class Updater {

  public:

    /**
     * Create a new instance that will update the parameter matrices in "parameters_list" using the corresponding
     * gradients from "gradients_list."
     *
     *
     * Parameters:
     *
     * @param parameters_list The learnable parameters of the model, such as the weights and/or bias matrices.
     * @param gradients The gradients corresponding to parameters_list. At the time update() is called, these
     * two list must have the same number of matrices and for each list index, the two matrices must have
     * the same dimensions.
     */
  Updater(MatrixRefVectorF& parameters_list, const MatrixRefVectorF& gradients_list, std::string name):
    m_parameters_list {parameters_list},
      m_gradients_list {gradients_list},
        m_name {name},
          m_learning_rate {0.01f}, // default learning rate.
            m_weight_decay {0.0f},
              m_force_nonnegative {false},
                m_enable_weight_decay {false},
                  m_current_mode {0} // default mode.
                  {

                  }

                  /**
                   * Update the elements in the parameters there were passed to the constructor using the
                   * corresponding gradients that were passed to the constructor.
                   *
                   * Depending on the current update mode, it is possible that the updated values may
                   * also be influenced by the history of past gradients supplied to this method
                   * up until now.
                   *
                   */
                  void update();



                  //////////////////////////////////////////////////////////////////////////////////////////////////
                  //
                  // Set update mode.
                  //
                  // Only one mode can be active at a time.

                  /**
                   * Set a constant learning rate.
                   *
                   * This is the default mode.
                   *
		   * @param learning_rate Default values: learning_rate = 0.01.
                   */
                  void set_mode_constant_learning_rate(float learning_rate);

		  /**
		   * Use Rmsprop.
		   */ 
                  void set_mode_rmsprop(float rmsprop_learning_rate, float rho);

                  /**
		   * Use Rmsprop with momentum.
		   *
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

                  MatrixRefVectorF& m_parameters_list;
                  const MatrixRefVectorF& m_gradients_list;
                  std::vector<std::unique_ptr<MatrixF>> m_momentum_list;
                  std::vector<std::unique_ptr<MatrixF>> m_sum_square_grad_list;
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

#endif /* _UPDATER_H */
