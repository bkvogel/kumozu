#ifndef _BATCHNORMALIZATION1D_H
#define _BATCHNORMALIZATION1D_H
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
#include "Updater.h"
#include "Layer.h"

namespace kumozu {

  /*
   * An instance of this class represents a layer that implements batch normalization as described in:
   * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
   * by Sergey Ioffe and Christian Szegedy.
   *
   * This layer accepts input activations that correspond to one mini-batch of 1-dimensional activation
   * vectors, represented by a 2D matrix of dimensions (input_layer_size x minibatch_size).
   *
   * The output activations will have the same size as the input activations.
   *
   * The m_W matrix of the base class corresponds to "gamma" in the paper. It is a 1D matrix of size input_layer_size.
   * The m_bias matrix of the base class corresponds to "beta" in the paper. It is a 1D matrix of size nput_layer_size.
   *
   * To use batch normalization with convolutional layers, use BatchNormalization3D instead.
   *
   * Usage:
   *
   * According the the paper, the layer should be inserted after a linear layer or convolutional layer and
   * should be inserted just before the activation function.
   */
  class BatchNormalization1D : public Layer {

  public:

    /*
     * Create a new instance.
     *
     * Parameters:
     *
     * enable_gamma_beta: If true, enable the "scale and shift" using the gamma and
     *                    beta parameters, which are learned. Otherwise, only perform
     *                    mean and variance normalization.
     *
     * momentum: Fraction of current mini-batch mean and variance to use in the running average. This
     *           running average is computed when the network is in training mode but is only applied
     *           when the network is in test/evaluation mode. That is, when the network is in training
     *           mode, only the mini-batch mean and variance are actually used to normalize the activations.
     *           The default value is 0.05.
     *
     * name: A descriptive name.
     *
     */
  BatchNormalization1D(bool enable_gamma_beta, float momentum, std::string name) :
    Layer(name),
      m_normalization_epsilon {1e-5f},
      m_enable_gamma_beta {enable_gamma_beta},
        m_momentum {momentum},
          m_bypass_running_average {false}
          {

          }

          /*
           * Create a new instance.
           *
           * Only mini-batch statistics are used in this version. No running averages are used.
           *
           * Parameters:
           *
           * enable_gamma_beta: If true, enable the "scale and shift" using the gamma and
           *                    beta parameters, which are learned. Otherwise, only perform
           *                    mean and variance normalization.
           *
           * name: A descriptive name.
           *
           * Notes: This version seems to work better than the version that uses momentum.
           */
  BatchNormalization1D(bool enable_gamma_beta, std::string name) :
          Layer(name),
            m_normalization_epsilon {1e-5f},
            m_enable_gamma_beta {enable_gamma_beta},
              m_momentum {1.0},
                m_bypass_running_average {true}
                {

                }


                /*
                 * Compute the gradients for W and bias.
                 *
                 * This updates m_W_grad and m_bias_grad.
                 *
                 * The output error (that is, "output deltas") must have already been updated before
                 * calling this method. Note that a reference to the output deltas can be obtained by
                 * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
                 * correctly.
                 */
                virtual void back_propagate_paramater_gradients(const MatrixF& input_activations);





                /*
                 * Back-propagate errors to compute new values for input_error.
                 *
                 * The output error (that is, "output deltas") must have already been updated before
                 * calling this method. Note that a reference to the output deltas can be obtained by
                 * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
                 * correctly.
                 */
                virtual void back_propagate_deltas(MatrixF& input_error);



                /*
                 * Save learned parameters, stats etc. to a file withe the prefix given
                 * by the supplied name.
                 */
                void save_learning_info(std::string name) const;

                /*
                 * Set the momentum parameter. This specifies the fraction of the mean and variance from the
                 * current mini-batch to use in the running average. Must be in the range [0, 1]. Typical
                 * reasonable values are close to 0, such as 0.05 for example.
                 */
                void set_momentum(float momentum) {
                  m_momentum = momentum;
                }

                /*
                 * In the paper, the authors recommend using mean and variance averaged from the training set
                 * and using this values as fixed parameters when the network is in test/evaluate mode.
                 * This mode can be enabled by setting "is_enable" to false.
                 *
                 * To simply compute and use the mean and variance of the current mini-batch for scaling
                 * in test/evaluate mode, set "is_enable" to true. The default value is "true" because this
                 * seems to work better for some reason.
                 */
                void use_minibatch_stats_test(bool is_enable) {
                  m_bypass_running_average = is_enable;
                }


                // Make private again after debug.
  private:

                int m_dim_input;
                int m_minibatch_size;

                // Size is m_dim_input.
                MatrixF m_mean_cur_batch;
                MatrixF m_mean_to_use;
                MatrixF m_var_cur_batch;
                MatrixF m_var_to_use;

                MatrixF m_mean_running_avg;
                MatrixF m_var_running_avg;
                float m_normalization_epsilon;
                MatrixF m_input_sized_normalize_offset; // negative means used for normalizing, expanded to the input mini-batch size.
                MatrixF m_input_sized_normalize_scale; // normalization scale factors using activation variances.
                MatrixF m_temp_size_input;
                MatrixF m_centered_input;
                MatrixF m_temp_size_input_1d; // 1d array
                MatrixF m_x_hat;
                MatrixF m_W_expanded;
                MatrixF m_bias_expanded;
                bool m_enable_gamma_beta;
                float m_momentum;
                MatrixF m_var_deltas; // Partials of loss wrt normalizing variance.
                MatrixF m_mean_deltas; // Partials of loss wrt normalizing mean.
                MatrixF m_xhat_deltas; // Partials of loss wrt normalizing x hat in paper..
                bool m_bypass_running_average; // Set to true to always use the statistics from the current mini-batch.

  protected:



                /*
                 * Compute the output activations as a function of input activations.
                 *
                 * The output activations can then be obtained by calling get_output().
                 *
                 */
                virtual void forward_propagate(const MatrixF& input_activations);

                /*
                 * Reinitialize the layer based on the supplied new input extent values.
                 * This must be called before the layer can be used and must also be called whenever the
                 * input extents change during runtime.
                 *
                 * Note that a call to this function can be somewhat expensive since several matrices (such
                 * as the output activations and possibly the parameters as well) might need to
                 * be reallocated and initialized.
                 *
                 * input_extents: Dimensions of the input activations, which are
                 *                (dim_input, minibatch_size)
                 */
                virtual void reinitialize(std::vector<int> input_extents);


  };

}

#endif /* _BATCHNORMALIZATION1D_H */
