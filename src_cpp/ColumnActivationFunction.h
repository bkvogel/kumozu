#ifndef _COLUMNACTIVATIONFUNCTION_H
#define _COLUMNACTIVATIONFUNCTION_H
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
#include "Layer.h"

namespace kumozu {



  /*
   * An instance of this class represents an activation function that operates on a mini-batch
   * of input column vectors and outputs a corresponding mini-batch of column vectors.
   *
   * The activation function is applied independently to each column. Specifically,
   * the inputs to the activation are supplied as a MatrixF of size (dim_input x minibatch_size).
   * The output of the activation function is a MatrixF of size (dim_output x minibatch_size).
   *
   */
  class ColumnActivationFunction : public Layer {

  public:

    enum class ACTIVATION_TYPE { ReLU, leakyReLU, linear, maxout, kmax, kmax_decay_unused, ReLU_decay_unused, maxout_decay_unused };

    /*
     * Create an instance of an activation function that operates on a mini-batch of
     * column activation vectors.
     *
     */
  ColumnActivationFunction(ACTIVATION_TYPE activation_type, std::string name) :
    Layer(name),
      m_activation_type {activation_type},
      m_maxout_factor {1},
        // Default values for kmax:
        m_partition_count {1},
          m_k {1},
            m_decay_unused_penalty {0.0f}
            {



            }

            /*
             * Set the decay penalty to be used in reverse_activation() with certain activation functions such as
             * kmax_decay_unused, ReLU_decay_unused, maxout_decay_unused.
             *
             * The supplied value must be in the range [0, 1] where 0 denotes no penalty and 1 denotes the maximum penalty.
             */
            void set_decay_penalty(float decay_penalty) {
              m_decay_unused_penalty = decay_penalty;
            }

            /*
             * Set parameters to be used by the kmax activation function.
             */
            void set_kmax_parameters(float partition_count, float k) {
              m_partition_count = partition_count;
              m_k = k;
            }

            /*
             * Set parameter to be used by the maxout activation function.
             *
             * This function must be called before forward_propagate(). Otherwise,
             * the default value of 1 will be used.
             */
            void set_maxout_factor(int maxout_factor) {
              m_maxout_factor = maxout_factor;
            }
            //



            /*
             * Back-propagate errors to compute new values for input_error.
             *
             * The output error (that is, "output deltas") must have already been updated before
             * calling this method. Note that a reference to the output deltas can be obtained by
             * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
             * correctly.
             */
            virtual void back_propagate_deltas(MatrixF& input_error);


  private:


            Matrix<int> m_state;
            ACTIVATION_TYPE m_activation_type;

            int m_maxout_factor;
            int m_partition_count; // fixme: let a method set this.
            int m_k; // fixme: let a method set this.
            float m_decay_unused_penalty;
            std::vector<int> m_input_extents;

  protected:

            /*
             * Compute the activation function in the forward direction.
             *
             * The activation function of the supplied input activations is computed.
             * The results are stored in the output activations member variable, which
             * can be obtained by calling get_output().
             *
             * input_activations: The input (and output activations) which are
             *                (minibatch_size x depth x height x width).
             *
             */
            virtual void forward_propagate(const MatrixF& input_activations);

            /*
             * Set the extents of the input activations. This must be called before the layer can be used.
             * This can be called at any time to re-initialize this layer with new input extents. Doing so
             * will of course erase any previous parameter values since the new parameter matrices may be
             * a different size.
             *
             * input_extents: Dimensions of the input activations, which are
             *                (dim_input, minibatch_size)
             */
            virtual void reinitialize(std::vector<int> input_extents);


  };

}

#endif /* _COLUMNACTIVATIONFUNCTION_H */
