#ifndef _BOXACTIVATIONFUNCTION_H
#define _BOXACTIVATIONFUNCTION_H
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
   * of input 3D (i.e., box-shaped) matrices and outputs a corresponding mini-batch of 3D
   * output matrices. Typically, the input corresponds to a mini-batch of activations from the
   * output of a linear convolutional layer.
   *
   * The activation function is applied independently to each 3D matrix in the mini-batch. Specifically,
   * the inputs to the activation are supplied as a MatrixF of size (minibatch_size, depth, height, width).
   * The output of the activation function is a matrix of the same size as the input matrix.
   *
   */
  class BoxActivationFunction : public Layer {

  public:

    enum class ACTIVATION_TYPE { ReLU, leakyReLU, linear, kmax, kmax_decay_unused, ReLU_decay_unused};


    /*
     * Create an instance of an activation function with the supplied activation type.
     *
     */
  BoxActivationFunction(ACTIVATION_TYPE activation_type, std::string name) :
    Layer(name),

      m_activation_type {activation_type},
      // Default values for kmax:
      m_box_depth {1},
        m_box_height {1},
          m_box_width {1},
            m_k {1},
              m_decay_unused_penalty {0.0f} // todo: default should be 1 or 0?
              {


              }

              /*
               * Set the decay penalty to be used in reverse_activation() with certain activation functions such as
               * kmax_decay_unused, ReLU_decay_unused, maxout_decay_unused.
               *
               * The supplied value must be in the range [0, 1] where 0 denotes no penalty and 1 denotes the maximum penalty.
               *
               * fixme: currently disabled.
               */
              void set_decay_penalty(float decay_penalty) {
                m_decay_unused_penalty = decay_penalty;
              }

              /*
               * Set parameters for kmax activation function.
               *
               * If this function is not called. The default value of 1 is used for all parameters.
               */
              void set_kmax_parameters(int box_depth, int box_height, int box_width, int k) {
                m_box_depth = box_depth;
                m_box_height = box_height;
                m_box_width = box_width;
                m_k = k;
              }



              /*
               * Back-propagate errors to compute new values for input_backward.
               *
               * The output error (that is, "output deltas") must have already been updated before
               * calling this method. Note that a reference to the output deltas can be obtained by
               * calling get_output_backward(). Otherwise, the error gradients will not be back-propagated
               * correctly.
               */
              virtual void back_propagate_deltas(MatrixF& input_backward, const MatrixF& input_forward);





  private:

              int m_minibatch_size;
              int       m_depth;
              int       m_height;
              int       m_width;
              Matrix<int> m_state;
              ACTIVATION_TYPE m_activation_type;

              int m_box_depth;
              int m_box_height;
              int m_box_width;
              int m_k; // When using kmax, number of activations to keep in each box partition.
              float m_decay_unused_penalty;

  protected:

              /*
               * Compute the activation function in the forward direction.
               *
               * The activation function of the supplied input activations is computed.
               * The results are stored in the output activations member variable, which
               * can be obtained by calling get_output_forward().
               *
               * input_activations: The input (and output activations) which are
               *                (minibatch_size, depth, height, width).
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
               *                (minibatch_size, depth, height, width)
               */
              virtual void reinitialize(std::vector<int> input_extents);

  };

}

#endif /* _BOXACTIVATIONFUNCTION_H */
