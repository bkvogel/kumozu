#ifndef _DROPOUT1D_H
#define _DROPOUT1D_H
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
#include <random>
#include "Layer.h"

namespace kumozu {



  /*
   * An instance of this class represents a dropout function that operates on a mini-batch
   * of input column vectors and outputs a corresponding mini-batch of column vectors.
   *
   * A single dropout mask is generated and then applied independently to each column in the mini-batch. Specifically,
   * the inputs to the dropout function are supplied as a MatrixF of size (dim_input x minibatch_size).
   * The output of the dropout function is a MatrixF of size (dim_output x minibatch_size).
   *
   * Usage: fixme
   *
   * Dropout should be applied only when in "training mode." To enable dropout, be sure to call
   * set_train_mode(true);
   *
   * Dropout should be turned off when in "testing/evaluation mode." To disable dropout, be sure to call
   * set_train_mode(false);
   *
   * This class implements "inverted dropout" as described in:
   * http://cs231n.github.io/neural-networks-2/
   *
   * Thus, we perform dropout and scale the activations in the forward training pass so that scaling is not performed
   * at test time.
   */
  class Dropout1D : public Layer {

  public:


    /*
     * Create an instance of a dropout function that operates on a mini-batch of
     * column activation vectors.
     *
     * Parameters
     *
     * prob_keep: Probability of keeping a unit active. Must be in the range (0, 1].
     */
  Dropout1D(float prob_keep, std::string name) :
    Layer(name),
      m_prob_keep {prob_keep},
      m_random_engine(m_rand_dev())
        {

        }




      /*
       * Compute the dropout function in the reverse direction.
       *
       * The reverse dropout function of the output deltas activations member variable
       * is computed and the result is stored in the supplied input error activations variable.
       * This method is typically used during the back-propigation step to back-propagate
       * deltas (errors) through the reverse dropout function.
       */
      virtual void back_propagate_deltas(MatrixF& input_backward, const MatrixF& input_forward);

      /*
       * Set the probability of keeping any given element.
       */
      void set_prob_keep(float prob_keep) {
        m_prob_keep = prob_keep;
      }

  protected:

      /*
       * Compute the dropout function in the forward direction.
       *
       * The dropout function of the supplied input activations is computed.
       * The results are stored in the output activations member variable, which
       * can be obtained by calling get_output_forward().
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

  private:


      float m_prob_keep;
      Matrix<int> m_dropout_mask;
      std::random_device m_rand_dev;
      //std::mt19937 m_random_engine;
      std::default_random_engine m_random_engine;
      // 0: Same random mask for each example in mini-batch (i.e., mini-batch size copies of same random mask)
      // 1: Random mask values for each input activation in mini-batch.
      int m_mode = 1;

  };

}

#endif /* _DROPOUT1D_H */
