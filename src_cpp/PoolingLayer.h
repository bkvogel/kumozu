#ifndef _POOLINGLAYER_H
#define _POOLINGLAYER_H
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



  /**
   * An instance of this class represents a pooling layer in a network.
   *
   * The input to the pooling layer is a mini-batch of 3D matrices of size (depth_in x height_in x width_in).
   * Adding the mini-batch index as a dimension results in a 4D matrix of size (minibatch_size x depth_in x height_in x width_in).
   *
   * The output of the pooling layer is a mini-batch of 3d matrices:
   * Output: 4D matrix of size (minibatch_size x depth_out x height_out x width_out).
   *
   * This class currently supports only max-pooling.
   * todo: Consider adding support for Ben Graham's "Fractional Max Pooling."
   *
   * For the i'th mini-batch, pooling is performed by partitioning the 3D (depth_in x weight_in x width_in).
   * input matrix into boxes of size (depth_in/depth_out x height_in/height_out x width_in/width_out).
   *
   */
  class PoolingLayer : public Layer {

  public:


    /**
     * Create an instance of an pooling function that operates on a mini-batch of
     * data at a time.
     *
     * Parameters:
     *
     *
     * @param pooling_region_extents: Extents that specify the size of each pooling region. This must correspond to a 3-D matrix of
     *                         size (pooling_size_depth, pooling_size_height, pooling_size_width).
     *
     * @param pooling_region_step_sizes: The i'th element of this vector contains the step size for the i'th region.
     *                         This must correspond to a 3-D matrix of
     *                         size (pooling_step_depth, pooling_step_height, pooling_step_width). Note that if the
     *                         step size is chosen to be the same as the region size, then the regions will have no overlap.
     *                         There will be overlap if the step size is less than the region size.
     *
     */
  PoolingLayer(const std::vector<int>& pooling_region_extents, const std::vector<int>& pooling_region_step_sizes,
               std::string name) :
    Layer(name),
      m_pooling_region_extents {pooling_region_extents},
      m_pooling_region_step_sizes {pooling_region_step_sizes},
        m_decay_unused_penalty {0.0f}
        {

        }

        /*
         * Set the decay penalty to be used in reverse_activation() with certain activation functions such as
         * kmax_decay_unused, ReLU_decay_unused, maxout_decay_unused.
         *
         * The supplied value must be in the range [0, 1] where 0 denotes no penalty and 1 denotes the maximum penalty.
         *
         * fixme: currently disabled in release version.
         */
        void set_decay_penalty(float decay_penalty) {
          m_decay_unused_penalty = decay_penalty;
        }

        /*
         * Back-propagate errors to compute new values for input_backward.
         *
         * The output error (that is, "output deltas") must have already been updated before
         * calling this method. Note that a reference to the output deltas can be obtained by
         * calling get_output_backward(). Otherwise, the error gradients will not be back-propagated
         * correctly.
         */
        virtual void back_propagate_deltas(MatrixF& input_backward, const MatrixF& input_forward) override;


  private:

        std::vector<int> m_pooling_region_extents;
        std::vector<int> m_pooling_region_step_sizes;
        float m_decay_unused_penalty;
        Matrix<int> m_state;

        /*
         * Compute maxout on each 3D matrix in a minibatch.
         *
         * input: Size (minibatch_size, dim0_in, dim1_in, dim2_in)
         * output: Size (minibatch_size, dim0_out, dim1_out, dim2_out)
         * state: Size (minibatch_size, dim0_out, dim1_out, dim2_out, 3)
         *
         * For each maximum value stored in "output", the corresponding location of this maximum value in "input" is stored
         * in "state". Suppose that for a given cube inside "input", the maximum value is found to occur at
         * input(batch, max_ind0, max_ind1, max_ind2) and that this maximum value is stored into state(batch, i,j,k).
         * Then we store the indices in maxout_indices as
         *
         * max_ind0 = state(batch, i,j,k,0)
         * max_ind1 = state(batch, i,j,k,1)
         * max_ind2 = state(batch, i,j,k,2)
         *
         * Note that the actual dimension of "output" are determined by the dimensions of "input" and the pooling regions extents
         * and step sizes.
         *
         */
        void forward_maxout_3d(const MatrixF& input, MatrixF& output, Matrix<int>& state,
                               const std::vector<int>& pooling_region_extents, const std::vector<int>& pooling_region_step_sizes);


        /*
         * Reverse-direction maxout.
         *
         * Performance optimization: We use the following trick to enable a fast parallel implementation. Note that each delta
         * must be added to the corresponding element of input_backward (i.e., where the maximum element in the 3D box occured
         * during the forward pass). Assume that the step sizes
         * are such that the 3D boxes overlap at most 50%. It then follows that it is not possible for the even-numbered indexes
         * in output_backward to point to the same element in input_backward. Likewise for the odd-number indexes in output_backward.
         * We can therefore perform the back propagation of deltas in two steps. First perform the operation in parrallel
         * over the even-numbered indexes. When this completes, we can perform the operation in parrellel over the
         * odd-numbered indexes.
         */
        void reverse_maxout_3d(MatrixF& input_backward, const MatrixF& output_backward, Matrix<int>& state,
                               const std::vector<int>& pooling_region_extents, const std::vector<int>& pooling_region_step_sizes);

  protected:

        /*
         * Compute the output activations as a function of input activations.
         *
         * The output activations can then be obtained by calling get_output_forward().
         *
         */
        virtual void forward_propagate(const MatrixF& input_activations) override;

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
         *                (minibatch_size, depth_in, height_in, width_in)
         */
        virtual void reinitialize(std::vector<int> input_extents) override;

  };

}

#endif /* _POOLINGLAYER_H */
