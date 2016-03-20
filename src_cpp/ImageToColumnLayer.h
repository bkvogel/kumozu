#ifndef _IMAGETOCOLUMNLAYER_H
#define _IMAGETOCOLUMNLAYER_H
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

  /**
   * This layer converts between two different formats for a mini-batch of data. It is intended to
   * be inserted into network to convert an image-formatted input activations matrix into a
   * fully-connected-formatted output activations matrix.
   * This class simply copies data from input to output, converting between the too formats.
   * Thus, the layer before this one has
   * image-formatted activations, such as the output of a convolutional layer or activation function.
   * The layer after this one has fully-connected-formatted activations, such as a fully-connected
   * linear layer.
   *
   * input activations: A multi-dimensional matrix of size minibatch_size x dim1 x dim2 ... dimR. This typically
   * corresponds to one minibatch of output activations from a convolutional layer.
   *
   * output_forward: Size P x minibatch_size matrix where P = (dim1 x dim2 ... dimR)..
   * This typically corresponds to one mini-batch of input activations
   * to a fully-connected layer.
   */
  class ImageToColumnLayer : public Layer {

  public:

    /**
     * Create a new instance.
     *
     * @param name: A descriptive name.
     *
     */
  ImageToColumnLayer(std::string name) :
    Layer(name)
    {

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

    /*
     * Convert between two different formats for a mini-batch of data.
     *
     * B is a multi-dimensional matrix of size minibatch_size x dim1 x dim2 ... dimR. This typically
     * corresponds to one minibatch of output activations from a convolutional layer.
     *
     * This function simply copies data from B to A, converting between the too formats. Note that
     * A and B must therefore contain the same number of elements.
     *
     * A: Size P x minibatch_size matrix. This typically corresponds to one mini-batch of input activations
     * to a fully-connected layer.
     *
     * B: Size minibatch_size x dim1 x dim2 ... dimR matrix where P = (dim1 x dim2 ... dimR).
     *
     */
    void multi_dim_minibatch_to_column_minibatch(MatrixF& A, const MatrixF&B);

    /*
     * Same as multi_dim_minibatch_to_column_minibatch() except copies in the opposite
     * direction.
     */
    void column_minibatch_to_multi_dim_minibatch(const MatrixF& A, MatrixF&B);

    int m_minibatch_size;

  protected:

    /*
     * Compute the output activations as a function of input activations.
     *
     * The output activations can then be obtained by calling get_output_forward().
     *
     */
    virtual void forward_propagate(const MatrixF& input_activations) override;

    /*
     * Set the extents of the input activations. This must be called before the layer can be used.
     * This can be called at any time to re-initialize this layer with new input extents. Doing so
     * will of course erase any previous parameter values since the new parameter matrices may be
     * a different size.
     *
     * input_extents: Dimensions of the input activations, which are
     *                (dim_input, minibatch_size)
     */
    virtual void reinitialize(std::vector<int> input_extents) override;


  };

}

#endif /* _IMAGETOCOLUMNLAYER_H */
