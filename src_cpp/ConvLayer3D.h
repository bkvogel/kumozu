#ifndef _CONVLAYER3D_H
#define _CONVLAYER3D_H
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
   * A convolutional layer that operates on a mini-batch of grayscale, color images or
   * hidden activations. If the input corresponds to grayscale images, image_depth will be 1.
   * If the input corresponds to color images, image_depth will be 3. For internal layers, image_depth
   * will likely be in the 10s or 100s.
   *
   * This class performs 2D convolution along the image_height and image_width dimensions of the input
   * activations but not along the image_depth dimension. There will be "filter_count" total convolutional
   * filters, each of size conv_filter_height x conv_filter_width x image_depth. Thus, each convolutional
   * filters corresponds to a 3D cube of filter coeficients that moves around the input along the image height
   * and image width dimensions only. The convolution is performed with zero padding so that the output image has
   * the same size (image_height, image_width) along two of its three dimensions as the input. The third dimension
   * of the output of course corresponds to the "filter_count".
   *
   * This class operates on a mini-batch of data at a time, where the mini-batch dimension is the first dimension of
   * the supplied input activations and is also the first dimension of the output activations. Thus, the input
   * activations are represented as a 4D matrix which is conceptually minibatch_size 3D matrices of size
   * (image_depth, image_height, image_width).
   *
   */
  class ConvLayer3D : public Layer {

  public:

    /*
     * Create a new instance.
     *
     * Parameters:
     *
     * filter_count: Number of convolutional filters.
     *
     * conv_filter_height: Height of a convolutional filter.
     *
     * conv_filter_width: Width of a convolutional filter.
     *
     * name: A descriptive name.
     *
     * Note that each convolutional filter also has depth image_depth but the filter is not convolved along
     * this dimension. That is, each filter corresponds to a 3D cube, but the convolution operation is
     * only performed along 2 or the 3 dimensions.
     *
     */
  ConvLayer3D(int filter_count, int conv_filter_height,
              int conv_filter_width, std::string name) :
    Layer(name),
      m_filter_count {filter_count},
      m_conv_filter_height {conv_filter_height},
        m_conv_filter_width {conv_filter_width},
          m_use_fixed_random_back_prop {false}
          {


          }




          /*
           * Set to true to enable using a fixed random weights matrix for back-propagation.
           *
           * The default value is false.
           *
           * I have found that in some cases, a network can still learn when a fixed random
           * weights matrix is used. This is a more biologically plausible form of back propagation
           * since the learned weight values are not used during the back-propagation step.
           */
          void enable_fixed_random_back_prop(bool enable) {
            if (enable) {
              m_use_fixed_random_back_prop = true;
            } else {
              m_use_fixed_random_back_prop = false;
            }
          }


          /*
           * Compute the gradients for W and bias.
           *
           * This updates m_W_grad and m_bias_grad.
           *
           * back_propagate_deltas() must have already been called before calling this function.
           *
           * The output error must have already been updated
           * by an external objct, such as the next layer in a network, before
           * calling this function. Otherwise, the error gradients will not be back-propagated
           * correctly. Note that a reference to the output error can be obtained by
           * calling get_output_deltas().
           */
          virtual void back_propagate_paramater_gradients(const MatrixF& input_activations);

          /*
           * Back-propagate errors to compute new values for input_error.
           *
           * The output error (that is, "output deltas") must have already been updated before
           * calling this function. Note that a reference to the output deltas can be obtained by
           * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
           * correctly.
           */
          virtual void back_propagate_deltas(MatrixF& input_error);

  private:

          int m_minibatch_size;
          int m_filter_count;
          int m_conv_filter_height;
          int m_conv_filter_width;
          int m_image_depth;
          int m_image_height;
          int m_image_width;

          MatrixF m_temp_size_W;
          MatrixF m_temp_size_bias;

          // Temp matrices for optimized convolution functions.
          MatrixF m_temp_Z2;
          MatrixF m_temp_A1;
          MatrixF m_temp_W;

          float m_force_nonnegative = false;
          MatrixF m_W_fixed_random;
          bool m_use_fixed_random_back_prop;

  protected:

          /*
           * Compute the output activations as a function of input activations.
           *
           * The output activations can then be obtained by calling get_output().
           *
           * input_activations: The input activations for 1 mini-batch of data. The
           *                    dimensions are (minibatch_size, image_depth, image_height, image_width).
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
           *                (minibatch_size, image_depth, image_height, image_width)
           */
          virtual void reinitialize(std::vector<int> input_extents);

  };

}

#endif /* _CONVLAYER3D_H */
