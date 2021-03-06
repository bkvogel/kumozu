#ifndef _BATCHNORMALIZATION3D_H
#define _BATCHNORMALIZATION3D_H
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
 * An instance of this class represents a layer that implements batch normalization as described in:
 * "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
 * by Sergey Ioffe and Christian Szegedy.
 *
 * This layer accepts input activations that correspond to one mini-batch of 3-dimensional activation
 * vectors, represented by a 4D matrix of dimensions (minibatch_size, image_depth, image_height, image_width).
 * It is assumed that the input to this layer is the output of a convolutional layer where the convolution
 * is applied on a 2D grid (image_height x image_width) so that each convolutional filter channel corresponds to a single value of the
 * image_depth dimension. As discussed in section 3.2 of the paper, we treat the effective mini-batch size
 * as  (minibatch_size x image_height x image_width). This is because we have one distinct convolutional filter
 * per value of image_depth.
 *
 * The output activations will have the same size as the input activations.
 *
 * The m_W matrix of the base class corresponds to "gamma" in the paper. It is a 1D matrix of size image_depth.
 * The m_bias matrix of the base class corresponds to "beta" in the paper. It is a 1D matrix of size image_depth.
 *
 * To use batch normalization with 1D linear layers, use BatchNormalization1D instead.
 *
 * Usage:
 *
 * According the the paper, the layer should be inserted after a linear layer or convolutional layer and
 * should be inserted just before the activation function.
 */
class BatchNormalization3D : public Layer {

public:


    /**
     * Create a new instance.
     *
     * Parameters:
     *
     * @param enable_gamma_beta: If true, enable the "scale and shift" using the gamma and
     *                    beta parameters, which are learned. Otherwise, only perform
     *                    mean and variance normalization.
     *
     * @param momentum: Fraction of current mini-batch mean and variance to use in the running average. This
     *           running average is computed when the network is in training mode but is only applied
     *           when the network is in test/evaluation mode. That is, when the network is in training
     *           mode, only the mini-batch mean and variance are actually used to normalize the activations.
     *           The default value is 0.01.
     *
     * @param name: A descriptive name.
     *
     */
    BatchNormalization3D(bool enable_gamma_beta, float momentum=0.01f, std::string name="BatchNormalization3D") :
        Layer(name),
        m_normalization_epsilon {1e-5f},
        m_enable_gamma_beta {enable_gamma_beta},
        m_momentum {momentum}
    {
        add_param("W");
        add_param("bias");
    }


    virtual void back_propagate_paramater_gradients(const MatrixF& input_activations) override;
    virtual void back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) override;

    /**
     * Set the momentum parameter. This specifies the fraction of the mean and variance from the
     * current mini-batch to use in the running average. Must be in the range [0, 1]. Typical
     * reasonable values are close to 0, such as 0.05 for example.
     */
    void set_momentum(float momentum) {
        m_momentum = momentum;
    }


private:

    int m_minibatch_size;
    int m_image_depth;
    int m_image_height;
    int m_image_width;
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
    MatrixF m_xhat_deltas; // Partials of loss wrt normalizing x hat in paper.
    bool m_is_first_batch {true};


protected:

    virtual void forward_propagate(const MatrixF& input_activations) override;
    virtual void reinitialize(std::vector<int> input_extents) override;

};

}

#endif /* _BATCHNORMALIZATION3D_H */
