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



/**
 * An instance of this class represents an activation function that operates on a mini-batch
 * of input column vectors and outputs a corresponding mini-batch of column vectors. The
 * activation function operates independently on each column of the input activations to
 * produce the corresponding column of the output activations. Note that each example
 * in a mini-batch therefore corresponds to a column vector in the 2D input matrix,
 * unlike many other frameworks that place the examples as the rows of the input matrix.
 *
 * Specifically,the inputs to the activation function are supplied as a MatrixF of
 * size (dim_input x minibatch_size).
 * The output of the activation function is a MatrixF of size (dim_output x minibatch_size).
 *
 */
// todo: Separate into 1 class per activation type.
// Also merge the 1D and 3D activation functions into a single class for each type of activation.
class ColumnActivationFunction : public Layer {

public:

    enum class ACTIVATION_TYPE { ReLU, leakyReLU, identity, tanh, sigmoid, maxout, kmax, kmax_decay_unused, ReLU_decay_unused, maxout_decay_unused };

    /**
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

    void set_activation_type(ACTIVATION_TYPE activation_type) {
        m_activation_type = activation_type;
        set_initialized(false);
    }

    /**
     * Set the decay penalty to be used in reverse_activation() with certain activation functions such as
     * kmax_decay_unused, ReLU_decay_unused, maxout_decay_unused.
     *
     * The supplied value must be in the range [0, 1] where 0 denotes no penalty and 1 denotes the maximum penalty.
     */
    void set_decay_penalty(float decay_penalty) {
        m_decay_unused_penalty = decay_penalty;
    }

    /**
     * Set parameters to be used by the kmax activation function.
     */
    void set_kmax_parameters(float partition_count, float k) {
        m_partition_count = partition_count;
        m_k = k;
    }

    /**
     * Set parameter to be used by the maxout activation function.
     *
     * This function must be called before forward_propagate(). Otherwise,
     * the default value of 1 will be used.
     */
    void set_maxout_factor(int maxout_factor) {
        m_maxout_factor = maxout_factor;
    }


    virtual void back_propagate_activation_gradients(MatrixF& input_grad, const MatrixF& input_data) override;

private:

    Matrix<int> m_state;
    ACTIVATION_TYPE m_activation_type;
    int m_maxout_factor;
    int m_partition_count;
    int m_k;
    float m_decay_unused_penalty;
    std::vector<int> m_input_extents;

protected:


    virtual void forward_propagate(const MatrixF& input_forward) override;


    virtual void reinitialize(std::vector<int> input_extents) override;


};

}

#endif /* _COLUMNACTIVATIONFUNCTION_H */
