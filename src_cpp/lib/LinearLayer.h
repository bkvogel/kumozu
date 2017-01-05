#ifndef _LINEARLAYER_H
#define _LINEARLAYER_H
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
 * A fully-connected linear layer in a neural network.
 *
 * This layer computes the following function:
 *
 * output(m) = W*input(m) + bias for each m in 1...minibatch_size
 *
 *
 * The m'th example in a mini-batch corresponds to the m'th column of the input and output matrices.
 *
 */
class LinearLayer : public Layer {

public:

    /**
     * Create a new instance.
     *
     * Parameters:
     *
     * @param dim_output: Number of units in the output layer. Note that the dimensions of the output activations
     *             will be (dim_output, minibatch_size).
     *
     * @param enable_bias Enable bias if true (default).
     *
     * @param name: An optional descriptive name.
     *
     */
    LinearLayer(int dim_output, bool enable_bias = true, std::string name = "") :
        Layer(name),
        m_dim_output {dim_output},
        m_use_fixed_random_back_prop {false},
        m_enable_bias {enable_bias}
    {
        add_param("W");
        add_param("bias");
    }

    /**
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

    // Item 33 in Effective C++, 3rd edition:
    using Node::back_propagate_paramater_gradients;
    using Node::back_propagate_activation_gradients;


    virtual void back_propagate_paramater_gradients(const MatrixF& input_data) override;


    virtual void back_propagate_activation_gradients(MatrixF& input_grad, const MatrixF& input_data) override;


private:
    int m_dim_output;
    int m_input_layer_units;
    int m_output_layer_units;
    int m_minibatch_size;
    MatrixF m_temp_size_W;
    MatrixF m_temp_size_bias;
    Matrix<int> m_output_forward_indices; // fixme: remove?
    MatrixF m_W_fixed_random;
    bool m_use_fixed_random_back_prop;
    bool m_enable_bias;


protected:

    virtual void forward_propagate(const MatrixF& input_data) override;

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
    virtual void reinitialize(std::vector<int> input_extents) override;

};

}

#endif /* _LINEARLAYER_H */
