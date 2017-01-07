#ifndef _MSECOSTFUNCTION_H
#define _MSECOSTFUNCTION_H
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
#include <functional>
#include "AtomicNode.h"


namespace kumozu {

/**
 * This is a Mean Squared Error (MSE) cost function in a neural network.
 *
 * A cost function is typically used during the training or evaluation process to compare the output
 * activations of a network to the "target" training output values. The degree to
 * which the outputs of the network differ from the true output values will be represented by
 * the "cost" or "loss" value.
 *
 * Input and output ports:
 *
 * By default, this cost function will automatically create its own output port with the name DEFAULT_OUTPUT_PORT_NAME.
 * During the forward data pass, the cost function value will be computed and stored in the "output forward" activations of
 * the output port.
 * The user should create excatly 1 input port (with any name) by connecting the output port of some other node to
 * this node.
 *
 * Usage:
 *
 * Assume that a node "C" contains the output activations for which we happen to have corresponding
 * target activations containing the known correct values. In this case, node "C" should be made the
 * parent of this cost function node. That is, the output port of node "C" should be connected to the
 * input port of this node.
 *
 * In order to compute the cost function, target activation values will be compared to the values in the
 * "input forward" activations of the input port. These target activations are supplied to this cost function
 * using the member function set_target_activations(), which must be called at least once before the first
 * call to forward(). Calling forward() will then compute the cost function, which will be written into the
 * "output forward" activations of the output port.
 *
 * It is possible to switch the target activations to another matrix at any time by calling set_target_activations()
 * again. Note that is not necessary to call set_target_activations() before each call to forward(). That is,
 * set_target_activations() should only be called when it is desired to change the target activations to
 * a different matrix.
 *
 * Gradient checking:
 *
 * It is possible to perform numerical gradient checking on the entire network, including the cost function.
 * This can be accomplished by placing everything (all nodes, including the cost function(s)) inside
 * a composite node. See the unit tests for an example.
 */
class MSECostFunction : public AtomicNode {

public:

    MSECostFunction(std::string name) :
        AtomicNode(name),
        m_target_activations {m_empty_target},
        m_has_target_activations {false}
    {
        if (VERBOSE_MODE) {
            std::cout << get_name() << std::endl;
        }
        // Create the output port with default name.
        create_output_port(m_output_var, DEFAULT_OUTPUT_PORT_NAME);
    }

    /**
     * Supply the target activations.
     *
     * The target activations contain the target values (i.e., assumed true/correct values) corresponding
     * to the input forward activations. That is, the input forward activations should contain the output
     * of the network and the target activations are the corresponding desired values for these activations, with
     * the same dimensions.
     *
     * Note that both the input activations corresponding to the input port and the target activations contain
     * one mini-batch of date, where the second dimension (column index) is the batch index within the mini-batch.
     *
     * This function must be called before the first forward() call, since the target activations
     * are needed by forward(). Otherwise, the program will exit with an error.
     * This function may be called an arbitrary number of times to supply a different target activations
     * matrix, if desired. Since matrix reference is saved, it is not necessary to call this function
     * repeatedly unless it is desired to change the reference to a different matrix.
     *
     * @param target_activations Contains the target activations which must be the same size as
     * the input forward activations: (unit_count x minibatch_size).
     */
    void set_target_activations(const MatrixF& target_activations) {
        m_target_activations = std::cref(target_activations);
        m_has_target_activations = true;
    }

    /**
     * Compute the output activations as a function of input activations.
     *
     * The computed cost will be returned in the output forward activations.
     *
     * Before calling this function for the first time, set_target_activations() must
     * be called to set the target activations. Otherwise, the program will exit with
     * an error.
     */
    virtual void forward_propagate() override;

    /**
     * Back-propagate errors to compute new values for input_backward.
     *
     */
    virtual void back_propagate_activation_gradients() override;

    /**
     * Reinitialize this node.
     *
     */
    virtual void reinitialize() override;

private:

    int m_minibatch_size;
    MatrixF m_temp_size_input;
    MatrixF m_temp_input_error;
    MatrixF m_empty_target;
    std::reference_wrapper<const MatrixF> m_target_activations;
    VariableF m_output_var; // associated with the default output port
    bool m_has_target_activations;
};

}


#endif /* _MSECOSTFUNCTION_H */
