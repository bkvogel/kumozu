#ifndef _MULTIPLYER_NODE_H
#define _MULTIPLYER_NODE_H
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
#include "AtomicNode.h"
#include <string>
#include <iostream>
#include "Utilities.h"
#include "Constants.h"

namespace kumozu {

/**
   * A Node the computes the element-wise multiplication of all "input data" matrices associated with its input ports.
   *
   * This node is allowed to have an arbitrary number of input ports. It will have 1 outptut port which will be given the default name.
   * All matrices associated with
   * the input ports must have the same dimensions. Arbitrary-dimensional matrices are supported. The matrices associated
   * with the output port will have the same dimensions as those associated with the input.
   *
   * This node simply performs the element-wise multiplication of the "input forward" matrices over all of the input ports. The names
   * of the input ports can be arbitrary.
   *
   * Usage:
   *
   * Obtain an instance of this class and call the create_input_port() functions of Node to create as many input ports as desired.
   * Although the port names can be arbitrary, the user is required to specify the port name when more than 1 input port is used.
   * If only 1 input port is created, this node will simply function as an identity function that will pass the input through unchanged.
   */
class MultiplyerNode : public AtomicNode {

public:

    /**
     * Create a new instance with the specified node name and create the output port with default name.
     */
    MultiplyerNode(std::string name) :
        AtomicNode(name) {
        // Create the 1 output port.
        create_output_port(m_output_var, DEFAULT_OUTPUT_PORT_NAME);
    }

    /**
     * Set output forward activations to the element-wise multiplication over all input forward activations.
     */
    virtual void forward_propagate() override {
        set_value(m_output_var.data, 1.0f);
        for_each_input_port_data([&] (const MatrixF& mat) {
            element_wise_multiply(m_output_var.data, m_output_var.data, mat);
        });
    }

    /**
     *
     */
    virtual void back_propagate_activation_gradients() override {
        MatrixF& deltas = get_output_grad();
        for (size_t i = 0; i < m_input_port_backward_matrices.size(); ++i) {
            MatrixF& input_backward = m_input_port_backward_matrices[i].get();
            //copy_matrix(input_backward, deltas);
            input_backward = deltas;
            for (size_t j = 0; j < m_input_port_forward_matrices.size(); ++j) {
                if (i != j) {
                    const MatrixF& other_mat = m_input_port_forward_matrices[j].get();
                    element_wise_multiply(input_backward, input_backward, other_mat);
                }
            }
        }
    }

    /**
     * Check that all input ports are the same size.
     */
    virtual void reinitialize() override {
        // First verify that all input ports are associated with matrices of the same dimensions.
        m_input_extents.clear();

        for_each_input_port_data([&] (const MatrixF& mat) {
            //std::cout << "in_mat:" << std::endl << mat << std::endl;
            if (m_input_extents.size() == 0) {
                m_input_extents = mat.get_extents();
                m_output_var.resize(m_input_extents);
            } else {
                if (m_input_extents != mat.get_extents()) {
                    error_exit(get_name() + ": Error: Not all input matrices have the same extents.");
                }
            }
        });
        m_input_port_backward_matrices.clear();
        for_each_input_port_grad([&] (MatrixF& mat) {
            m_input_port_backward_matrices.push_back(std::ref(mat));
        });
        m_input_port_forward_matrices.clear();
        for_each_input_port_data([&] (const MatrixF& mat) {
            m_input_port_forward_matrices.push_back(std::cref(mat));
        });
    }

private:

    std::vector<int> m_input_extents; // Extents of each input port matrix.
    //MatrixF m_output_forward; // associated with the default output port
    //MatrixF m_output_backward; // associated with the default output port
    VariableF m_output_var; // associated with the default output port
    std::vector<std::reference_wrapper<MatrixF>> m_input_port_backward_matrices;
    std::vector<std::reference_wrapper<const MatrixF>> m_input_port_forward_matrices;

};

}

#endif /* _MULTIPLYER_NODE_H */
