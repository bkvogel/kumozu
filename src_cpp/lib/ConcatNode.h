#ifndef _CONCAT_NODE_H
#define _CONCAT_NODE_H
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
 * A Node that concatenates all of the "input data" matrices associated with its input ports.
 *
 * This node is allowed to have an arbitrary number of input ports. It will have 1 output port with the default name.
 * All matrices associated with should have the same extents except for possibly the axis extent along which
 * the concatenation is performed. For example, in the case where the input matrices are 2D, the second
 * extent is typically the mini-batch extent, and so the concatentation axis will by 0. In this case,
 * all input matrices must have the same size of the second extent.
 *
 * This node simply performs the concatenation of the "input data" matrices over all of the input ports. The concatenation
 * is performed along the "axis" dimension. The size of the axis dimension of the "output data"
 * matrix will be the sum of the sizes of the corresponding dimension over all input ports.
 * The names of the input ports can be arbitrary.
 *
 * Concatenation order:
 *
 * The order in which the inputs are concatenated is the same as their order in a lexicographical sort of the port names. Therefore,
 * if you would like for the concatenation to be carried out in a certain order, just give the input ports names such that
 * their lexicographical ordering is the same as the desired concatenation ordering.
 *
 * Usage:
 *
 * Obtain an instance of this class and call the create_input_port() functions of Node to create as many input ports as desired.
 * Although the port names can be arbitrary, the user is required to specify the port name when more than 1 input port is used.
 * If only 1 input port is created, this node will simply function as an identity function that will pass the input through unchanged.
 */
class ConcatNode : public AtomicNode {

public:

    /**
     * Create a new instance with the specified node name and create the output port with default name.
     *
     * @param axis The axis along which to perform the cancatenation operation.
     */
    ConcatNode(int axis, std::string name) :
        AtomicNode(name),
        m_axis {axis}
    {
        // Create the 1 output port.
        create_output_port(m_output_var, DEFAULT_OUTPUT_PORT_NAME);
    }

    /**
     * Set output forward activations to the concatenation over all input forward activations.
     */
    virtual void forward_propagate() override {
        set_value(m_output_var.data, 0.0f);
        int axis_offset = 0;
        for_each_input_port_data([&] (const MatrixF& mat) {
            copy_along_axis(mat, m_output_var.data, axis_offset, m_axis);
            axis_offset += mat.extent(0);
        });
    }

    virtual void back_propagate_paramater_gradients() {}

    /**
     * Copy a submatrix from the "output backward" values into each "input backward" matrix.
     */
    virtual void back_propagate_activation_gradients() override {
        MatrixF& deltas = get_output_grad();
        int axis_offset = 0;
        for_each_input_port_grad([&] (MatrixF& mat) {
            extract_along_axis(mat, deltas, axis_offset, m_axis);
            axis_offset += mat.extent(0);
        });
    }

    /**
     * Check that all inputs have the same mini-batch size.
     */
    virtual void reinitialize() override {
        // First verify that all input ports are associated with matrices of the same dimensions.
        int out_axis_dim = 0;
        for_each_input_port_data([&] (const MatrixF& mat) {
            if (m_out_extents.size() == 0) {
                m_out_extents = mat.get_extents();
            } else {
                const auto& temp_extents = mat.get_extents();
                for (auto n = 0; n < m_out_extents.size(); ++n) {
                    if (n != m_axis) {
                        assertion(temp_extents.at(n) == m_out_extents.at(n),
                                  "reinitialize(): Not all input matrices have matching non-axis dimensions.");
                    }
                }
            }
            out_axis_dim += mat.extent(m_axis);
        });
        m_out_extents.at(m_axis) = out_axis_dim;
        m_output_var.resize(m_out_extents);
    }

private:

    VariableF m_output_var; // associated with the default output port
    std::vector<int> m_out_extents;
    int m_axis;
};

}

#endif /* _CONCAT_NODE_H */
