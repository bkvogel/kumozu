#ifndef _MEAN_NODE_H
#define _MEAN_NODE_H
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
   * A Node the computes the element-wise mean of all "input forward" matrices associated with its input ports.
   *
   * This node is allowed to have an arbitrary number of input ports. It will have 1 output port with the default name.
   * All matrices associated with
   * the input ports must have the same dimensions. Arbitrary-dimensional matrices are supported. The matrices associated
   * with the output port will have the same dimensions as those associated with the input.
   *
   * This node simply performs the element-wise mean of the "input forward" matrices over all of the input ports. The names
   * of the input ports can be arbitrary but must be distinct.
   *
   * Usage:
   *
   * Obtain an instance of this class and call the create_input_port() functions of Node to create as many input ports as desired.
   * Although the port names can be arbitrary, the user is required to specify the port name when more than 1 input port is used.
   * If only 1 input port is created, this node will simply function as an identity function that will pass the input through unchanged.
   */
class MeanNode : public AtomicNode {

public:

    /**
     * Create a new instance with the specified node name and create the output port with default name.
     *
     * Forward pass:
     *
     * The mean of all input_forward matrices is computed and produced on output_forward.
     *
     * Backward pass:
     *
     * PFN mode is on:
     * If pfn_mode is true and is_terminal_node is true, the mean value from the forward pass is copied into each of
     * the input_backward matrices. Otherwise, if pfn_mode is true and is_terminal_node is false, the output_backward
     * values are copied into each of the input_backward matrices.
     *
     * PFN mode is off:
     * If pfn_mode is false and is_terminal_node is true, the error gradient for each input_backward matrix is
     * computed to be the difference between the corresponding input_forward values and the mean values that were
     * computed during the forwarad pass.
     * If pfn_mode is false and is_terminal_node is false, a scaled version of output_backward is copied into each
     * input_backward matrix where the scale factor is 1/input_port_count.
     *
     * @param pfn_mode If the network uses gradients in the backward pass (i.e., the usual SGD methods), set to false.
     * If the network uses updated values of the outputs in the backward pass, such as a positive factor network (PFN),
     * set to true.
     *
     * @param is_terminal_node If true, assume that this node is not connected to any down-stream nodes and ignore the
     * values of "output_backward" during the backward data pass.
     */
    MeanNode(std::string name, bool pfn_mode=false, bool is_terminal_node=false) :
        AtomicNode(name),
        m_pfn_mode {pfn_mode},
        m_is_terminal_mode {is_terminal_node}
    {
        // Create the 1 output port.
        create_output_port(m_output_var, DEFAULT_OUTPUT_PORT_NAME);
    }

    /**
     * Set output forward activations to the mean over all input forward activations.
     */
    virtual void forward_propagate() override {
        set_value(m_output_var.data, 0.0f);
        for_each_input_port_data([&] (const MatrixF& mat) {
            element_wise_sum(m_output_var.data, m_output_var.data, mat);
        });
        // Scale sum by 1/<input port count>
        const float scale_factor = 1.0f/static_cast<float>(get_input_port_count());
        scale(m_output_var.data, m_output_var.data, scale_factor);
    }

    /**
     * Perform the backward data pass.
     */
    virtual void back_propagate_activation_gradients() override {
        MatrixF& deltas = get_output_grad();
        const MatrixF& mean_values = get_output_data();
        const float scale_factor = 1.0f/static_cast<float>(get_input_port_count());
        for_each_input_port([&] (const MatrixF& input_forward, MatrixF& input_backward) {
            if (m_pfn_mode) {
                // PFN mode is on.
                if (m_is_terminal_mode) {
                    //copy_matrix(input_backward, mean_values);
                    input_backward = mean_values;
                } else {
                    //copy_matrix(input_backward, deltas);
                    input_backward = deltas;
                }
            } else {
                // PFN mode is off.
                if (m_is_terminal_mode) {
                    // Error is difference between forward values and the mean value.
                    element_wise_difference(input_backward, input_forward, mean_values);
                } else {
                    //copy_matrix(input_backward, deltas);
                    input_backward = deltas;
                    scale(input_backward, scale_factor);
                }
            }
        });
    }

    /**
     * Perform the backward data pass.
     */
    /*
    virtual void back_propagate_deltas() override {
      MatrixF& deltas = get_output_backward();
      const float scale_factor = 1.0f/static_cast<float>(get_input_port_count());
      for_each_input_port_backward([&] (MatrixF& mat) {
      copy_matrix(mat, deltas);
      if (!m_pfn_mode) {
        scale(mat, mat, scale_factor); // todo: replace with scale() that takes only 2 arguments.
      }
    });
    }
    */
    
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
    }

private:

    std::vector<int> m_input_extents; // Extents of each input port matrix.
    //MatrixF m_output_forward; // associated with the default output port
    //MatrixF m_output_backward; // associated with the default output port
    VariableF m_output_var; // associated with the default output port
    bool m_pfn_mode;
    bool m_is_terminal_mode;
};

}

#endif /* _MEAN_NODE_H */
