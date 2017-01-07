#ifndef _EXTRACTOR_NODE_H
#define _EXTRACTOR_NODE_H
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
#include <memory>
#include "Utilities.h"
#include "Constants.h"

namespace kumozu {

/**
   * A Node that extracts sub-matrices from the single "input forward" matrix and copies them to the (multiple) output ports.
   *
   * This node may have an arbitrary number of output ports. It will have 1 input port with the default name.
   * The "input forward" matrix associated with the input port must be 2-dimensional. The second dimension will
   * be interpreted as the mini-batch size.
   *
   * For each example in the input mini-batch (that is, for each column in "input forward"), the column is partitioned into
   * sub-columns and each sub-column is copied to "output forward" of a different output port. Thus, the number of sub-columns
   * will be equal to the number of output ports. The output ports will be given names from the default sequence: "0", "1", "2", etc.
   * Note that this operation can be viewed as "reverse concatenation" since after performing the forward pass, the "input forward"
   * matrix will be the concatenation of the various "output forward" matrices:
   * input_forward = [output_forward("0"), output_forward("1"), output_forward("2"), ... output_forward("N-1")] for N
   * sub-columns = N output ports. Note also that the total number of elements in the "input foward" matrix is equal to the
   * total number of elements over all of the "output forward" matrices for all of the output ports.
   *
   *
   * Usage:
   *
   * Obtain an instance of this class and call the create_input_port() functions of Node to create 1 input port.
   * The constructor will take the parameters necessary to split the input into the desired sub-matrices, and an
   * output port will automatically be created for each submatrix. Note that the order in which sub-matrices are
   * extracted (starting from row 0 of the input) is the same as the output port naming order "0", then "1", then "2" etc.
   */
class ExtractorNode : public AtomicNode {

public:

    /**
     * Create a new instance with the specified node name and create the output ports corresponding
     * to the number of sub-matrices that will be extracted from the input.
     *
     * @param partition_sizes A list of the sizes for each extracted sub-column. The sum of all values must
     * be equal to the number of rows in the input matrix. Otherwise, the program will exit with an
     * error message. The size of the list is equal to the number of output ports that will be created.
     *
     * @name Name for this node.
     */
    ExtractorNode(std::vector<int> partition_sizes, std::string name) :
        AtomicNode{name},
        m_partition_sizes {partition_sizes} {
        // Create the output ports.
        for (size_t i = 0; i < partition_sizes.size(); ++i) {
            //m_output_ports_forward.push_back(std::make_unique<MatrixF>());
            //m_output_ports_backward.push_back(std::make_unique<MatrixF>());
            m_output_ports_var.push_back(std::make_unique<VariableF>());
            //create_output_port(*m_output_ports_forward.back(), *m_output_ports_backward.back(), std::to_string(i));
            create_output_port(*m_output_ports_var.back(), std::to_string(i));
        }
    }

    /**
     * Extract sub-matrices from input and send one sub-matrix to each output port.
     */
    virtual void forward_propagate() override {
        int row_offset = 0;
        const MatrixF& input_forward_mat = get_input_port_data();
        for (size_t i = 0; i < m_output_ports_var.size(); ++i) {
            MatrixF& out_forward_mat = m_output_ports_var.at(i)->data;
            copy_large_to_small_mat_2d(out_forward_mat, input_forward_mat, row_offset, 0);
            row_offset += m_partition_sizes.at(i);
            /*
            MatrixF& out_forward_mat = *m_output_ports_forward.at(i);
            copy_large_to_small_mat_2d(out_forward_mat, input_forward_mat, row_offset, 0);
            row_offset += m_partition_sizes.at(i);
            */
        }
    }

    /**
     * Concatenate "output backward" matrices into "input backward."
     */
    virtual void back_propagate_activation_gradients() override {
        MatrixF& input_backward_mat = get_input_port_grad();
        int row_offset = 0;
        for (size_t i = 0; i < m_output_ports_var.size(); ++i) {
            const MatrixF& out_backward_mat = m_output_ports_var.at(i)->grad;
            copy_small_to_large_mat_2d(out_backward_mat, input_backward_mat, row_offset, 0);
            row_offset += m_partition_sizes.at(i);
            /*
            const MatrixF& out_backward_mat = *m_output_ports_backward.at(i);
            copy_small_to_large_mat_2d(out_backward_mat, input_backward_mat, row_offset, 0);
            row_offset += m_partition_sizes.at(i);
            */
        }
    }

    /**
     * Resize output ports and verify that the "partition_sizes" passed to the constructor are allowed.
     */
    virtual void reinitialize() override {
        std::vector<int> input_extents = get_input_port_data().get_extents();
        if (input_extents.size() != 2) {
            error_exit(get_name() + ": Error: input forward matrix is not 2-dimensional.");
        }
        const int minibatch_size = input_extents.at(1);
        const int input_row_count = input_extents.at(0);
        int partition_row_sum = 0;
        for (size_t i = 0; i < m_output_ports_var.size(); ++i) {
            m_output_ports_var.at(i)->resize(m_partition_sizes.at(i), minibatch_size);
            partition_row_sum += m_partition_sizes.at(i);
        }
        /*
        for (size_t i = 0; i < m_output_ports_forward.size(); ++i) {
            MatrixF& out_forward_mat = *m_output_ports_forward.at(i);
            out_forward_mat.resize(m_partition_sizes.at(i), minibatch_size);
            MatrixF& out_backward_mat = *m_output_ports_backward.at(i);
            out_backward_mat.resize(m_partition_sizes.at(i), minibatch_size);
            partition_row_sum += m_partition_sizes.at(i);
        }
        */
        if (partition_row_sum != input_row_count) {
            error_exit(get_name() + ": Error: Sum of partition_sizes supplied to constructor do not match the number of rows in input forward matrix.");
        }
    }

private:
    std::vector<int> m_partition_sizes;
    //MatrixF m_output_forward; // associated with the default output port
    //MatrixF m_output_backward; // associated with the default output port
    VariableF m_output_var; // associated with the default output port
    int m_minibatch_size {0};
    //std::vector<std::unique_ptr<MatrixF>> m_output_ports_forward;
    //std::vector<std::unique_ptr<MatrixF>> m_output_ports_backward;
    std::vector<std::unique_ptr<VariableF>> m_output_ports_var;
};

}

#endif /* _EXTRACTOR_NODE_H */
