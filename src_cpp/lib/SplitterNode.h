#ifndef _SPLITTER_NODE_H
#define _SPLITTER_NODE_H
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
   * A Node with 1 input port and possibly several output ports that simply copies the input to each output port.
   *
   * This node is allowed to have an arbitrary number of output ports. It will have 1 inptut port with the default name.
   * Arbitrary-dimensional matrices are supported. The matrices associated
   * with the output port(s) will have the same dimensions as those associated with the input port.
   *
   * This node simply copies the contents of the "input forward" matrix into each of the output ports. The number of
   * output ports is specified in the constructor. The output ports will have the names "0", "1", "2", ...
   * The input port can have any name. If the user tries to add more than 1 input port, the program will exit with an error.
   *
   * Note: if pfn_mode is set to true, during the backward pass, the element-wise mean of the "output backward" activations
   * is computed and then copied into the "input backward" activations.
   *
   * Usage:
   *
   * Obtain an instance of this class, specifying the number of output ports to use in the constructor.
   * Be sure to connect exactly 1 input port (any name is allowed) before calling forward() for the first time.
   * Otherwise, the program will exit with an error.
   */
  class SplitterNode : public AtomicNode {

  public:

    /**
     * Create a new instance with the specified node name and create the output port with default name.
     *
     * @param output_port_count The number of output ports to create. The ports will be given names that correspond
     * to their index: "0", "1", "2", ..., "output_port_count-1".
     *
     * @param pfn_mode If the network uses gradients in the backward pass (i.e., the usual SGD methods), set to false.
     * If the network uses updated values of the outputs in the backward pass, such as a positive factor network (PFN),
     * set to true.
     */
  SplitterNode(int output_port_count, std::string name, bool pfn_mode=false) :
    AtomicNode{name},
      m_pfn_mode {pfn_mode}
      {
      for (int i = 0; i < output_port_count; ++i) {
	m_output_ports_forward.push_back(std::move(std::make_unique<MatrixF>()));
	m_output_ports_backward.push_back(std::move(std::make_unique<MatrixF>()));
	create_output_port(*m_output_ports_forward.back(), *m_output_ports_backward.back(), std::to_string(i));
      }
    }

    /**
     * Set output forward activations to the sum over all input forward activations.
     */
    virtual void forward_propagate() override {
      const MatrixF& input_forward_mat = get_input_port_data();
      for (size_t i = 0; i < m_output_ports_forward.size(); ++i) {
	MatrixF& out_forward_mat = *m_output_ports_forward.at(i);
    //copy_matrix(out_forward_mat, input_forward_mat);
    out_forward_mat = input_forward_mat;
      }
    }

    /**
     * 
     */
    virtual void back_propagate_activation_gradients() override {
      MatrixF& input_backward_mat = get_input_port_grad();
      const float scale_factor = 1.0f/static_cast<float>(get_output_port_count());
      set_value(input_backward_mat, 0.0f);
      for (size_t i = 0; i < m_output_ports_backward.size(); ++i) {
	const MatrixF& out_backward_mat = *m_output_ports_backward.at(i);
	element_wise_sum(input_backward_mat, input_backward_mat, out_backward_mat);
      }
      if (m_pfn_mode) {
	scale(input_backward_mat, input_backward_mat, scale_factor);
      }
    }

    /**
     * Resize output ports to be same size as the input port.
     */
    virtual void reinitialize() override {
      std::vector<int> input_extents = get_input_port_data().get_extents();
      // Set output matrices to have same extents as input matrix.
      for (size_t i = 0; i < m_output_ports_forward.size(); ++i) {
	MatrixF& out_forward_mat = *m_output_ports_forward.at(i);
	out_forward_mat.resize(input_extents);
	MatrixF& out_backward_mat = *m_output_ports_backward.at(i);
	out_backward_mat.resize(input_extents);
      }
    }

  private:

    std::vector<std::unique_ptr<MatrixF>> m_output_ports_forward;
    std::vector<std::unique_ptr<MatrixF>> m_output_ports_backward;
    bool m_pfn_mode;
  };

}

#endif /* _SPLITTER_NODE_H */
