#ifndef _SUBTRACTOR_NODE_H
#define _SUBTRACTOR_NODE_H
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
#include "Node.h"
#include <string>
#include <iostream>
#include "Utilities.h"
#include "Constants.h"

namespace kumozu {

  /**
   * A Node the computes the element-wise subtraction between the two "input forward" matrices associated with its input ports.
   *
   * This node is allowed to have exactly two input ports names "plus" and "minus." It will have 1 outptut port with the default name.. 
   * All matrices associated with
   * the input ports must have the same dimensions. Arbitrary-dimensional matrices are supported. The matrices associated
   * with the output port will have the same dimensions as those associated with the input.
   *
   * This node simply computes
   *
   * output <- "plus" - "minus"
   *
   * Usage:
   *
   * Obtain an instance of this class and call the create_input_port() functions of Node to create the two input ports with the
   * names "plus" and "minus." An instance of this class will expect these 2 input ports to exist by the time forward() is 
   * called for the first time. Otherwise, an error will occur.
   */
  class SubtractorNode : public Node {

  public:

    /**
     * Create a new instance with the specified node name and create the output port with default name.
     */
  SubtractorNode(std::string name) :
    Node(name),
      m_plus {"plus"},
      m_minus {"minus"}
      {
	// Create the 1 output port.
	create_output_port(m_output_forward, m_output_backward, DEFAULT_OUTPUT_PORT_NAME); 
    }

    /**
     * Set output forward activations to the sum over all input forward activations.
     */
    virtual void forward_propagate() {
      const MatrixF& plus_forward = get_input_port_forward(m_plus);
      const MatrixF& minus_forward = get_input_port_forward(m_minus);
      element_wise_difference(m_output_forward, plus_forward, minus_forward);
    }

    /**
     * 
     */
    virtual void back_propagate_deltas() {
      MatrixF& deltas = get_output_backward();
      MatrixF& plus_deltas = get_input_port_backward(m_plus);
      copy_matrix(plus_deltas, deltas);
      MatrixF& minus_deltas = get_input_port_backward(m_minus);
      copy_matrix(minus_deltas, deltas);
      scale(minus_deltas, minus_deltas, -1.0f);
    }

    /**
     * Check that all input ports are the same size.
     */
    virtual void reinitialize() {
      // First verify that all input ports are associated with matrices of the same dimensions.
      m_input_extents = get_input_port_forward(m_plus).get_extents();
      m_output_forward.resize(m_input_extents);
      m_output_backward.resize(m_input_extents);
      if (m_input_extents != get_input_port_forward(m_minus).get_extents()) {
	error_exit(get_name() + ": Error: plus and minux ports have different extents.");
      }
      if (get_input_port_count() != 2) {
	error_exit(get_name() + ": Error: Was expecting 2 input ports but found: " + std::to_string(get_input_port_count()));
      }
    }

  private:

    std::vector<int> m_input_extents; // Extents of each input port matrix.
    MatrixF m_output_forward; // associated with the default output port
    MatrixF m_output_backward; // associated with the default output port
    std::string m_plus;
    std::string m_minus;

  };

}

#endif /* _SUBTRACTOR_NODE_H */
