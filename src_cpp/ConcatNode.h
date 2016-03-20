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
#include "Node.h"
#include <string>
#include <iostream>
#include "Utilities.h"
#include "Constants.h"

namespace kumozu {

  /**
   * A Node the concatenates all of the "input forward" matrices associated with its input ports.
   *
   * This node is allowed to have an arbitrary number of input ports. It will have 1 outptut port with the default name.
   * All matrices associated with
   * the input ports must be 2-dimensional with the same size for the second dimension (that is, the mini-batch size
   * must be the same for all input matrices).
   *
   * This node simply performs the concatenation of the "input forward" matrices over all of the input ports. The concatenation
   * is performed along the first dimension (non-minibatch dimension). The size of the first dimension of the "output forward"
   * matrix will be the sum of the sizes of the first dimension over all input ports.
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
  class ConcatNode : public Node {

  public:

    /**
     * Create a new instance with the specified node name and create the output port with default name.
     */
  ConcatNode(std::string name) :
    Node(name) {
      // Create the 1 output port.
      create_output_port(m_output_forward, m_output_backward, DEFAULT_OUTPUT_PORT_NAME); 
    }

    /**
     * Set output forward activations to the concatenation over all input forward activations.
     */
    virtual void forward_propagate() override {
      set_value(m_output_forward, 0.0f);
      int row_offset = 0;
      for_each_input_port_forward([&] (const MatrixF& mat) {
	  copy_from_submatrix(mat, m_output_forward, row_offset, 0);
	  row_offset += mat.extent(0);
	});
    }

    /**
     * Copy a submatrix from the "output backward" values into each "input backward" matrix.
     */
    virtual void back_propagate_deltas() override {
      MatrixF& deltas = get_output_backward();
      int row_offset = 0;
      for_each_input_port_backward([&] (MatrixF& mat) {
	  copy_to_submatrix(mat, deltas, row_offset, 0);
	  row_offset += mat.extent(0);
	});
    }

    /**
     * Check that all inputs have the same mini-batch size.
     */
    virtual void reinitialize() override {
      // First verify that all input ports are associated with matrices of the same dimensions.
      int out_rows = 0;
      m_minibatch_size = 0;
      for_each_input_port_forward([&] (const MatrixF& mat) {
	  if (mat.order() != 2) {
	    error_exit(get_name() + ": Error: input matrix found that is not 2-dimensional.");
	  }
	  if (m_minibatch_size == 0) {
	    m_minibatch_size = mat.extent(1);
	  } else {
	    if (m_minibatch_size != mat.extent(1)) {
	      error_exit(get_name() + ": Error: Not all input matrices have the same mini-batch size.");
	    }
	  }
	  out_rows += mat.extent(0);
	});
      m_output_forward.resize(out_rows, m_minibatch_size);
      m_output_backward.resize(out_rows, m_minibatch_size);
    }

  private:

    MatrixF m_output_forward; // associated with the default output port
    MatrixF m_output_backward; // associated with the default output port
    int m_minibatch_size {0};
  };

}

#endif /* _CONCAT_NODE_H */
