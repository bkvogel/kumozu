#ifndef _SLICE_H
#define _SLICE_H
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
#include <map>
#include "Constants.h"
#include "Utilities.h"
#include "Node.h"
#include <memory>

namespace kumozu {

  /**
   * This is an abstract base class for a slice in a Recurrent Neural Network (RNN).
   *
   * An RNN is typically trained by first unrolling the network, which corresponds to
   * replicating an instance of Slice N times. Each of the N slices will have shared
   * parameters. Backpropagation can then be carried out in the usual way.
   *
   * In the Kumozu framework, an unrolled network corresponds to an instance of SliceUnroller.
   * In order to unroll a Slice N times, however, it is necessary to know both the number and names
   * of the hidden state ports of a Slice. Note that this information is needed in order to
   * make the inter-slice connections. This base class provides member functions that perform
   * the three basic operations when unrolling a Slice:
   *
   * - For the first unrolled slice, create input ports of the container (i.e., SliceUnroller) and connect
   * these input ports to the corresponding input ports of the first Slice. This operation is
   * performed by create_unrolled_container_hidden_input_ports().
   *
   * - For each pair of connected (adjacent) slices, connect the output ports of the previous slice to
   * the input ports of the next slice. This operation is performed by connect_hidden_state_to_next().
   *
   * - For the last unrolled slice, connect the output ports of the slice to the corresponding output
   * ports of the container (i.e., SliceUnroller). This operation is performed by
   * create_unrolled_container_hidden_output_ports().
   *
   *
   */
  class Slice : public Node {

  public:

  Slice(int rnn_dim, int output_dim, std::string name) :
    Node(name)
    {

    }

    // fixme: remove below functions. Reason: We only need a function that returns the list of hiddden port names
    // for a Slice. It is then easy for an UnrolledSlice to connect things up.

    /**
     * Create the hidden-state input ports for the container of this node.
     *
     * The container of this node corresponds to an unrolled RNN such that each replicated
     * slice is an instance of this class. Of course, the hidden state inputs to the first
     * slice will need to come from corresponding hidden state input ports of the container.
     * This function will create these input ports and connect them to the corresponding
     * input ports of this slice node.
     *
     * Usage:
     *
     * This function should be called by the unrolled container node once while it is processing
     * the first unrolled slice in the network.
     *
     * @param unrolled_container The container node that corresponds to an unrolled network
     * such that each replicated slice is an instance of this class.
     */
    virtual void create_unrolled_container_hidden_input_ports(Node& unrolled_container) = 0;

    /**
     * Connect the hidden state from this slice to the next slice.
     *
     * Make connections between the hidden state output ports of this slice and
     * the hidden state input ports of the next slice.
     *
     * Usage:
     *
     * This function should be called by the unrolled container to connect all adjacent
     * slices in the network.
     *
     * @param next The next slice in the unrolled RNN.
     */
    virtual void connect_hidden_state_to_next(Node& next) = 0;

    /**
     * Create the hidden-state output ports for the container of this node.
     *
     * The container of this node corresponds to an unrolled RNN such that each replicated
     * slice is an instance of this class. Of course, the hidden state outputs from the last
     * slice will need to be connected to output ports of the container.
     * This function will create these output ports and connect them to the corresponding
     * output ports of this slice node.
     *
     * Usage:
     *
     * This function should be called by the unrolled container node once while it is processing
     * the last unrolled slice in the network.
     *
     * @param unrolled_container The container node that corresponds to an unrolled network
     * such that each replicated slice is an instance of this class.
     */
    virtual void create_unrolled_container_hidden_output_ports(Node& unrolled_container) = 0;

    /**
     * Create the "input forward" or "input backward" matrices for the hidden states and return them in a map.
     *
     * Create each "input forward" or "input backward" matrix and return them in a map so that the matrix can be
     * looked up using its input port name.
     *
     * Note that since the "input forward" and "input backward" matrices are the same size for any given port, this
     * function can be called once to create the "input forward" matrices and then called again to create
     * the "input backward" matrices.
     *
     * @param minibatch_size mini-batch size.
     */
    virtual std::map<std::string, std::unique_ptr<MatrixF>> make_hidden_state_matrices(int minibatch_size) = 0;

  };

}

#endif /* _SLICE_H */
