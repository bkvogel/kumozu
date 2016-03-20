#ifndef _SLICE_UNROLLER_H
#define _SLICE_UNROLLER_H
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
   * Unroll a recurrent neural network (RNN) slice many times to create an unrolled network.
   *
   * Starting with a single slice of an RNN having type T, unroll the slice "num_slices" times.
   * All slices in the unrolled network other than the first slice will share parameters with the
   * first slice.
   */
  template <typename T>
    class SliceUnroller : public Node {

  public:

    /**
     * Create a new unrolled network where each slice has type T.
     *
     * @param num_slices The number of unrolled slices in the network.
     * @param "name" The name of the unrolled network.
     * @param args The parameters for the constructor for a single slice of type T, minus the
     * final string name parameter. This assumes that the final parameter of a slice represents the
     * string-valued name of the slice, which should be left out in args. For example, if the constructor
     * for a slice T has parameters (int rnn_dim, float dropout_val, string name), then "args" should consist
     * of (rnn_dim, dropout_val). The name parameter is not supplied because this constructor will supply it
     * to the constructor of T so that each slice index has a different name.
     */
    template<typename... Args>
      SliceUnroller(int num_slices, std::string name, Args&&... args) :
    Node(name)
    {
      for (int i = 0; i < num_slices; ++i) {
        std::string slice_name = "Slice: " + std::to_string(i);
        if (VERBOSE_MODE) {
          std::cout << "Creating: " << slice_name << std::endl;
        }
        m_slices.push_back(std::make_unique<T>(std::forward<Args>(args)..., slice_name));
        T& current_contained = *m_slices.back();
        const std::vector<std::string>& hidden_port_names = current_contained.get_hidden_port_names();
        if (i == 0) {
          if (VERBOSE_MODE) {
            std::cout << "Adding first rnn node." << std::endl;
          }
          // Connect hidden-state input ports of this container to corresponding input ports of current_contained.
          for (std::string hidden_port : hidden_port_names) {
            connect_input_to_contained_node(hidden_port, current_contained, hidden_port);
          }
        } else {
          // Make the contained node share paramters with the first of the contained nodes.
          if (VERBOSE_MODE) {
            std::cout << "Adding rnn node: " << i << std::endl;
          }
          Node& contained_0 = *m_slices.at(0);
          current_contained.set_shared(contained_0); // Make all slices other than the first slice have shared parameters with the first slice.

          // Connect hidden output "h_t" of previous contained node to input port "h_t_prev" of the current contained node.
          T& prev_contained = *m_slices.at(i-1);
          for (std::string hidden_port : hidden_port_names) {
            current_contained.create_input_port(prev_contained, hidden_port, hidden_port);
          }
        }
        // Connect input port "i" of the this container node to input port "x_t" of the current contained node.
        connect_input_to_contained_node(std::to_string(i), current_contained, "x_t");
        // Connect output port "y_t" of the current contained node to output port "i" of this container node.
        create_output_port(current_contained, "y_t", std::to_string(i));
        add_node(current_contained);
        if (i == num_slices-1) {
          // For the final slice, need to create an output port that sends the hidden state "h_t" as an output.
          for (std::string hidden_port : hidden_port_names) {
            create_output_port(current_contained, hidden_port,hidden_port);
          }
        }
      }
    }

    T& get_slice(int i) {
      return *m_slices.at(i);
    }

  private:

    std::vector<std::unique_ptr<T>> m_slices;

  };

}

#endif /* _SLICE_UNROLLER_H */
