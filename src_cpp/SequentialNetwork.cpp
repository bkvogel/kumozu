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

#include "SequentialNetwork.h"
#include "Utilities.h"
#include "MatrixIO.h"

using namespace std;

namespace kumozu {

  Layer& SequentialNetwork::get_layer(int n) {
    return m_layers[n].get();
  }

  const Layer& SequentialNetwork::get_layer(int n) const {
    return m_layers[n].get();
  }

  void SequentialNetwork::reinitialize(std::vector<int> input_extents) {
    // Nothing to do.
  }

  void SequentialNetwork::forward_propagate(const MatrixF& input_activations) {
    copy_weights_this_to_contained_layers();
    copy_bias_this_to_contained_layers();
    for (size_t i = 0; i < m_layers.size(); i++) {
      //std::cout << "Layer " << i << ": " << m_layers[i].get().get_name() << std::endl;
      if (i == 0) {
        //m_layers[i].get().forward_propagate(input_activations);
        m_layers[i].get().forward(input_activations);
      } else {
        //m_layers[i].get().forward_propagate(m_layers[i-1].get().get_output());
        m_layers[i].get().forward(m_layers[i-1].get().get_output());
      }
    }
    copy_weights_contained_layers_to_this();
    copy_bias_contained_layers_to_this();
  }

  void SequentialNetwork::back_propagate_paramater_gradients(const MatrixF& input_activations) {
    for (int i = static_cast<int>(m_layers.size())-1; i >= 0; i--) {
      //std::cout << "Layer " << i << ": " << m_layers[i].get().get_name() << std::endl;
      if (i == 0) {
        m_layers[i].get().back_propagate_paramater_gradients(input_activations);
      } else {
        m_layers[i].get().back_propagate_paramater_gradients(m_layers[i-1].get().get_output());
      }
    }
    copy_weights_gradients_contained_layers_to_this();
    copy_bias_gradients_contained_layers_to_this();
  }

  void SequentialNetwork::back_propagate_deltas(MatrixF& input_error) {
    for (int i = static_cast<int>(m_layers.size())-1; i >= 0; i--) {
      //std::cout << "Layer " << i << ": " << m_layers[i].get().get_name() << std::endl;
      if (i == 0) {
        m_layers[i].get().back_propagate_deltas(input_error);
      } else {
        m_layers[i].get().back_propagate_deltas(m_layers[i-1].get().get_output_deltas());
      }
    }
  }


  void SequentialNetwork::set_train_mode(bool is_train) {
    for (size_t i = 0; i < m_layers.size(); i++) {
      m_layers[i].get().set_train_mode(is_train);
    }
  }

  void SequentialNetwork::save_parameters(std::string name) const {
    if (!m_is_initialized) {
      std::cerr << m_layer_name <<  ": save_parameters() called before being initialized." << std::endl;
      exit(1);
    }
    save_matrix(m_W, name + "_" + m_layer_name + "_W.dat");
    save_matrix(m_bias, name + "_" + m_layer_name + "_bias.dat");
  }

  void SequentialNetwork::load_parameters(std::string name) {
    if (!m_is_initialized) {
      std::cerr << m_layer_name <<  ": load_parameters() called before being initialized." << std::endl;
      exit(1);
    }
    m_W = load_matrix(name + "_" + m_layer_name + "_W.dat");
    m_bias = load_matrix(name + "_" + m_layer_name + "_bias.dat");
    copy_weights_this_to_contained_layers();
    copy_bias_this_to_contained_layers();
  }


  void SequentialNetwork::copy_weights_contained_layers_to_this() {
    // Do an initial pass through all layers to compute the total number of weight
    // parameters.
    int total_size = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_W = m_layers[i].get().get_weights();
      total_size += temp_W.size();
    }
    // If this parameter count is different the size of the current m_W, then reinitialize.
    if (total_size != m_W.size()) {
      // Create 1-dim matrix of size total_size.
      cout << "Creating new m_W of size = " << total_size << endl;
      m_W = MatrixF(total_size);
    }
    // Now do another pass through all layers, this time copying the parameters into m_W.
    int cur_pos = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_W = m_layers[i].get().get_weights();
      for (int backing_index = 0; backing_index < temp_W.size(); ++backing_index) {
        m_W[cur_pos + backing_index] = temp_W[backing_index];
      }
      cur_pos += temp_W.size();
    }
  }

  void SequentialNetwork::copy_weights_this_to_contained_layers() {
    if (m_W.size() == 0) {
      return;
    }
    int cur_pos = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_W = m_layers[i].get().get_weights();
      for (int backing_index = 0; backing_index < temp_W.size(); ++backing_index) {
        temp_W[backing_index] = m_W[cur_pos + backing_index];
      }
      cur_pos += temp_W.size();
    }
  }

  void SequentialNetwork::copy_weights_gradients_contained_layers_to_this() {
    // Do an initial pass through all layers to compute the total number of weight gradient
    // parameters.
    int total_size = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_W_grad = m_layers[i].get().get_weight_gradient();
      total_size += temp_W_grad.size();
    }
    // If this parameter count is different the size of the current m_W_grad, then reinitialize.
    if (total_size != m_W_grad.size()) {
      // Create 1-dim matrix of size total_size.
      cout << m_layer_name << ": Creating new m_W_grad of size = " << total_size << endl;
      m_W_grad = MatrixF(total_size); // fixme: move to init()
    }
    // Now do another pass through all layers, this time copying the parameters into m_W_grad.
    int cur_pos = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_W_grad = m_layers[i].get().get_weight_gradient();
      for (int backing_index = 0; backing_index < temp_W_grad.size(); ++backing_index) {
        m_W_grad[cur_pos + backing_index] = temp_W_grad[backing_index];
      }
      cur_pos += temp_W_grad.size();
    }
  }

  void SequentialNetwork::copy_bias_contained_layers_to_this() {
    // Do an initial pass through all layers to compute the total number of bias
    // parameters.
    int total_size = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_bias = m_layers[i].get().get_bias();
      total_size += temp_bias.size();
    }
    // If this parameter count is different the size of the current m_bias, then reinitialize.
    if (total_size != m_bias.size()) {
      // Create 1-dim matrix of size total_size.
      cout << m_layer_name << ": Creating new m_bias of size = " << total_size << endl;
      m_bias = MatrixF(total_size);
    }
    // Now do another pass through all layers, this time copying the parameters into m_bias.
    int cur_pos = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_bias = m_layers[i].get().get_bias();
      for (int backing_index = 0; backing_index < temp_bias.size(); ++backing_index) {
        m_bias[cur_pos + backing_index] = temp_bias[backing_index];
      }
      cur_pos += temp_bias.size();
    }
  }

  void SequentialNetwork::copy_bias_this_to_contained_layers() {
    if (m_bias.size() == 0) {
      return;
    }
    int cur_pos = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_bias = m_layers[i].get().get_bias();
      for (int backing_index = 0; backing_index < temp_bias.size(); ++backing_index) {
        temp_bias[backing_index] = m_bias[cur_pos + backing_index];
      }
      cur_pos += temp_bias.size();
    }
  }

  void SequentialNetwork::copy_bias_gradients_contained_layers_to_this() {
    // Do an initial pass through all layers to compute the total number of weight gradient
    // parameters.
    int total_size = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_bias_grad = m_layers[i].get().get_bias_gradient();
      total_size += temp_bias_grad.size();
    }
    // If this parameter count is different the size of the current m_bias_grad, then reinitialize.
    if (total_size != m_bias_grad.size()) {
      // Create 1-dim matrix of size total_size.
      cout << m_layer_name << ": Creating new m_bias_grad of size = " << total_size << endl;
      m_bias_grad = MatrixF(total_size); // fixme: move to init()
    }
    // Now do another pass through all layers, this time copying the parameters into m_bias_grad.
    int cur_pos = 0;
    for (size_t i = 0; i < m_layers.size(); i++) {
      MatrixF& temp_bias_grad = m_layers[i].get().get_bias_gradient();
      for (int backing_index = 0; backing_index < temp_bias_grad.size(); ++backing_index) {
        m_bias_grad[cur_pos + backing_index] = temp_bias_grad[backing_index];
      }
      cur_pos += temp_bias_grad.size();
    }
  }

  void SequentialNetwork::print_paramater_stats() const {
    cout << get_name() << ": Parameter stats:" << endl;
    for (size_t i = 0; i < m_layers.size(); i++) {
      m_layers[i].get().print_paramater_stats();
    }
  }


}
