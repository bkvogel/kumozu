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

#include "Node.h"
#include "Utilities.h"
#include "MatrixIO.h"
//#include <memory>

using namespace std;



namespace kumozu {

  void Node::forward() {
    // Check for connection errors
    //cout << get_name() << " : size of m_input_port_fan_out_map = " << m_input_port_fan_out_map.size() << endl;
    //for (const auto& x : m_input_port_fan_out_map) {
    //  if (x.second != 1) {
    //    error_exit("forward(): Error: Node: " + x.first + " has " + std::to_string(x.second) + " outgoing connections but should have 1.");
    //  }
    //}
    // and check for size changes in the input activations. If any of the input ports changed size,
    // call reinitialize().
    //cout << get_name() << " : size of m_input_port_forward_map = " << m_input_port_forward_map.size() << endl;
    bool extents_changed = false;
    for (const auto& x : m_input_port_forward_map) {
      auto& prev_extents = m_input_port_extents_map[x.first];
      const auto& cur_extents = x.second.get().get_extents();
      if (cur_extents != prev_extents) {
        extents_changed = true;
        prev_extents = cur_extents;
      }
    }
    if (extents_changed || (is_initialized() == false))  {
      if (VERBOSE_MODE) {
	std::cout << std::endl << "Initializing " << get_name() << ":" << std::endl;
      }
      // Re-connect input ports to internal nodes, if any.
      make_internal_input_port_connections();
      // Other re-initialization stuff (for subclasses).
      reinitialize();
    }
    forward_propagate();
    zero_parameter_gradients();
    zero_input_backward();
    set_initialized(true);
  }

  void Node::forward_propagate() {
    // If has at least one contained node:
    if (get_contained_node_count() != 0) {
      //copy params to contained nodes.
      copy_weights_this_to_contained_layers();
      copy_bias_this_to_contained_layers();
      for (int i = 0; i < get_contained_node_count(); ++i) {
	get_node(i).forward();
      }
      // copy params from contained nodes to this.
      copy_weights_contained_layers_to_this();
      copy_bias_contained_layers_to_this();
    }
  }

  void Node::back_propagate() {
    if (!is_initialized()) {
      error_exit("back_propagate(): " + get_name() + ": back_propagate() called before being initialized.");
    }
    back_propagate_deltas();
    back_propagate_paramater_gradients();
  }

  void Node::back_propagate_deltas() {
    for (int i = get_contained_node_count()-1; i >= 0; --i) {
      get_node(i).back_propagate_deltas();
    }
  }

  void Node::back_propagate_paramater_gradients() {
    // If has at least one contained node:
    if (get_contained_node_count() != 0) {
      for (int i = get_contained_node_count()-1; i >= 0; --i) {
        get_node(i).back_propagate_paramater_gradients();
      }
      // todo: might be better to just use a list of matrix references for the parameters, to avoid the copying.
      copy_weights_gradients_contained_layers_to_this(); 
      copy_bias_gradients_contained_layers_to_this();
    } 
  }

  void Node::zero_parameter_gradients() {
    // If has at least one contained node:
    if (get_contained_node_count() != 0) {
      for (int i = 0; i < get_contained_node_count(); ++i) {
        get_node(i).zero_parameter_gradients();
      }
      copy_weights_gradients_contained_layers_to_this(); 
      copy_bias_gradients_contained_layers_to_this();
    } else {
      set_value(get_weight_gradient(), 0.0f);
      set_value(get_bias_gradient(), 0.0f);
    }
  }

  void Node::zero_input_backward() {
    for (const auto& x : m_input_port_backward_map) {
      auto& cur_input_backward = x.second.get();
      set_value(cur_input_backward, 0.0f);
    }
    for (int i = 0; i < get_contained_node_count(); ++i) {
      get_node(i).zero_input_backward();
    }
  }

  void Node::create_input_port(const MatrixF& input_forward, MatrixF& input_backward, std::string input_name) {
    if (VERBOSE_MODE) {
      cout << get_name() << " : create_input_port(): with input port name: " << input_name << endl;
    }
    if (m_input_port_forward_map.find(input_name) != m_input_port_forward_map.end()) {
      m_input_port_forward_map.erase(input_name);
    }
    m_input_port_forward_map.emplace(input_name , std::cref(input_forward));
    if (m_input_port_backward_map.find(input_name) != m_input_port_backward_map.end()) {
      m_input_port_backward_map.erase(input_name);
    }
    m_input_port_backward_map.emplace(input_name, std::ref(input_backward));
    set_initialized(false);
  }

  void Node::create_input_port(const MatrixF& input_forward, MatrixF& input_backward) {
    create_input_port(input_forward, input_backward, DEFAULT_INPUT_PORT_NAME);
  }

  void Node::create_input_port(Node& parent, std::string parent_output, std::string input_name) {
    const MatrixF& parent_out_mat = parent.get_output_forward(parent_output);
    MatrixF& parent_out_deltas_mat = parent.get_output_backward(parent_output);
    create_input_port(parent_out_mat, parent_out_deltas_mat, input_name);
  }

  void Node::create_input_port_this_name(Node& parent, std::string input_name) {
    create_input_port(parent, DEFAULT_OUTPUT_PORT_NAME, input_name);
  }

  void Node::create_input_port_parent_name(Node& parent, std::string parent_output) {
    create_input_port(parent, parent_output, DEFAULT_INPUT_PORT_NAME);
  }

  void Node::connect_parent(Node& parent) {
    const MatrixF& parent_out_mat = parent.get_output_forward();
    MatrixF& parent_out_deltas_mat = parent.get_output_backward();
    create_input_port(parent_out_mat, parent_out_deltas_mat);
  }

  void Node::delete_input_port(std::string name) {
    auto it = m_input_port_forward_map.find(name);
    if (it != m_input_port_forward_map.end()) {
      m_input_port_forward_map.erase(it);
    }
    auto it2 = m_input_port_backward_map.find(name);
    if (it2 != m_input_port_backward_map.end()) {
      m_input_port_backward_map.erase(it2);
    }
    auto it3 = m_input_port_fan_out_map.find(name);
    if (it3 != m_input_port_fan_out_map.end()) {
      m_input_port_fan_out_map.erase(it3);      
    }
    auto it4 = m_input_port_extents_map.find(name);
    if (it4 != m_input_port_extents_map.end()) {
      m_input_port_extents_map.erase(it4);      
    }
    set_initialized(false);
  }

  void Node::delete_all_input_ports() {
    m_input_port_forward_map.clear();
    m_input_port_backward_map.clear();
    m_input_port_fan_out_map.clear();
    m_input_port_extents_map.clear();
    set_initialized(false);
  }

  void Node::create_output_port(const MatrixF& output_forward, MatrixF& output_backward, std::string output_name) {
    if (m_output_port_forward_map.find(output_name) != m_output_port_forward_map.end()) {
      error_exit("create_output_port(): Error: " + output_name + " is already an output.");
    }
    m_output_port_forward_map.emplace(output_name , std::cref(output_forward));
    m_output_port_backward_map.emplace(output_name, std::ref(output_backward));
    set_initialized(false);
  }

  void Node::create_output_port(Node& contained, std::string contained_output, std::string output_name) {
    const MatrixF& contained_out_mat = contained.get_output_forward(contained_output);
    MatrixF& contained_out_deltas_mat = contained.get_output_backward(contained_output);
    create_output_port(contained_out_mat, contained_out_deltas_mat, output_name); 
  }

  void Node::create_output_port_this_name(Node& contained, std::string output_name) {
    create_output_port(contained, DEFAULT_OUTPUT_PORT_NAME, output_name);
  }

  void Node::create_output_port_contained_name(Node& contained, std::string output_name) {
    create_output_port(contained, output_name, DEFAULT_OUTPUT_PORT_NAME);
  }  

  void Node::create_output_port(Node& contained) {
    create_output_port(contained, DEFAULT_OUTPUT_PORT_NAME, DEFAULT_OUTPUT_PORT_NAME);
  }

  const MatrixF& Node::get_output_forward(std::string name) const {
    auto it = m_output_port_forward_map.find(name);
    if (it == m_output_port_forward_map.end()) {
      error_exit("get_output_forward(): Error: " + name + " is not an output.");
    }
    return it->second;
  }

  const MatrixF& Node::get_output_forward() const {
    if (get_output_port_count() != 1) {
      error_exit("get_output_forward(): " + get_name() + " should have 1 output port but instead has " + std::to_string(get_output_port_count()) + " ports.");
    }
    return m_output_port_forward_map.begin()->second;
  }

  void Node::delete_output_port(std::string name) {
    auto it = m_output_port_forward_map.find(name);
    if (it == m_output_port_forward_map.end()) {
      error_exit("delete_output_port(): Error: " + name + " is not an existing output.");
    }
    m_output_port_forward_map.erase(it);

    auto it2 = m_output_port_backward_map.find(name);
    if (it2 == m_output_port_backward_map.end()) {
      error_exit("delete_output_port(): Error: " + name + " is not an existing output.");
    }
    m_output_port_backward_map.erase(it2);

    auto it3 = m_output_port_fan_out_map.find(name);
    if (it3 == m_output_port_fan_out_map.end()) {
      error_exit("delete_output_port(): Error: " + name + " is not an existing output.");
    }
    m_output_port_fan_out_map.erase(it3);
    set_initialized(false);
  }

  void Node::delete_all_output_ports() {
    m_output_port_forward_map.clear();
    m_output_port_backward_map.clear();
    m_output_port_fan_out_map.clear();
    set_initialized(false);
  }

  const MatrixF& Node::get_output_backward(std::string name) const {
    auto it = m_output_port_backward_map.find(name);
    if (it == m_output_port_backward_map.end()) {
      error_exit("get_output_forward(): Error: " + name + " is not an output.");
    }
    return it->second;
  }

  MatrixF& Node::get_output_backward(std::string name) {
    auto it = m_output_port_backward_map.find(name);
    if (it == m_output_port_backward_map.end()) {
      error_exit("get_output_forward(): Error: " + name + " is not an output.");
    }
    return it->second;
  }

  const MatrixF& Node::get_output_backward() const {
    if (get_output_port_count() != 1) {
      error_exit("get_output_backward(): " + get_name() + " should have 1 output port but instead has "
                 + std::to_string(get_output_port_count()) + " ports.");
    }
    return m_output_port_backward_map.begin()->second;
  }

  MatrixF& Node::get_output_backward() {
    if (get_output_port_count() != 1) {
      error_exit("get_output_backward(): " + get_name() + " should have 1 output port but instead has "
                 + std::to_string(get_output_port_count()) + " ports.");
    }
    return m_output_port_backward_map.begin()->second;
  }

  std::vector<int> Node::get_output_extents(std::string name) const {
    return get_output_forward(name).get_extents();
  }


  const MatrixF& Node::get_input_port_forward(std::string name) const {
    auto it = m_input_port_forward_map.find(name);
    if (it == m_input_port_forward_map.end()) {
      error_exit("get_input_port_forward(): Error: " + name + " is not an input.");
    }
    return it->second;
  }

  const MatrixF& Node::get_input_port_forward() const {
    if (get_input_port_count() != 1) {
      error_exit("get_input_port_forward(): " + get_name() +
                 " should have 1 input port but instead has " + std::to_string(get_input_port_count()) + " ports.");
    }
    return m_input_port_forward_map.begin()->second;
  }

  const MatrixF& Node::get_input_port_backward(std::string name) const {
    auto it = m_input_port_backward_map.find(name);
    if (it == m_input_port_backward_map.end()) {
      error_exit("get_input_port_backward(): Error: " + name + " is not an input.");
    }
    return it->second;
  }

  MatrixF& Node::get_input_port_backward(std::string name) {
    auto it = m_input_port_backward_map.find(name);
    if (it == m_input_port_backward_map.end()) {
      error_exit("get_input_port_backward(): Error: " + name + " is not an input.");
    }
    return it->second;
  }

  MatrixF& Node::get_input_port_backward() {
    if (get_input_port_count() != 1) {
      error_exit("get_input_port_backward(): " + get_name() +
                 " should have 1 input port but instead has " + std::to_string(get_input_port_count()) + " ports.");
    }
    return m_input_port_backward_map.begin()->second;
  }

  const MatrixF& Node::get_input_port_backward() const {
    if (get_input_port_count() != 1) {
      error_exit("get_input_port_backward(): " + get_name() +
                 " should have 1 input port but instead has " + std::to_string(get_input_port_count()) + " ports.");
    }
    return m_input_port_backward_map.begin()->second;
  }


  void Node::add_node(Node& node) {
    m_contained_nodes.push_back(node);
    set_initialized(false);
  }

  Node& Node::get_node(int n) {
    return m_contained_nodes[n].get();
  }

  const Node& Node::get_node(int n) const {
    return m_contained_nodes[n].get();
  }

  int Node::get_contained_node_count() const {
    return static_cast<int>(m_contained_nodes.size());
  }

  const bool Node::is_composite() const {
    if (get_contained_node_count() != 0) {
      return true;
    } else {
      return false;
    }
  }

  void Node::connect_input_to_contained_node(std::string input_name, Node& contained, std::string contained_input) {
    m_input_to_internal_connections.push_back(input_to_contained_info(input_name, contained, contained_input));
    set_initialized(false);
  }

  void Node::connect_input_to_contained_node(std::string input_name, Node& contained) {
    connect_input_to_contained_node(input_name, contained, DEFAULT_INPUT_PORT_NAME);
  }

  void Node::connect_input_to_contained_node(Node& contained) {
    connect_input_to_contained_node(DEFAULT_INPUT_PORT_NAME, contained, DEFAULT_INPUT_PORT_NAME);
  }


  void Node::set_train_mode(bool is_train) {
    m_is_train = is_train;
    for (int i = 0; i < get_contained_node_count(); ++i) {
      get_node(i).set_train_mode(is_train);
    }
  }

  void Node::print_paramater_stats() const {
    if (!is_initialized()) {
      std::cerr << get_name() <<  ": print_paramater_stats() called before being initialized." << std::endl;
      exit(1);
    }
    if (get_weights().size() > 0) {
      print_stats(get_weights(), get_name() + " : weights");
    }
    if (get_bias().size() > 0) {
      print_stats(get_bias(), get_name() + " : bias");
    }
    if (get_weight_gradient().size() > 0) {
      print_stats(get_weight_gradient(), get_name() + " : weight gradients");
    }
    if (get_bias_gradient().size() > 0) {
      print_stats(get_bias_gradient(), get_name() + " : bias gradients");
    }
    if(get_output_forward().size() > 0) {
      print_stats(get_output_forward(), get_name() + " : output activations");
    }
    if (get_output_backward().size() > 0) {
      print_stats(get_output_backward(), get_name() + " : output activations deltas");
    }
    if (get_contained_node_count() != 0) {
      cout << get_name() << ": Parameter stats for contained nodes:" << endl;
      for (int i = 0; i < get_contained_node_count(); ++i) {
        get_node(i).print_paramater_stats();
      }
    }
  }

  void Node::save_parameters(std::string name) const {
    if (!is_initialized()) {
      std::cerr << get_name() <<  ": save_parameters() called before being initialized." << std::endl;
      exit(1);
    }
    save_matrix(m_W_ref, name + "_" + get_name() + "_W.dat");
    save_matrix(m_bias, name + "_" + get_name() + "_bias.dat");
  }

  void Node::load_parameters(std::string name) {
    //m_W_ref.get() = load_matrix(name + "_" + get_name() + "_W.dat");
    get_weights() = load_matrix(name + "_" + get_name() + "_W.dat");
    m_bias = load_matrix(name + "_" + get_name() + "_bias.dat");
    if (get_contained_node_count() != 0) {
      copy_weights_this_to_contained_layers();
      copy_bias_this_to_contained_layers();
    }
  }

  void Node::check_jacobian_weights(std::map<std::string, std::vector<int>> input_port_extents_map) {
    cout << get_name() << ": Checking Jacobian for weights..." << endl;
    delete_all_input_ports();
    // We need to create a Matrix for each input port in the map and then connect it to this Node as a new
    // input port.
    vector<MatrixF> input_forward_list;
    vector<MatrixF> input_backward_list;
    
    input_forward_list.reserve(input_port_extents_map.size());
    input_backward_list.reserve(input_port_extents_map.size());
    // Alocate whole vector to be safe. Even with the reserve, it might decide to realocate while adding elements?
    for (const auto& x: input_port_extents_map) {
      auto& input_extents = x.second;
      input_forward_list.push_back(MatrixF(input_extents));
      input_backward_list.push_back(MatrixF(input_extents));
    }
    int total_size = 0;
    int i = 0;
    for (const auto& x: input_port_extents_map) {
      create_input_port(input_forward_list.at(i), input_backward_list.at(i), x.first);
      total_size += input_forward_list.at(i).size();
      ++i;
    }

    // Create random input activations for the layer.
    MatrixF input_forward_flat(total_size);
    randomize_uniform(input_forward_flat, 0.0f, 1.0f);
    MatrixF input_backward_flat(total_size);
    randomize_uniform(input_backward_flat, 0.0f, 1.0f);

    // Copy from flat input matrix into the list of input matrices:
    copy_flat_matrix_to_list(input_forward_list, input_forward_flat);
    copy_flat_matrix_to_list(input_backward_list, input_backward_flat);

    forward(); // Initialize node.
    // Now the output activations have been initialized to the correct sizes.

    MatrixF output_forward_flat;
    copy_individual_to_flat_output_forward(output_forward_flat);

    // Now check Jacobian for weights:
    // Size will be total_output_dim x total_weights_dim =
    // (dim_output*minibatch_size) x total_weights_dim
    const int total_output_dim = output_forward_flat.size();
    const int total_weights_dim = get_weights().size();
    // This will contain the Jacobian computed using finite differences method.
    MatrixF numerical_jacobian_weights(total_output_dim, total_weights_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(numerical_jacobian_weights, 0.0f, 1.0f);

    // This will contain the Jacobian computed using the back-prop method of the class
    MatrixF backprop_jacobian_weights(total_output_dim, total_weights_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(backprop_jacobian_weights, 0.0f, 1.0f);

    // Now compute the numerical Jacobian:
    // This will be computed one column at a time.
    MatrixF& W = get_weights();

    for (int j=0; j < W.size(); ++j) {
      // j is column index int Jacobian matrix.
      float orig = W[j];
      W[j] += m_epsilon;
      // Now compute output of layer -> output_forward
      forward();
      copy_individual_to_flat_output_forward(output_forward_flat);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_forward_flat.size(); ++i) {
        numerical_jacobian_weights(i,j) = output_forward_flat[i];
      }
      W[j] = orig - m_epsilon;
      // Now compute output of layer -> output_forward
      forward();
      copy_individual_to_flat_output_forward(output_forward_flat);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_forward_flat.size(); ++i) {
        numerical_jacobian_weights(i,j) -= output_forward_flat[i];
        numerical_jacobian_weights(i,j) /= 2*m_epsilon;
      }
      // Put back original value.
      W[j] = orig;
    }
    // Now compute the Jacobian using the backprop function.
    MatrixF output_backwards_flat(output_forward_flat.size());
    MatrixF& grad_W = get_weight_gradient();
    set_value(output_backwards_flat, 0.0f);
    for (int i=0; i < output_backwards_flat.size(); ++i) {
      output_backwards_flat[i] = 1.0f;
      // Now if we perform backprop, the result should be the same is row i of Jacobian.
      copy_flat_output_backward_to_individual(output_backwards_flat);
      zero_parameter_gradients();
      zero_input_backward();
      back_propagate();
      for (int j=0; j < W.size(); ++j) {
        backprop_jacobian_weights(i,j) = grad_W[j];
      }
      output_backwards_flat[i] = 0.0f;
    }

    const float relative_error_score = relative_error(numerical_jacobian_weights, backprop_jacobian_weights);
    std::cout << "numerical-back-prop gradients relative error = " << relative_error_score << std::endl;
    //cout << "numerical_jacobian_weights = " << endl << numerical_jacobian_weights << endl;
    //cout << "backprop_jacobian_weights = " << endl << backprop_jacobian_weights << endl;
    assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);
    delete_all_input_ports();
    cout << "PASSED" << endl;
  }

  void Node::check_jacobian_bias(std::map<std::string, std::vector<int>> input_port_extents_map) {
    cout << get_name() << ": Checking Jacobian for bias..." << endl;
    delete_all_input_ports();
    // We need to create a Matrix for each input port in the map and then connect it to this Node as a new
    // input port.
    vector<MatrixF> input_forward_list;
    vector<MatrixF> input_backward_list;
    input_forward_list.reserve(input_port_extents_map.size());
    input_backward_list.reserve(input_port_extents_map.size());
    // Alocate whole vector to be safe. Even with the reserve, it might decide to realocate while adding elements?
    for (const auto& x: input_port_extents_map) {
      auto& input_extents = x.second;
      input_forward_list.push_back(MatrixF(input_extents));
      input_backward_list.push_back(MatrixF(input_extents));
    }
    int total_size = 0;
    int i = 0;
    for (const auto& x: input_port_extents_map) {
      create_input_port(input_forward_list.at(i), input_backward_list.at(i), x.first);
      total_size += input_forward_list.at(i).size();
      ++i;
    }

    // Create random input activations for the layer.
    MatrixF input_forward_flat(total_size);
    randomize_uniform(input_forward_flat, 0.0f, 1.0f);
    MatrixF input_backward_flat(total_size);
    randomize_uniform(input_backward_flat, 0.0f, 1.0f);

    // Copy from flat input matrix into the list of input matrices:
    copy_flat_matrix_to_list(input_forward_list, input_forward_flat);
    copy_flat_matrix_to_list(input_backward_list, input_backward_flat);

    forward(); // Initialize node.
    // Now the output activations have been initialized to the correct sizes.

    MatrixF output_forward_flat;
    copy_individual_to_flat_output_forward(output_forward_flat);

    // Now check Jacobian for bias:
    // Size will be total_output_dim x total_bias_dim =
    // (dim_output*minibatch_size) x total_bias_dim
    const int total_output_dim = output_forward_flat.size();
    const int total_bias_dim = get_bias().size();
    // This will contain the Jacobian computed using finite differences method.
    MatrixF numerical_jacobian_bias(total_output_dim, total_bias_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(numerical_jacobian_bias, 0.0f, 1.0f);

    // This will contain the Jacobian computed using the back-prop method of the class
    MatrixF backprop_jacobian_bias(total_output_dim, total_bias_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(backprop_jacobian_bias, 0.0f, 1.0f);

    // Now compute the numerical Jacobian:
    // This will be computed one column at a time.
    MatrixF& bias = get_bias();

    //const MatrixF& output_forward = get_output_forward();
    for (int j=0; j < bias.size(); ++j) {
      // j is column index int Jacobian matrix.
      float orig = bias[j];
      bias[j] += m_epsilon;
      // Now compute output of layer -> output_forward
      forward();
      copy_individual_to_flat_output_forward(output_forward_flat);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_forward_flat.size(); ++i) {
        numerical_jacobian_bias(i,j) = output_forward_flat[i];
      }
      bias[j] = orig - m_epsilon;
      // Now compute output of layer -> output_forward
      forward();
      copy_individual_to_flat_output_forward(output_forward_flat);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_forward_flat.size(); ++i) {
        numerical_jacobian_bias(i,j) -= output_forward_flat[i];
        numerical_jacobian_bias(i,j) /= 2*m_epsilon;
      }
      // Put back original value.
      bias[j] = orig;
    }
    // Now compute the Jacobian using the backprop function.
    MatrixF output_backwards_flat(output_forward_flat.size());
    MatrixF& grad_bias = get_bias_gradient();
    set_value(output_backwards_flat, 0.0f);
    for (int i=0; i < output_backwards_flat.size(); ++i) {
      output_backwards_flat[i] = 1.0f;
      // Now if we perform backprop, the result should be the same is row i of Jacobian.
      copy_flat_output_backward_to_individual(output_backwards_flat);
      zero_parameter_gradients();
      zero_input_backward();
      back_propagate();
      for (int j=0; j < bias.size(); ++j) {
        backprop_jacobian_bias(i,j) = grad_bias[j];
      }
      output_backwards_flat[i] = 0.0f;
    }

    const float relative_error_score = relative_error(numerical_jacobian_bias, backprop_jacobian_bias);
    std::cout << "numerical-back-prop gradients relative error = " << relative_error_score << std::endl;
    //cout << "numerical_jacobian_bias = " << endl << numerical_jacobian_bias << endl;
    //cout << "backprop_jacobian_bias = " << endl << backprop_jacobian_bias << endl;
    assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);
    delete_all_input_ports();
    cout << "PASSED" << endl;
  }

  void Node::check_jacobian_input_backward(std::map<std::string, std::vector<int>> input_port_extents_map) {
    cout << get_name() << ": Checking Jacobian for input error gradients..." << endl;
    delete_all_input_ports();
    // We need to create a Matrix for each input port in the map and then connect it to this Node as a new
    // input port.
    vector<MatrixF> input_forward_list;
    vector<MatrixF> input_backward_list;
    input_forward_list.reserve(input_port_extents_map.size());
    input_backward_list.reserve(input_port_extents_map.size());
    // Alocate whole vector to be safe. Even with the reserve, it might decide to realocate while adding elements?
    for (const auto& x: input_port_extents_map) {
      auto& input_extents = x.second;
      input_forward_list.push_back(MatrixF(input_extents));
      input_backward_list.push_back(MatrixF(input_extents));
    }
    int total_size = 0;
    int i = 0;
    for (const auto& x: input_port_extents_map) {
      create_input_port(input_forward_list.at(i), input_backward_list.at(i), x.first);
      total_size += input_forward_list.at(i).size();
      ++i;
    }

    // Create random input activations for the layer.
    MatrixF input_forward_flat(total_size);
    randomize_uniform(input_forward_flat, 0.0f, 1.0f);
    MatrixF input_backward_flat(total_size);
    randomize_uniform(input_backward_flat, 0.0f, 1.0f);

    // Copy from flat input matrix into the list of input matrices:
    copy_flat_matrix_to_list(input_forward_list, input_forward_flat);
    copy_flat_matrix_to_list(input_backward_list, input_backward_flat);

    forward(); // Initialize node.
    // Now the output activations have been initialized to the correct sizes.

    MatrixF output_forward_flat;
    copy_individual_to_flat_output_forward(output_forward_flat);



    // Size will be total_output_dim x total_input_dim =
    // (dim_output*minibatch_size) x total_input_dim
    const int total_output_dim = output_forward_flat.size();
    if (total_output_dim == 0) {
      error_exit(get_name() + ": Checking Jacobian for input error gradients: total_output_dim is 0!");
    }
    const int total_input_dim = input_forward_flat.size();
    // This will contain the Jacobian computed using finite differences method.
    MatrixF numerical_jacobian_input(total_output_dim, total_input_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(numerical_jacobian_input, 0.0f, 1.0f);

    // This will contain the Jacobian computed using the back-prop method of the class
    MatrixF backprop_jacobian_input(total_output_dim, total_input_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(backprop_jacobian_input, 0.0f, 1.0f);

    // Now compute the numerical Jacobian:
    // This will be computed one column at a time.
    for (int j=0; j < input_forward_flat.size(); ++j) {
      // j is column index int Jacobian matrix.
      float orig = input_forward_flat[j];
      input_forward_flat[j] += m_epsilon;
      // Now compute output of layer -> output_forward
      copy_flat_matrix_to_list(input_forward_list, input_forward_flat);
      forward();
      copy_individual_to_flat_output_forward(output_forward_flat);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_forward_flat.size(); ++i) {
        numerical_jacobian_input(i,j) = output_forward_flat[i];
      }
      input_forward_flat[j] = orig - m_epsilon;
      // Now compute output of layer -> output_forward
      copy_flat_matrix_to_list(input_forward_list, input_forward_flat);
      forward();
      copy_individual_to_flat_output_forward(output_forward_flat);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_forward_flat.size(); ++i) {
        numerical_jacobian_input(i,j) -= output_forward_flat[i];
        numerical_jacobian_input(i,j) /= 2*m_epsilon;
      }
      // Put back original value.
      input_forward_flat[j] = orig;
    }
    // Now compute the Jacobian using the backprop function.
    MatrixF output_backwards_flat(output_forward_flat.size());

    set_value(output_backwards_flat, 0.0f);
    for (int i=0; i < output_backwards_flat.size(); ++i) {
      output_backwards_flat[i] = 1.0f;
      // Now if we perform backprop, the result should be the same is row i of Jacobian.
      copy_flat_output_backward_to_individual(output_backwards_flat);
      zero_parameter_gradients();
      zero_input_backward();
      back_propagate();
      copy_list_to_flat_matrix(input_backward_list, input_backward_flat);

      for (int j=0; j < input_forward_flat.size(); ++j) {
        backprop_jacobian_input(i,j) = input_backward_flat[j];
      }
      output_backwards_flat[i] = 0.0f;
    }

    const float relative_error_score = relative_error(numerical_jacobian_input, backprop_jacobian_input);
    std::cout << "check_jacobian_input_backward(): relative error = " << relative_error_score << std::endl;
    //cout << "numerical_jacobian_input = " << endl << numerical_jacobian_input << endl;
    //cout << "backprop_jacobian_input = " << endl << backprop_jacobian_input << endl;
    if (relative_error_score == 0) {
      error_exit("check_jacobian_input_backward(): Error: relative error must be greater than 0.");
    }
    assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);
    delete_all_input_ports();
    cout << "PASSED" << endl;
  }

  void Node::check_jacobian_weights(std::vector<int> input_extents) {
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map[DEFAULT_INPUT_PORT_NAME] = input_extents;
    check_jacobian_weights(input_port_extents_map);
  }

  void Node::check_jacobian_bias(std::vector<int> input_extents) {
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map[DEFAULT_INPUT_PORT_NAME] = input_extents;
    check_jacobian_bias(input_port_extents_map);
  }

  void Node::check_jacobian_input_backward(std::vector<int> input_extents) {
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map[DEFAULT_INPUT_PORT_NAME] = input_extents;
    check_jacobian_input_backward(input_port_extents_map);
  }


  void Node::copy_flat_output_backward_to_individual(const MatrixF& flat_output_backward) {
    // Do an initial pass through all output matrices to determine the total number of elements.
    int total_size = 0;
    //
    for (auto& x: m_output_port_backward_map) {
      auto& temp_mat = x.second.get();
      total_size += temp_mat.size();
    }
    // If the element count is different the size of the current flat_mat, then exit with error.
    if (total_size != flat_output_backward.size()) {
      error_exit("copy_flat_output_backward_to_many(): Supplied matrix list has different element count than supplied flat matrix.");
    }
    int cur_pos = 0;
    for (auto& x: m_output_port_backward_map) {
      auto& temp_mat = x.second.get();
      for (int backing_index = 0; backing_index < temp_mat.size(); ++backing_index) {
        temp_mat[backing_index] = flat_output_backward[cur_pos + backing_index];
      }
      cur_pos += temp_mat.size();
    }
  }

  void Node::copy_individual_to_flat_output_forward(MatrixF& flat_output_forward) {
    // Do an initial pass through all output matrices to determine the total number of elements.
    int total_size = 0;
    //
    for (auto& x: m_output_port_forward_map) {
      auto& temp_mat = x.second.get();
      total_size += temp_mat.size();
    }
    // If the element count is different the size of the current flat_mat, then exit with error.
    if (total_size != flat_output_forward.size()) {
      if (VERBOSE_MODE) {
	std::cout << "Resizing flat_mat to size = " << total_size << std::endl;
      }
      flat_output_forward.resize(total_size);
    }
    int cur_pos = 0;
    for (auto& x: m_output_port_forward_map) {
      auto& temp_mat = x.second.get();
      for (int backing_index = 0; backing_index < temp_mat.size(); ++backing_index) {
        flat_output_forward[cur_pos + backing_index] = temp_mat[backing_index];
      }
      cur_pos += temp_mat.size();
    }
  }

  // fixme: rename "layers" to "nodes" below in func name.
  void Node::copy_weights_contained_layers_to_this() {
    if (!is_composite()) {
      return;
    }
    // Do an initial pass through all layers to compute the total number of weight
    // parameters. Skip over nodes with shared parameters.
    int total_size = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_W = get_node(i).get_weights();
	total_size += temp_W.size();
      }
    }
    // If this parameter count is different the size of the current m_W, then reinitialize.
    if (total_size != get_weights().size()) {
      // Create 1-dim matrix of size total_size.
      if (VERBOSE_MODE) {
	cout << get_name() << ": Resizing weights matrix to size = " << total_size << endl;
      }
      get_weights().resize(total_size);
    }
    // Now do another pass through all layers, this time copying the parameters into m_W.
    MatrixF& W = get_weights();
    int cur_pos = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_W = get_node(i).get_weights();
	for (int backing_index = 0; backing_index < temp_W.size(); ++backing_index) {
	  W[cur_pos + backing_index] = temp_W[backing_index];
	}
	cur_pos += temp_W.size();
      }
    }
  }

  void Node::copy_weights_this_to_contained_layers() {
    if (get_weights().size() == 0) {
      return;
    }
    const MatrixF& W = get_weights();
    int cur_pos = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_W = get_node(i).get_weights();
	for (int backing_index = 0; backing_index < temp_W.size(); ++backing_index) {
	  temp_W[backing_index] = W[cur_pos + backing_index];
	}
	cur_pos += temp_W.size();
      }
    }
  }

  void Node::copy_weights_gradients_contained_layers_to_this() {
    if (!is_composite()) {
      return;
    }
    // Do an initial pass through all layers to compute the total number of weight gradient
    // parameters.
    int total_size = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_W_grad = get_node(i).get_weight_gradient();
	total_size += temp_W_grad.size();
      }
    }
    // If this parameter count is different the size of the current m_W_grad, then reinitialize.
    if (total_size != m_W_grad.size()) {
      // Create 1-dim matrix of size total_size.
      if (VERBOSE_MODE) {
	cout << get_name() << ": Resizing weight gradients to size = " << total_size << endl;
      }
      get_weight_gradient().resize(total_size);
    }
    // Now do another pass through all layers, this time copying the parameters into m_W_grad.
    int cur_pos = 0;
    MatrixF& grad_W = get_weight_gradient();
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_W_grad = get_node(i).get_weight_gradient();
	for (int backing_index = 0; backing_index < temp_W_grad.size(); ++backing_index) {
	  grad_W[cur_pos + backing_index] = temp_W_grad[backing_index];
	}
	cur_pos += temp_W_grad.size();
      }
    }
  }

  void Node::copy_bias_contained_layers_to_this() {
    if (!is_composite()) {
      return;
    }
    // Do an initial pass through all layers to compute the total number of bias
    // parameters.
    int total_size = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_bias = get_node(i).get_bias();
	total_size += temp_bias.size();
      }
    }
    // If this parameter count is different the size of the current m_bias, then reinitialize.
    if (total_size != m_bias.size()) {
      // Create 1-dim matrix of size total_size.
      if (VERBOSE_MODE) {
	cout << get_name() << ": Resizing m_bias to size = " << total_size << endl;
      }
      m_bias.resize(total_size);
    }
    // Now do another pass through all layers, this time copying the parameters into m_bias.
    int cur_pos = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_bias = get_node(i).get_bias();
	for (int backing_index = 0; backing_index < temp_bias.size(); ++backing_index) {
	  m_bias[cur_pos + backing_index] = temp_bias[backing_index];
	}
	cur_pos += temp_bias.size();
      }
    }
  }

  void Node::copy_bias_this_to_contained_layers() {
    if (m_bias.size() == 0) {
      return;
    }
    int cur_pos = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_bias = get_node(i).get_bias();
	for (int backing_index = 0; backing_index < temp_bias.size(); ++backing_index) {
	  temp_bias[backing_index] = m_bias[cur_pos + backing_index];
	}
	cur_pos += temp_bias.size();
      }
    }
  }

  void Node::copy_bias_gradients_contained_layers_to_this() {
    if (!is_composite()) {
      return;
    }
    // Do an initial pass through all layers to compute the total number of weight gradient
    // parameters.
    int total_size = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_bias_grad = get_node(i).get_bias_gradient();
	total_size += temp_bias_grad.size();
      }
    }
    // If this parameter count is different the size of the current m_bias_grad, then reinitialize.
    if (total_size != m_bias_grad.size()) {
      // Create 1-dim matrix of size total_size.
      if (VERBOSE_MODE) {
	cout << get_name() << ": Resizing m_bias_grad of size = " << total_size << endl;
      }
      //m_bias_grad = MatrixF(total_size); // fixme: move to init()
      m_bias_grad.resize(total_size);
    }
    // Now do another pass through all layers, this time copying the parameters into m_bias_grad.
    int cur_pos = 0;
    for (int i = 0; i < get_contained_node_count(); i++) {
      if (get_node(i).is_shared() == false) {
	MatrixF& temp_bias_grad = get_node(i).get_bias_gradient();
	for (int backing_index = 0; backing_index < temp_bias_grad.size(); ++backing_index) {
	  m_bias_grad[cur_pos + backing_index] = temp_bias_grad[backing_index];
	}
	cur_pos += temp_bias_grad.size();
      }
    }
  }

  void Node::make_internal_input_port_connections() {
    if (VERBOSE_MODE) {
      cout << "Connecting input ports of " << get_name() << " to internal nodes..." << endl;
    }
    for (auto& connection : m_input_to_internal_connections) {
      string input_name = connection.get_input_name();
      Node& contained_node = connection.get_contained_node();
      string contained_input = connection.get_contained_input();
      if (VERBOSE_MODE) {
	cout << "Connecting input port: " << input_name << " of " << get_name()  << " to input port: " 
	     << contained_input << " of contained node: " << contained_node.get_name() << endl;
      }
      const MatrixF& this_input_forward = get_input_port_forward(input_name); 
      MatrixF& this_input_backward = get_input_port_backward(input_name);
      contained_node.delete_input_port(contained_input);
      contained_node.create_input_port(this_input_forward, this_input_backward, contained_input); 
    }
    if (VERBOSE_MODE) {
      cout << endl;
    }
  }

}
