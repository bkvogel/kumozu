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


using namespace std;



namespace kumozu {

// The number of nodes created so far.
int Node::node_id = 0;

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
    //cout << "m_input_port_var_map.size(): " << m_input_port_var_map.size() << endl;
    for (const auto& x : m_input_port_var_map) {
        //cout << "name: " << x.first << endl;
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
        // Initialize this node and all deeply-contained nodes (allocated storage for parameters, connect ports, etc.)
        initialize();

    }
    // Recursively call forward computation.
    forward_propagate();
    // Recursively zero the paramater gradients.
    zero_parameter_gradients();
    // Recursively zero the input gradients.
    zero_input_grad();
}


void Node::back_propagate() {
    if (!is_initialized()) {
        error_exit("back_propagate(): " + get_name() + ": back_propagate() called before being initialized.");
    }
    back_propagate_activation_gradients();
    back_propagate_paramater_gradients();
}


void Node::create_input_port(VariableF& var, std::string input_name) {
    if (VERBOSE_MODE) {
        cout << get_name() << " : create_input_port(): with input port name: " << input_name << endl;
    }
    if (m_input_port_var_map.find(input_name) != m_input_port_var_map.end()) {
        m_input_port_var_map.erase(input_name);
    }
    m_input_port_var_map.emplace(input_name , std::ref(var));
    set_initialized(false);
}

void Node::create_input_port(VariableF& var) {
    create_input_port(var, DEFAULT_INPUT_PORT_NAME);
}

void Node::create_input_port(Node& parent, std::string parent_output, std::string input_name) {
    auto& parent_out_var = parent.get_output(parent_output);
    create_input_port(parent_out_var, input_name);
}

void Node::create_input_port_this_name(Node& parent, std::string input_name) {
    create_input_port(parent, DEFAULT_OUTPUT_PORT_NAME, input_name);
}

void Node::create_input_port_parent_name(Node& parent, std::string parent_output) {
    create_input_port(parent, parent_output, DEFAULT_INPUT_PORT_NAME);
}

void Node::connect_parent(Node& parent) {
    create_input_port(parent.get_output());
}

void Node::delete_input_port(std::string name) {
    auto it = m_input_port_var_map.find(name);
    if (it != m_input_port_var_map.end()) {
        m_input_port_var_map.erase(it);
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
    m_input_port_var_map.clear();
    m_input_port_fan_out_map.clear();
    m_input_port_extents_map.clear();
    set_initialized(false);
}

void Node::create_output_port(VariableF& var, std::string output_name) {
    if (m_output_port_var_map.find(output_name) != m_output_port_var_map.end()) {
        error_exit("create_output_port(): Error: " + output_name + " is already an output.");
    }
    m_output_port_var_map.emplace(output_name, std::ref(var));
    set_initialized(false);
}

void Node::create_output_port(Node& contained, std::string contained_output, std::string output_name) {
    create_output_port(contained.get_output(contained_output), output_name);
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

VariableF& Node::get_output(std::string name) {
    auto it = m_output_port_var_map.find(name);
    if (it == m_output_port_var_map.end()) {
        error_exit("get_output_forward(): Error accessing output port: Node: " + get_name() + " : port: " + name + " does not exist.");
    }
    return it->second;
}

const VariableF& Node::get_output(std::string name) const {
    auto it = m_output_port_var_map.find(name);
    if (it == m_output_port_var_map.end()) {
        error_exit("get_output_forward(): Error accessing output port: Node: " + get_name() + " : port: " + name + " does not exist.");
    }
    return it->second;
}

VariableF& Node::get_output() {
    if (get_output_port_count() != 1) {
        error_exit("get_output_forward(): " + get_name() + " should have 1 output port but instead has " + std::to_string(get_output_port_count()) + " ports.");
    }
    return m_output_port_var_map.begin()->second;
}

const VariableF& Node::get_output() const {
    if (get_output_port_count() != 1) {
        error_exit("get_output_forward(): " + get_name() + " should have 1 output port but instead has " + std::to_string(get_output_port_count()) + " ports.");
    }
    return m_output_port_var_map.begin()->second;
}

const MatrixF& Node::get_output_data(std::string name) const {
    return get_output(name).data;
}

const MatrixF& Node::get_output_data() const {
    return get_output().data;
}

void Node::delete_output_port(std::string name) {
    auto it = m_output_port_var_map.find(name);
    if (it == m_output_port_var_map.end()) {
        error_exit("delete_output_port(): Error accessing output port: " + name + " does not exist.");
    }
    m_output_port_var_map.erase(it);

    auto it3 = m_output_port_fan_out_map.find(name);
    if (it3 == m_output_port_fan_out_map.end()) {
        error_exit("delete_output_port(): Error accessing output port: " + name + " does not exist.");
    }
    m_output_port_fan_out_map.erase(it3);
    set_initialized(false);
}

void Node::delete_all_output_ports() {
    m_output_port_var_map.clear();
    m_output_port_fan_out_map.clear();
    set_initialized(false);
}

const MatrixF& Node::get_output_grad(std::string name) const {
    return get_output(name).grad;
}

MatrixF& Node::get_output_grad(std::string name) {
    return get_output(name).grad;
}

const MatrixF& Node::get_output_grad() const {
    return get_output().grad;
}

MatrixF& Node::get_output_grad() {
    return get_output().grad;
}

std::vector<int> Node::get_output_extents(std::string name) const {
    return get_output_data(name).get_extents();
}

const MatrixF& Node::get_input_port_data(std::string name) const {
    return get_input_port(name).data;
}

const MatrixF& Node::get_input_port_data() const {
    return get_input_port().data;
}

const MatrixF& Node::get_input_port_grad(std::string name) const {
    return get_input_port(name).grad;
}

MatrixF& Node::get_input_port_grad(std::string name) {
    return get_input_port(name).grad;
}

VariableF& Node::get_input_port(std::string name) {
    auto it = m_input_port_var_map.find(name);
    if (it == m_input_port_var_map.end()) {
        error_exit("get_input_port_backward(): Error accessing input port: node: " + get_name() + " : port: " + name + " does not exist.");
    }
    return it->second;
}

const VariableF& Node::get_input_port(std::string name) const {
    auto it = m_input_port_var_map.find(name);
    if (it == m_input_port_var_map.end()) {
        error_exit("get_input_port_backward(): Error accessing input port: node: " + get_name() + " : port: " + name + " does not exist.");
    }
    return it->second;
}

MatrixF& Node::get_input_port_grad() {
    return get_input_port().grad;
}

const MatrixF& Node::get_input_port_grad() const {
    return get_input_port().grad;
}

VariableF& Node::get_input_port() {
    if (get_input_port_count() != 1) {
        error_exit("get_input_port_backward(): " + get_name() +
                   " should have 1 input port but instead has " + std::to_string(get_input_port_count()) + " ports.");
    }
    return m_input_port_var_map.begin()->second;
}

const VariableF& Node::get_input_port() const {
    if (get_input_port_count() != 1) {
        error_exit("get_input_port_backward(): " + get_name() +
                   " should have 1 input port but instead has " + std::to_string(get_input_port_count()) + " ports.");
    }
    return m_input_port_var_map.begin()->second;
}

void Node::set_train_mode(bool is_train) {
    m_is_train = is_train;
}

void Node::print_paramater_stats() {
    if (!is_initialized()) {
        std::cerr << get_name() <<  ": print_paramater_stats() called before being initialized." << std::endl;
        error_exit("Exiting.");
    }
    if (get_input_port_data().size() > 0) {
        print_stats(get_input_port_data(), get_name() + " : input activations");
    }
    for (auto& param : get_params_map()) {
        print_stats(param.second->data, get_name() + " : " + param.first + " + : data");
        print_stats(param.second->grad, get_name() + " : " + param.first + " : grad");
    }

    if(get_output_data().size() > 0) {
        print_stats(get_output_data(), get_name() + " : output activations");
    }
    if (get_output_grad().size() > 0) {
        print_stats(get_output_grad(), get_name() + " : output activations deltas");
    }

}

void Node::save_parameters(std::string name) {
    if (!is_initialized()) {
        error_exit(get_name() +  ": save_parameters() called before being initialized.");
    }
    auto params = get_params();
    for (auto i = 0; i < params.size(); ++i) {
        save_matrix(params.at(i)->data, name + "_" + std::to_string(i) + "_" + get_name() + ".dat");
    }
}

void Node::load_parameters(std::string name) {
    auto params = get_params();
    for (int i = 0; i < params.size(); ++i) {
        auto data = load_matrix(name + "_" + std::to_string(i) + "_" + get_name() + "_data.dat");

        auto grad = load_matrix(name + "_" + std::to_string(i) + "_" + get_name() + "_grad.dat");
        // fixme:
        // not implemented yet...
    }
}

void Node::check_jacobian_parameters(std::map<std::string, std::vector<int>> input_port_extents_map) {
    cout << get_name() + ": Checking Jacobian for weights..." << endl;
    delete_all_input_ports();
    // We need to create a Variable for each input port in the map and then connect it to this Node as a new
    // input port.
    vector<VariableF> input_var_list;

    input_var_list.reserve(input_port_extents_map.size());
    // Alocate whole vector to be safe. Even with the reserve, it might decide to realocate while adding elements?
    for (const auto& x: input_port_extents_map) {
        auto& input_extents = x.second;
        input_var_list.push_back(VariableF(input_extents));
    }
    int total_size = 0;
    int i = 0;
    for (const auto& x: input_port_extents_map) {
        create_input_port(input_var_list.at(i), x.first);
        total_size += input_var_list.at(i).size();
        ++i;
    }

    // Create random input activations for the layer.
    VariableF input_var_flat(total_size);
    randomize_uniform(input_var_flat.data, 0.0f, 1.0f);
    randomize_uniform(input_var_flat.grad, 0.0f, 1.0f);

    // Copy from flat input matrix into the list of input matrices:
    copy_flat_variable_to_list(input_var_list, input_var_flat);

    forward(); // Initialize node.
    // Now the output activations have been initialized to the correct sizes.

    MatrixF output_forward_flat;
    copy_individual_to_flat_output_forward(output_forward_flat);

    // Now check Jacobian for weights:
    // Size will be total_output_dim x total_weights_dim =
    // (dim_output*minibatch_size) x total_weights_dim
    const int total_output_dim = output_forward_flat.size();
    // Now compute the numerical Jacobian:
    // This will be computed one column at a time.
    auto params = get_params();
    for (int n = 0; n < params.size(); ++n) {
        //cout << "n = " << n << endl;
        MatrixF& W = params.at(n)->data;
        //cout << "W = " << endl << W << endl;
        const int total_weights_dim = W.size();
        // This will contain the Jacobian computed using finite differences method.
        MatrixF numerical_jacobian_weights(total_output_dim, total_weights_dim);
        // Randomize to make accidental matches less likely.
        randomize_uniform(numerical_jacobian_weights, 0.0f, 1.0f);

        // This will contain the Jacobian computed using the back-prop method of the class
        MatrixF backprop_jacobian_weights(total_output_dim, total_weights_dim);
        // Randomize to make accidental matches less likely.
        randomize_uniform(backprop_jacobian_weights, 0.0f, 1.0f);

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
        MatrixF& grad_W = params.at(n)->grad;
        set_value(output_backwards_flat, 0.0f);
        for (int i=0; i < output_backwards_flat.size(); ++i) {
            output_backwards_flat[i] = 1.0f;
            // Now if we perform backprop, the result should be the same is row i of Jacobian.
            copy_flat_output_backward_to_individual(output_backwards_flat);
            zero_parameter_gradients();
            zero_input_grad();
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
    }
    delete_all_input_ports();
    cout << "PASSED" << endl;
}



void Node::check_jacobian_input_grad(std::map<std::string, std::vector<int>> input_port_extents_map) {
    cout << get_name() << ": Checking Jacobian for input error gradients..." << endl;
    delete_all_input_ports();
    // We need to create a Matrix for each input port in the map and then connect it to this Node as a new
    // input port.
    vector<VariableF> input_var_list;
    input_var_list.reserve(input_port_extents_map.size());
    // Alocate whole vector to be safe. Even with the reserve, it might decide to realocate while adding elements?
    for (const auto& x: input_port_extents_map) {
        auto& input_extents = x.second;
        input_var_list.push_back(VariableF(input_extents));
    }
    int total_size = 0;
    int i = 0;
    for (const auto& x: input_port_extents_map) {
        create_input_port(input_var_list.at(i), x.first);
        total_size += input_var_list.at(i).size();
        ++i;
    }

    // Create random input activations for the layer.
    VariableF input_var_flat(total_size);
    randomize_uniform(input_var_flat.data, 0.0f, 1.0f);
    randomize_uniform(input_var_flat.grad, 0.0f, 1.0f);

    // Copy from flat input matrix into the list of input matrices:
    copy_flat_variable_to_list(input_var_list, input_var_flat);

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
    const int total_input_dim = input_var_flat.size();
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
    for (int j=0; j < input_var_flat.size(); ++j) {
        // j is column index int Jacobian matrix.
        float orig = input_var_flat.data[j];
        input_var_flat.data[j] += m_epsilon;
        // Now compute output of layer -> output_forward
        // fixme: but this also copies into grad matrices. Only
        // need to copy the data part.
        copy_flat_variable_to_list(input_var_list, input_var_flat);
        forward();
        copy_individual_to_flat_output_forward(output_forward_flat);
        // Copy the output into column j of Jacobian.
        for (int i=0; i < output_forward_flat.size(); ++i) {
            numerical_jacobian_input(i,j) = output_forward_flat[i];
        }
        input_var_flat.data[j] = orig - m_epsilon;
        // Now compute output of layer -> output_forward
        // fixme: but this also copies into grad matrices. Only
        // need to copy the data part.
        copy_flat_variable_to_list(input_var_list, input_var_flat);
        forward();
        copy_individual_to_flat_output_forward(output_forward_flat);
        // Copy the output into column j of Jacobian.
        for (int i=0; i < output_forward_flat.size(); ++i) {
            numerical_jacobian_input(i,j) -= output_forward_flat[i];
            numerical_jacobian_input(i,j) /= 2*m_epsilon;
        }
        // Put back original value.
        input_var_flat.data[j] = orig;
    }
    // Now compute the Jacobian using the backprop function.
    MatrixF output_backwards_flat(output_forward_flat.size());

    set_value(output_backwards_flat, 0.0f);
    for (int i=0; i < output_backwards_flat.size(); ++i) {
        output_backwards_flat[i] = 1.0f;
        // Now if we perform backprop, the result should be the same is row i of Jacobian.
        copy_flat_output_backward_to_individual(output_backwards_flat);
        zero_parameter_gradients();
        zero_input_grad();
        back_propagate();
        // fixme: this also copies into data, but only need to copy the gradients part.
        copy_list_to_flat_variable(input_var_list, input_var_flat);
        for (int j=0; j < input_var_flat.size(); ++j) {
            backprop_jacobian_input(i,j) = input_var_flat.grad[j];
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

void Node::check_jacobian_parameters(std::vector<int> input_extents) {
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map[DEFAULT_INPUT_PORT_NAME] = input_extents;
    check_jacobian_parameters(input_port_extents_map);
}



void Node::check_jacobian_input_grad(std::vector<int> input_extents) {
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map[DEFAULT_INPUT_PORT_NAME] = input_extents;
    check_jacobian_input_grad(input_port_extents_map);
}


void Node::copy_flat_output_backward_to_individual(const MatrixF& flat_output_backward) {
    // Do an initial pass through all output matrices to determine the total number of elements.
    int total_size = 0;
    //
    for (auto& x: m_output_port_var_map) {
        auto& temp_mat = x.second.get().grad;
        total_size += temp_mat.size();
    }
    // If the element count is different the size of the current flat_mat, then exit with error.
    if (total_size != flat_output_backward.size()) {
        error_exit("copy_flat_output_backward_to_many(): Supplied matrix list has different element count than supplied flat matrix.");
    }
    int cur_pos = 0;
    for (auto& x: m_output_port_var_map) {
        auto& temp_mat = x.second.get().grad;
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
    for (auto& x: m_output_port_var_map) {
        auto& temp_mat = x.second.get().data;
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
    for (auto& x: m_output_port_var_map) {
        auto& temp_mat = x.second.get().data;
        for (int backing_index = 0; backing_index < temp_mat.size(); ++backing_index) {
            flat_output_forward[cur_pos + backing_index] = temp_mat[backing_index];
        }
        cur_pos += temp_mat.size();
    }
}

}
