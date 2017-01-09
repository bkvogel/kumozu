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

#include "CompositeNode.h"
#include "Utilities.h"
#include "MatrixIO.h"


using namespace std;



namespace kumozu {

void CompositeNode::forward_propagate() {
    if (get_contained_node_count() != 0) {
        for (int i = 0; i < get_contained_node_count(); ++i) {
            get_node(i).forward_propagate();
        }
    }
}

void CompositeNode::back_propagate_activation_gradients() {
    for (int i = get_contained_node_count()-1; i >= 0; --i) {
        get_node(i).back_propagate_activation_gradients();
    }
}

void CompositeNode::back_propagate_paramater_gradients() {
    if (get_contained_node_count() != 0) {
        for (int i = get_contained_node_count()-1; i >= 0; --i) {
            get_node(i).back_propagate_paramater_gradients();
        }
    }
}

void CompositeNode::zero_input_grad() {
    for (int i = 0; i < get_contained_node_count(); ++i) {
        get_node(i).zero_input_grad();
    }
}


void CompositeNode::schedule_node(Node& node) {
    m_scheduled_nodes.push_back(node);
    set_initialized(false);
}

void CompositeNode::add_node(std::unique_ptr<Node> node_ptr) {
    m_scheduled_nodes.push_back(*node_ptr);
    m_added_nodes.push_back(std::move(node_ptr));
    set_initialized(false);
}

Node& CompositeNode::get_node(int n) {
    return m_scheduled_nodes.at(n).get();
}

const Node& CompositeNode::get_node(int n) const {
    return m_scheduled_nodes.at(n).get();
}

int CompositeNode::get_contained_node_count() const {
    return static_cast<int>(m_scheduled_nodes.size());
}


void CompositeNode::make_internal_input_port_connections() {
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
        auto& this_input_var = get_input_port(input_name);
        contained_node.delete_input_port(contained_input);
        contained_node.create_input_port(this_input_var, contained_input);
    }
    if (VERBOSE_MODE) {
        cout << endl;
    }
}

void CompositeNode::connect_input_to_contained_node(std::string input_name, Node& contained, std::string contained_input) {
    m_input_to_internal_connections.push_back(input_to_contained_info(input_name, contained, contained_input));
    set_initialized(false);
}

void CompositeNode::connect_input_to_contained_node(std::string input_name, Node& contained) {
    connect_input_to_contained_node(input_name, contained, DEFAULT_INPUT_PORT_NAME);
}

void CompositeNode::connect_input_to_contained_node(Node& contained) {
    connect_input_to_contained_node(DEFAULT_INPUT_PORT_NAME, contained, DEFAULT_INPUT_PORT_NAME);
}

void CompositeNode::set_train_mode(bool is_train) {
    m_is_train = is_train;
    for (int i = 0; i < get_contained_node_count(); ++i) {
        get_node(i).set_train_mode(is_train);
    }
}

void CompositeNode::print_paramater_stats() {
    if (get_contained_node_count() != 0) {
        cout << get_name() << ": Parameter stats for contained nodes:" << endl;
        for (int i = 0; i < get_contained_node_count(); ++i) {
            get_node(i).print_paramater_stats();
        }
    }
}

}
