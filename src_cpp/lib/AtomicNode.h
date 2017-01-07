#ifndef _ATOMIC_NODE_H
#define _ATOMIC_NODE_H
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
#include <unordered_map>
#include <map>
#include <memory>
#include "Constants.h"
#include "Utilities.h"
#include "Variable.h"
#include "Node.h"

namespace kumozu {

/**
 * An that only performs the forward and backward computations and does not contain other nodes.
 *
 * An AtomicNode is a node that performs forward and backward computations. It can be though
 * of as a "layer" in a network. Unlike a CompositeNode, an AtomicNode does not contain
 * other nodes.
 *
 * Usage:
 *
 * An AtomicNode subclass performs that actual computations in a network. Examples include
 * Layer subclasses such as LinearLayer. If the subclass will only contain 1 input port and
 * 1 output port, it should probably extend Layer. Otherwise it should extend this class.
 *
 * It is then required to override the following member functions:
 *
 * reinitialize()
 *
 * forward_propagate()
 *
 * back_propagate_activation_gradients()
 *
 * and, if the node contains any learnable parmaeters:
 * back_propagate_paramater_gradients().
 *
 * If the node contains learnable parameters, it is reqired to add each parmaeter by calling
 * add_param() in the constructor. Note that since AtomicNode manages the creation and resizing
 * of parameters, it is not necessary for the subclass to contain them as member variables.
 */
class AtomicNode : public Node {


public:

    AtomicNode(std::string name):
        Node{name}
    {

    }

    virtual void set_shared(Node &node) override {
        if (VERBOSE_MODE) {
            std::cout << "Node::set_shared():" << std::endl;
            std::cout << get_name() + ".set_shared( " + node.get_name() + " )" << std::endl;
        }
        if (this == &node) {
            error_exit("set_shared(): Error. Cannot make a node share parameters with itself.");
        }
        if (node.is_shared()) {
            error_exit("set_shared(): Error: Cannot make a node share parameters with another shared node.");
        }
        m_is_shared = true;

        auto other_params_map = node.get_params_map();
        for (auto& kv : m_params) {
            m_params[kv.first] = other_params_map[kv.first];
        }
    }

    virtual void initialize() override {
        reinitialize();
        set_initialized(true);
    }

    virtual Node &get_node(int n) override {
        error_exit("Cannot call this.");
        return *this;
    }

    virtual const Node &get_node(int n) const  override {
        error_exit("Cannot call this.");
        return *this;
    }


    virtual std::map<std::string, std::shared_ptr<VariableF>> get_params_map() override {
        if (is_shared()) {
            std::map<std::string, std::shared_ptr<VariableF>> ret;
            return ret;
        }
        return m_params;
    }

    /**
     * Add a new parameter to this node.
     *
     * For each learnable parameter that a node contains, this function should be
     * called to add that parameter to the parameters list. This is required
     * because the optimizer will query a node for its parameters list and then
     * only update the parmaeters that are found in the list. Thus, if a node
     * fails to add a learnable parameter to the list, that parameter will
     * not get updated during the training process.
     *
     * To retrieve the corresponding parameter later, call get_param().
     *
     * @param name The name of the parameter to add.
     * @param extents The extents of the parmaeter to add. Note that the extents
     * can be resized later and so the default extents create an empty Varaible.
     */
    void add_param(std::string name, std::vector<int> extents = {}) {
        auto var = std::make_shared<VariableF>(extents);
        if (m_params.find(name) != m_params.end()) {
            error_exit("add_param(): Error. Parameter: " + name + " already exists.");
        }
        m_params.emplace(name, var);
        set_initialized(false);
    }

    /**
     * Return the parameter with the supplied name.
     *
     * @param name The name of the parameter to return. If no parameter is found
     * with the supplied name, exit with an error.
     * @return The parameter.
     */
    VariableF& get_param(const std::string& name) {
        if (m_params.find(name) == m_params.end()) {
            error_exit("get_param(): Error. Parameter: " + name + " does not exist.");
        }
        return *m_params[name];
    }

    virtual std::vector<std::shared_ptr<VariableF>> get_params() override {
        std::vector<std::shared_ptr<VariableF>> ret;
        if (is_shared()) {
            return ret;
        }
        for (auto& kv : m_params) {
            ret.push_back(kv.second);
        }
        return ret;
    }

    virtual void copy_params_to(Node& other) override {
        if (VERBOSE_MODE) {
            std::cout << "AtomicNode::copy_params_to():" << std::endl;
            std::cout << get_name() + ".copy_params_to( " + other.get_name() + " )" << std::endl;
        }
        auto other_params_map = other.get_params_map();
        auto this_params_map = get_params_map();
        for (auto& kv : this_params_map) {
            if (VERBOSE_MODE) {
                std::cout << "AtomicNode::copy_params_to():" << std::endl;
                std::cout << "kv.fist: " << kv.first << std::endl;
                auto rhs = *m_params[kv.first];
                auto rhs_data = rhs.data;
                std::cout << "rhs_data size: " << rhs_data.size() << std::endl;
                if (other_params_map.size() == 0) {
                    std::cout << "oops, params map is empty!" << std::endl;
                }
                if (other_params_map.find(kv.first) == other_params_map.end()) {
                    std::cout << "AtomicNode::copy_params_to(): other node: " << other.get_name() << "does not contain the parmaeter: " << kv.first << std::endl;
                    error_exit("exiting.");
                }
                if (other_params_map[kv.first] == nullptr) {
                    error_exit("oops");
                }
                auto lhs = *other_params_map[kv.first];
                auto lhs_data = lhs.data;
                std::cout << "lhs_data size: " << lhs_data.size() << std::endl;
            }
            *other_params_map[kv.first] = *this_params_map[kv.first];
        }
    }

    virtual void zero_parameter_gradients() override {
        for (auto& param : m_params) {
            set_value(param.second->grad, 0.0f);
        }
    }

    virtual void zero_input_grad() override {
        for (const auto& x : m_input_port_var_map) {
            auto& cur_input_backward = x.second.get().grad;
            set_value(cur_input_backward, 0.0f);
        }
    }

    /**
     * Reinitialize this node based on the current extents that are associated with the input ports.
     *
     * This function must be called before the node can be used and must also be called whenever the
     * input extents change during runtime. This function will automatically be called by deep_initialize().
     * The user may assume that if the input port extents change during runtime or if the computational graph
     * of this node is changed during runtime, that this function will be automatically called
     * before forward_propagate() is called.
     *
     * If this is an atomic node, any initialization of parameters and output port matrices should be performed
     * here. For a subclass such as a composite node, the computational sub-graph any contained nodes may be modified here.
     *
     * Note that a call to this function can be somewhat expensive since several matrices (such
     * as the output activations and possibly the parameters as well) might need to
     * be reallocated and initialized. As a general convention, parameter matrices should only
     * be reallocated and/or reinitialized if their size/shape changes.
     *
     * Usage:
     *
     * Parameters that were added in the constructor should typically be initialized or resized
     * in this function. Other data that is needed for the forward pass and that is a function of
     * the input activation extents should also be initialized or resized here.
     */
    virtual void reinitialize() { }

private:

    // Map from parameter name to the corresponding Variable.
    std::map<std::string, std::shared_ptr<VariableF>> m_params;

};

}

#endif /* _ATOMIC_NODE_H */
