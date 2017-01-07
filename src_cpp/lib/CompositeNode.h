#ifndef _COMPOSITE_NODE_H
#define _COMPOSITE_NODE_H
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
 * A node that contains a sub-graph of other nodes.
 *
 * A CompositeNode is a node that contains a sub-graph of other nodes, thus supporting
 * models with hiearchical structure. The contained nodes may each be an instance of
 * either CompositeNode, AtomicNode, or another subclass of Node.
 *
 * Usage:
 *
 * It is recomended to create a subclass of CompositeNode in which each of the contained nodes
 * appears as a member variable of the class. Each of these member nodes should then be initialized
 * in the constructor initializer list. The graph structure should then be specified in
 * the constructor, by calling the various member functions of CompositeNode to add input and/or
 * output ports to the node and to make the port connections between the contained nodes.
 *
 * Alternatively, if one perfers to allocate the contained nodes dynamically using smart pointers,
 * this setup code should be called from the constructor of the subclass of CompositeNode, making
 * sure to either call "add_node()" or using a container for the smart pointers that appears as
 * a member variable in the subclass.
 */
class CompositeNode : public Node {

    // Do not allow copy construction. fixme: make this work.
    // Reason: When constructing a computational graph, a node instance generally accepts references to externally-created nodes.
    // The default copy constructor will also use these same refernces so that the copy will potentially reference some of the same
    // nodes as the original node. Since it is generally not desirable for a "copy" node to share references with another node,
    // we do not allow it.
    // The preferred way to make a "copy" is to simply use the regular constructor to allocate the two nodes and then copy the
    // parameters from one to the other. For now, is probably easiest to do by creating a custom class for the node to be copied, and making
    // any contained nodes be member variables of the class.
    //
    // todo: Maybe the cleanest way to solve this is to write a custom compy constructor that copies parameters (values only,
    // not references).
    CompositeNode(const CompositeNode &node) = delete;

public:

    CompositeNode(std::string name):
        Node{name}
    {

    }


    virtual void forward_propagate() override;

    virtual void back_propagate_activation_gradients() override;

    virtual void back_propagate_paramater_gradients() override;

    /**
     * Recursively initialize this node and all deeply-contained nodes.
     *
     * This function is automatically called by forward() if the extents of the input ports have changed
     * or the node is no longer initialized (i.e., is_initialized() returns false). This function
     * can also be called explicitly if it is desired to initialize a node without performing a
     * forward propagation.
     *
     * This function will first connect input ports to any contained nodes (because the calls
     * to connect_input_to_contained_node() are delayed until this function is called).
     * This function will then be recursivly called on any
     * contained nodes.
     *
     */
    virtual void initialize() override {
        // Re-connect input ports to internal nodes, if any.
        make_internal_input_port_connections();
        for (int i = 0; i < get_contained_node_count(); ++i) {
            get_node(i).initialize();
        }
        set_initialized(true);
    }


    /**
     * For each input port of this node, set the gradients to 0.
     *
     * If this node has contained nodes, this function will be called recursively on them.
     *
     * The default implementation of forward() calls this function so that when it is time to start the
     * backwards data pass, all input gradients will have already been set to 0. This behavior
     * is typically desired because most Node subclasses will accumulate into their input gradients during the backward
     * pass.
     */
    virtual void zero_input_grad() override;

    /**
     * Add a node to the schedule.
     *
     * Calling this function adds the specified node to the static schedule. This
     * schedule represents a list of the nodes in the computation graph of this
     * node. When forward() is called on this node, the schedule is executed
     * which then causes forward() to be called on each node in the schedule
     * in the same order that the nodes were added to the schedule.
     *
     * The ordering of nodes in the schedule should be consistent with
     * a topological sort of the computation graph, but this check is not
     * performed.
     *
     * All input ports to the supplied node must have been added before this
     * function is called. Otherwise, an error will occur if an attempt is made to add
     * another input port after this function is called.
     *
     * Usage:
     *
     * This function should typically be used to schedule executtion of nodes that are
     * managed by a sub-class of Node. In this case, such nodes will typically be
     * member variables of the sub-class. For a node that was allocated dynamically
     * (as a shared_ptr), use add_node() instead.
     *
     *
     * todo: It is straightforward to use the ideas from "Dataflow Architectures" by Arvind and Culler, 1986.
     * to automate the scheduling so that a node becomes eligible to "fire" (in either forward or backward graph)
     * as soon as it has data on all input/output ports and is not blocked from firing on output/input ports.
     * Consider implementing this.
     *
     * @param node A node to add to the schedule.
     */
    void schedule_node(Node &node);

    /**
     * Add a node to the schedule and use this node to manage its lifetime.
     *
     * This function is provided as a convinience so that it may be used
     * instead of shedule_node() for nodes that are allocated dynamically.
     * Otherwise, shedule_node() is preferred.
     *
     * @param node_ptr A node to add to the schedule and be contained by this node. This is a unique_ptr
     * because if the caller wishes to retain ownership, schedule_node() should be used intead.
     */
    void add_node(std::unique_ptr<Node> node_ptr);

    /**
     * Return a reference to the n'th Node contained by this node.
     */
    virtual Node &get_node(int n) override;

    /**
     * Return a reference to the n'th Node contained by this node.
     */
    virtual const Node &get_node(int n) const override;

    /**
     * Return the number of nodes contained by this node, which is equal to the number of times
     * add_node() has been called.
     *
     * @return Number of contained nodes.
     */
    virtual int get_contained_node_count() const override;

    /**
     * Make all parameters of this node refer to those of the supplied node.
     *
     * @param node The node to which the parameters of this node will refer.
     */
    virtual void set_shared(Node &node) override {
        if (VERBOSE_MODE) {
            std::cout << "CompositeNode::set_shared():" << std::endl;
            std::cout << get_name() + ".set_shared( " + node.get_name() + " )" << std::endl;
        }
        for (int i = 0; i < get_contained_node_count(); ++i) {
            get_node(i).set_shared(node.get_node(i));
        }
    }

    virtual void copy_params_to(Node& other) override {
        if (VERBOSE_MODE) {
            std::cout << "CompositeNode::copy_params_to():" << std::endl;
            std::cout << get_name() + ".copy_params_to( " + other.get_name() + " )" << std::endl;
        }
        for (int i = 0; i < get_contained_node_count(); ++i) {
            get_node(i).copy_params_to(other.get_node(i));
        }
    }

    virtual std::vector<std::shared_ptr<VariableF>> get_params() override {
        std::vector<std::shared_ptr<VariableF>> ret;
        if (is_shared()) {
            return ret;
        }
        for (int i = 0; i < get_contained_node_count(); ++i) {
            auto temp_params = get_node(i).get_params();
            for (auto& var : temp_params) {
                ret.push_back(var);
            }
        }
        return ret;
    }

    virtual std::map<std::string, std::shared_ptr<VariableF>> get_params_map() override {
        std::map<std::string, std::shared_ptr<VariableF>> ret;
        // fixme: implement.
        // For each contained node, append the prefix of that nodes position in the shcedule (or its name)
        // to its parameter name and add it to "ret".
        error_exit("Sorry, not implemented yet.");
        return ret;
    }

    virtual void zero_parameter_gradients() override {
        for (int i = 0; i < get_contained_node_count(); ++i) {
            get_node(i).zero_parameter_gradients();
        }
    }



    /**
     * Connect an input port of this node to an input of a contained node.
     *
     * The input port \p input_name of this node is connected to the input \p contained_input of the
     * contained node \p contained.
     * This connection will be made just before the forward data pass, by the forward() function. At that time,
     * this node must already have the specified input port. Otherwise this function will exit with an error. If the contained
     * node already has the specified input port, it will be deleted and a new input port with the same name will be
     * created.
     *
     * @param input_name Name of input port of this node.
     * @param contained Name of the contained node.
     * @param contained_input Name of input port of the contained node.
     */
    void connect_input_to_contained_node(std::string input_name, Node &contained, std::string contained_input);

    /**
     * Connect an input port of this node to the input of a contained node that only uses 1 input port.
     *
     * This function may be used when this node has multiple input ports, one of which we wish to connect to a
     * contained node that only uses 1 input port.
     * The input port "input_name" of this node is connected to the input port of the contained node "contained" and
     * given the default input port name. Note that this function saves the user from having to specify a name for
     * the input port of the contained node. Since the contained node will only use 1 input port, there is no
     * ambiguity and the default port name can be used.
     *
     * This connection will be made just before the forward data pass, by the forward() function. At that time,
     * this node must have the specified input port. Otherwise this function will exit with an error. If the contained
     * node already has the specified input port, it will be deleted and a new input port with the same default name will be
     * created.
     *
     * @param input_name Name of input port of this node.
     * @param contained Name of the contained node.
     */
    void connect_input_to_contained_node(std::string input_name, Node &contained);

    /**
     * Connect the input port of this node to an input of a contained node.
     *
     * For the case where both this node and the contained node will have exactly 1 input port, this function
     * may be used to avoid specifying the port names and the default names will be used.
     *
     * This connection will be made just before the forward data pass, by the forward() function. At that time,
     * this node must have exactly 1 input port. Otherwise this function will exit with an error. If the contained
     * node already has an input port, it will be deleted and a new input port with the default name will be
     * created.
     *
     * @param contained Name of the contained node.
     */
    void connect_input_to_contained_node(Node &contained);

    virtual void set_train_mode(bool is_train) override;

    virtual void print_paramater_stats() override;



private:

    /*
     * Make connections from input ports of this node to various internal nodes.
     *
     * Be sure to wait until the expected input ports exist prior to calling.
     */
    void make_internal_input_port_connections();

    // Schedule list of nodes.
    // Holds a reference to each node in contained by this node in the order in which they were added.
    std::vector<std::reference_wrapper<Node>> m_scheduled_nodes;

    // Simple class to wrap the triple of things that are needed in order to connect an input port of this node to the input
    // port of a contained node.
    class input_to_contained_info {
    public:
        input_to_contained_info(std::string input_name, Node &contained_node, std::string contained_input) :
            m_input_name{input_name},
            m_contained_node{contained_node},
            m_contained_input{contained_input} {
        }

        std::string get_input_name() {
            return m_input_name;
        }

        Node &get_contained_node() {
            return m_contained_node;
        }

        const Node &get_contained_node() const {
            return m_contained_node;
        }

        std::string get_contained_input() {
            return m_contained_input;
        }

    private:
        std::string m_input_name;
        Node &m_contained_node;
        std::string m_contained_input;
    };

    // Represents a list of connections to make from an input port of this node to the input port of
    // a contained node.
    std::vector<input_to_contained_info> m_input_to_internal_connections;

    std::vector<std::unique_ptr<Node>> m_added_nodes;

};



}


#endif /* _COMPOSITE_NODE_H */
