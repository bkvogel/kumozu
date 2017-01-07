#ifndef _NODE_H
#define _NODE_H
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

namespace kumozu {

/**
 * An abstract base class that represents a node in a computational graph.
 *
 * A Node is an entity in a computational graph that can have an arbitrary number of named input ports and output ports. Each input or
 * output port contains a reference to two Matrix objects: one Matrix for the forward pass and one Matrix for the backward pass. Specifically,
 * an input port is associated with a read-only Matrix (the input activations) that data is read from during the forward pass.
 * This is the means by which
 * external data makes its way into the node during the forward pass. The same input port also contains a reference to a writeable
 * Matrix that the error gradients are written into during the backward pass. Likewise, each output port also contains a reference
 * to two Matrix objects, one representing the output activations and the other representing the corresponding output error
 * gradients which are read during the backward pass.
 *
 * Computational graph:
 *
 * The forward computational graph (and the backward graph as well) corresponds to the static dataflow model of computation as
 * discussed in "Dataflow Architectures"
 * by Arvind and Culler, 1986. This model of computation, also known as synchronous dataflow or SDF, has subsequently been used in
 * a variety of software, such as the UC Berkeley Ptolemy project. Some of the inspiration for Kumozu's graph architecture, such
 * as the use of atomic and composite containers with explicit input and output ports, comes from the Ptolemy II software.
 *
 * One motivation for representing the computational graph explicitly as a data flow graph
 * is that such a representation can directly expose parallelism such as "spatial parallelsim" to a scheduling algorithm.
 * A variety of scheduling algorithms (which are not yet implemented) can potentially be implemented as modules and applied to a given graph.
 * Since the scheduling task can be automated, the user then
 * only needs to focus on defining a valid and useful graph, but does not need to be concerned with the actual
 * scheduling of the nodes in the graph. Although the present version of Kumozu does require the user to specify the scheduling
 * order (by calling add_node()), we intend to remove this requirement in a future version.
 *
 * Usage:
 *
 * Suppose we have a Node that contains two input ports "A" and "B" and two output ports "C" and "D."
 * During the forward pass, data is read from the "input_activaions" matrices of "A" and "B". The Node may perform some operations
 * on this data and then write the output values into the "data" matrices of "C" and "D." During the backward pass, the
 * same thing happens in the reverse direction (think of reversing all of the edge directions in the DAG for the forward computational
 * graph).
 * Data is first read from the "grad" matrices of "C" and "D", the Node may perform some operations on this data, and then
 * the results are written into the "grad" matrices of "A" and "B."
 *
 * Shared nodes:
 *
 * In some networks, such as recurrent networks, it may be desired to replicate a node or sub-graph across time, keeping
 * parameters tied across all time slices. To build such a network, first define a node (for example, called "slice") that represents the sub-graph
 * corresponding to a single time slice. Then create as many copies of "slice" as desired, calling set_shared("slice") on each copy.
 * This will cause the parameters in each of the copies to refer to the parameters in "slice.".
 *
 * Fan out:
 *
 * Many nodes support output ports with a fan out greater than 1. This is posssible as long as all nodes in the network accumulate into
 * their "grad" activations during the backpropagation step. That is, in order for output ports with multiple fan out to function
 * correctly, all "grad" matrices should be set to 0 during the final part of the forward() call, which already is implemented
 * in this class. However, it is also necessary for each sub-class of Node in the network to accumulate rather than overwrite during the
 * backpropagation call.
 *
 * For nodes that do not accumulate into their "grad" matrices during the backpropagation call, an explicit SplitterNode may be
 * used to acheive the same effect as multiple fan out.
 *
 */
class Node {

    // Do not allow copy construction.
    // Reason: When constructing a computational graph, a node instance generally accepts references to externally-created nodes.
    // The default copy constructor will also use these same refernces so that the copy will potentially reference some of the same
    // nodes as the original node. Since it is generally not desirable for a "copy" node to share references with another node,
    // we do not allow it.
    // The preferred way to make a "copy" is to simply use the regular constructor to allocate the two nodes and then copy the
    // parameters from one to the other.
    //
    // todo: consider implementing a custom copy constructor to make this correct as expected.
    Node(const Node &node) = delete;

public:

    /**
     * Create a new node.
     *
     * @param name An optional name for this node.
     */
    Node(std::string name="") :
        m_is_initialized{false},
        m_name{name},
        m_is_train{true},
        m_epsilon{1e-4f},
        m_pass_relative_error{3e-2f}, // Minimum relative error needed to pass Jacobian checker.
        m_is_shared{false},
        m_id {++node_id}
    {

    }

    /**
     * Compute the output activations as a function of input activations.
     *
     * After computing the output activations, the parameter gradients are set to 0.
     *
     * During the first call to this function, the network will be initialized using
     * the extents of the supplied input activations. The network will also be reinitialized
     * any time that the extents of the supplied input activations change.
     *
     * Before calling this function for the first time, be sure that input ports have already beeen added
     * to this node and connected to the appropriate contained sub-graph(fixme: not true). Otherwise, this function will exit
     * with an error.
     */
    void forward();

    /**
     * Compute the output activations as a function of input activations.
     *
     * The output activations that are updated by this function can be obtained by calling get_output_forward().
     *
     * Sub-classes should override this function to perform the required forward computations.
     */
    virtual void forward_propagate() {}

    /**
     * Perform a full back-propagation pass through the network.
     *
     * It is assumed that forward() has already been called with the same
     * input activations.
     *
     * This call reads the curent output error gradients and updates the input error gradients
     * (i.e., input deltas) and also updates the parameter gradients
     * (weights and bias) for layers that contain and use such parameters.
     *
     * Note: This function is not virtual becuase subclasses should have no need to override it. A
     * subclasse should instead override the back_propagate_deltas() and back_propagate_paramater_gradients()
     * functions since
     * they will be called by this function.
     *
     */
    void back_propagate();

    /**
     * Back-propagate activation gradients to compute new values for the input activation
     * gradients.
     *
     * The convention is that this function should be called before back_propagate_paramater_gradients().
     * This function can also be called without calling back_propagate_paramater_gradients() if it is
     * only desired to back-propagate the activation gradients from the network output to its inputs. However,
     * it is normally not recommended to call this function directly since it will be called by
     * back_propagate().
     *
     * The output activation gradients must have already been updated before
     * calling this method. Otherwise, the error gradients will not be back-propagated
     * correctly.
     *
     * The default implementation assumes that this is a composite node. A subclass that is not
     * a composite node should override this function.
     */
    virtual void back_propagate_activation_gradients() {}

    /**
     * Compute the gradients for the paramaters.
     *
     * The convention is that this function should be called after back_propagate_deltas(). However,
     * it is normally not recommended to call this function directly since it will be called by
     * back_propagate().
     *
     */
    virtual void back_propagate_paramater_gradients() {}

    /**
     * Set all parameter gradients to zero.
     *
     * This function is automatically called by forward().
     */
    virtual void zero_parameter_gradients() = 0;

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
    virtual void zero_input_grad() = 0;

    /**
     * Return the name of this node.
     *
     * The returned name will consist of the string supplied to the constructor
     *
     * @return The name of this node.
     */
    std::string get_name() const {
        //return m_name + "/" + std::to_string(get_id());
        return m_name;
    }

    /**
     * Return the id of this node.
     *
     * The node id starts from 1 and is incremented when a new node is created.
     *
     * @return The id of this node.
     */
    int get_id() const {
        return m_id;
    }

    /**
     * Make all parameters of this node refer to those of the supplied node.
     *
     * Calling this function will cause the parameters of this
     * node to refer to those of the supplied node, meaning that any changes either node
     * makes to the parameters will be seen by all other nodes that are sharing the same
     * parameters.
     *
     * This function is intended to be used in networks such as recurrent networks where it is
     * convinient for training purposes to replicate a node in time so that each instance
     * corresponds to a different time slice, but the parameters are tied (i.e., shared) across all time
     * slices.
     *
     *
     * @param node The node to which the parameters of this node will refer.
     */
    virtual void set_shared(Node &node) = 0;

    /**
     * Return true if this node is using shared parameters. Otherwise return false.
     *
     * By default, a node is not shared and so this function will return false. However,
     * if set_shared() is called, this function will return true.
     */
    const bool is_shared() const {
        return m_is_shared;
    }

    /**
     * Create a new input port that will be associated with the supplied matrices.
     *
     * The two supplied matrices will be associated with the newly-created input port name \p input_name of
     * this node. The first of these matrices is \p data which is read from during the
     * forward data propagation. The second of these is \p grad in which this node will write
     * the input error gradients during the backward propagation.
     *
     * The input port \p input_name will be created by this function. If an existing port with the same name already
     * exists, it will be replaced and associated with the supplied matrices.
     *
     * In typical usage, this function may be called repeatedly (with different input port names) to create
     * as many input ports as desired for a node.
     *
     * Since the graph structure is changed when a port is added,
     * calling this function will cause this node to become uninitialied so that
     * is_initialized() will then return false if called.
     * fixme:
     * @param data An input activations matrix to connect to a new input port. This is a read-only matrix
     * that contains the activations for the forward pass.
     *
     * @param grad An input gradients matrix to connect to a new input port. This matrix will
     * be written to during the backward pass.
     * @param input_name The name of a new input to this node that will be created by this function and connected to
     * the supplied matrices.
     */
    void create_input_port(VariableF& var, std::string input_name);

    /**
     * Create a new input port for this node that is associated with the supplied matrices.
     *
     * The two supplied matrices will be associated with the newly-created input port with the default name.
     * The first of these matrices is "data" which is read from during the
     * forward data propagation. The second of these is "grad" in which this node will write
     * the input error gradients during the backward propagation.
     *
     * Calling this function will remove any existing input port with the default name and then create an input port with the
     * default name that will be associated with the supplied matrices. This function should therefore only be used when
     * it is desired for this node to have exactly 1 input port.
     *
     * In all cases, calling this function will cause is_initialized() to return false.
     *
     * @param var An input activations variable to connect to a new input port.
     */
    void create_input_port(VariableF& var);

    /**
     * Connect an output port of a parent node to an input port of this node, creating the input port in the process.
     *
     * Before calling this function, the output port "parent_output" of Node "parent" must already exist. Otherwise,
     * an error will occur. The input port "input_name" will be created by this function. If an input port of the same
     * name already exists, it will be replaced with a port of the same name that will be associated with the supplied
     * parent output port.
     *
     * In all cases, calling this function will cause is_initialized() to return false.
     *
     * This function may be called multiple times to connect various output ports of various parent nodes to different
     * input ports of this node.
     *
     * @param parent A parent node of this node.
     * @param parent_output The name of an output port of "parent."
     * @param input_name The name of a new input port for this node that will be created by this function and connected to the
     * parent's output port as specified.
     */
    void create_input_port(Node &parent, std::string parent_output, std::string input_name);

    /**
     * Connect an output port of a parent node to an input port of this node, creating the input port in the process.
     *
     * This function is intended to be used for the case where the supplied "parent" node only has 1 output node. In
     * this case, there corresponding name of the parent's output port does not need to be specified. However, it is
     * assumed that this node might have several input nodes and so the input port name "input_name" does need to
     * be specified.
     *
     * Before calling this function, the output port of Node "parent" must already exist. Otherwise,
     * an error will occur. The input port of this node with name "input_name" will be created by this function. If an input port of the same
     * name already exists, it will be replaced with a port of the same name.
     *
     * In all cases, calling this function will cause is_initialized() to return false.
     *
     * This function may be called multiple times to connect various output ports of various parent nodes to different
     * input ports of this node.
     *
     *
     * @param parent A parent node of this node.
     * @param input_name The name of a new input port for this node that will be created by this function and connected to the
     * parent's output port as specified.
      */
    void create_input_port_this_name(Node &parent, std::string input_name);

    /**
     * Connect an output port of a parent node to an input port of this node, creating the input port in the process.
     *
     * This function is intended to be used for the case where the supplied "parent" node has potentially several output nodes
     * and this node will have only 1 input port. In this case, only the parent's output port name needs to be specified; the
     * name of the input port to be created for this node does not need to be specified since this function will create
     * an input port with the default input port name.
     *
     * Before calling this function, the output port of Node "parent" must already exist. Otherwise,
     * an error will occur. If this node already contains an input port with the default name, it will be replaced by
     * with a new input port of the same default name.
     *
     * In all cases, calling this function will cause is_initialized() to return false.
     *
     * @param parent A parent node of this node.
     * @param parent_output The name of an output port of "parent."
     */
    void create_input_port_parent_name(Node &parent, std::string parent_output);

    /**
     * Connect the output of the specified parent node to this node, creating an input port
     * with the default name in the process.
     *
     * This function is intended to be used when it is desired to connect a parent node that
     * contains exactly 1 output port to this node, giving this node exactly 1 input port.
     * Therefore, when this function is called, the specified parent node must have exactly 1 output port
     * and any existing input port of this node with the default name will be removed and replaced with an input port with the
     * default name.
     * Since no port names are used by this function, the name of the parent node's output port does not matter.
     *
     * In all cases, calling this function will cause is_initialized() to return false.
     *
     * @param parent A parent node of this node.
     */
    void connect_parent(Node &parent);

    /**
     * Return the number of input ports.
     *
     * @return the number of input ports.
     */
    int get_input_port_count() const {
        return static_cast<int>(m_input_port_var_map.size());
    }

    /**
     * Delete the input port of this Node with the supplied name.
     *
     * If no port already exists with the supplied name, do nothing.
     *
     * @param name Name of input port to delete.
     */
    void delete_input_port(std::string name);

    /**
     * Delete all input ports from this Node.
     *
     * If no ports already exist, do nothing.
     */
    void delete_all_input_ports();

    /**
     * Create an output port for this Node that is associated with the supplied variable.
     *
     * @param var The output activations for the output port.
     * @param output_name The name for the output port.
     */
    void create_output_port(VariableF& var, std::string output_name);

    /**
     * Create an output port for this node that is connected to the output port of a contained node.
     *
     * @param contained A node that is contained inside this node.
     * @param contained_output The name of an output port of node "contianed."
     * @param output_name The name of an output port of this node that will be created and connected to
     * "contained_output" of node "contained."
     */
    void create_output_port(Node &contained, std::string contained_output, std::string output_name);

    /**
     * Create a new output port for this node that is connected to the output port of a contained node.
     *
     * This function is intended to be used to connect a contained node with exactly 1 output port to one of
     * the (typically multiple) output ports of this node.
     * This function may be used for the case where the contained node has excatly 1 output port (with any name),
     * so that the corresponding port name does not need to be specified. However, the name of the output port
     * for this node does need to be specified.
     *
     * @param contained A node that is contained inside this node.
     * @param output_name The name of an output port of this node that will be created and connected to
     * "contained_output" of node "contained."
     */
    void create_output_port_this_name(Node &contained, std::string output_name);

    /**
     * Create a new output port for this node that is connected to an output port of a contained node.
     *
     * This function is intended to be used to connect a contained node with possibly several output ports to the
     * sole output port of this node.
     * Since this node will only have 1 output port, the port name does not need to be specified. However, since the
     * contained node may have several output ports, its port name does need to be specified.
     *
     * @param contained A node that is contained inside this node.
     * @param output_name The name of the output port of the contained node that will be connected to a new output
     * port of default name of this node.
     */
    void create_output_port_contained_name(Node &contained, std::string output_name);

    /**
     * Create a new output port for this node that is connected to the output port of the specified contained ndoe.
     *
     * Since this function takes no port names, it is intended to be used for the case where the contained node
     * has exactly 1 output port (with any name) and also this node will have exactly 1 output port (created by
     * this function call) which will be given the default output port name.
     * This thus releives the user from having to specify port names when both nodes will only use 1 output port.
     *
     */
    void create_output_port(Node &contained);

    /**
     * Return the number of output ports.
     *
     * @return the number of output ports.
     */
    int get_output_port_count() const {
        return static_cast<int>(m_output_port_var_map.size());
    }

    /**
     * Return the output activations associated with the specified output port.
     *
     * @param name The name of the output port.
     *
     * @return A reference to the associated output activations (data and
     * gradients).
     */
    VariableF& get_output(std::string name);

    /**
     * Return the output activations associated with the specified output port.
     *
     * @param name The name of the output port.
     *
     * @return A reference to the associated output activations (data and
     * gradients).
     */
    const VariableF& get_output(std::string name) const;

    /**
     * Return the output activations associated with the specified output port.
     *
     * @param name The name of the output port.
     *
     * @return A reference to the output activations (data part only).
     */
    const MatrixF &get_output_data(std::string name) const;

    /**
     * Return the output activations associated with the output port.
     *
     * This function can be called if the node has exactly 1 output
     * port.
     *
     * @param name The name of the output port.
     *
     * @return A reference to the output activations (data part only).
     */
    const MatrixF &get_output_data() const;

    /**
     * Return the output activations associated with the output port.
     *
     * This function can be called if the node has exactly 1 output
     * port.
     *
     * @return A reference to the output activations.
     */
    VariableF& get_output();

    /**
     * Return the output activations associated with the output port.
     *
     * This function can be called if the node has exactly 1 output
     * port.
     *
     * @return A reference to the output activations.
     */
    const VariableF& get_output() const;

    /**
     * Delete the output port of this Node with the supplied name.
     *
     * If no port already exists with the supplied name, exit with an error.
     *
     * @param output_name Name of output port to delete.
     */
    void delete_output_port(std::string name);

    /**
     * Delete all output ports from this Node.
     *
     * If no ports already exist, do nothing.
     */
    void delete_all_output_ports();

    /**
     * Return the gradients associated with the specified output port.
     *
     * @param name The name of the output port.
     *
     * @return A reference to the associated output gradients.
     */
    const MatrixF &get_output_grad(std::string name) const;

    /**
     * Return the gradients associated with the specified output port.
     *
     * @param name The name of the output port.
     *
     * @return A reference to the associated output gradients.
     */
    MatrixF &get_output_grad(std::string name);

    /**
     * Return the output deltas/errors associated with the single output port.
     *
     * This function can only be used for a node that has exactly 1 output port. Otherwise,
     * this function will exit with an error.
     *
     * @return A reference to the associated output deltas.
     */
    const MatrixF &get_output_grad() const;

    /**
     * Return the output deltas/errors associated with the single output port.
     *
     * This function can only be used for a node that has exactly 1 output port. Otherwise,
     * this function will exit with an error.
     *
     * @return A reference to the associated output deltas.
     */
    MatrixF &get_output_grad();

    /**
     * Return the output extents associated with the specified output port.
     *
     * Note that the output extents will not have meaningful values until the
     * node has been initialized (i.e., forward() has been
     * called at least once).
     */
    std::vector<int> get_output_extents(std::string name) const;


    /**
     * Return true if this node has already been initialized. Otherwise return false.
     *
     * If this node is currently uninitialized, calling forward() will initialize it.
     */
    bool is_initialized() const {
        return m_is_initialized;
    }

    /**
     * Set the mode of this layer to either train or test/evaluate.
     *
     * Some layers, such as dropout layers, behave differently between training
     * and evaluation modes. Most other sub-layers can ignore this mode, however.
     *
     * The default value is true.
     */
    virtual void set_train_mode(bool is_train);

    /**
     * Return the current train/test mode of this node.
     *
     * @return True if currently in "train mode." False if currently in "test/evaluation mode."
     */
    bool is_train_mode() const {
        return m_is_train;
    }

    /**
     * Return a list of all parameters contained by this node.
     *
     * Note that only the variables are returned. The corresponding names are
     * not returned. If the names are also needed, call get_params_map() instead.
     *
     * If a node has parameters that are shared with another node, the shared node
     * will return an empty paramaters list.
     *
     * This function is used by an optimizer to obtain the list of learnable
     * parameters for a node. This function only needs to be called once if the
     * computation graph is static, typically
     * after the first forward pass.
     *
     * @return The parameters of this node.
     */
    // todo: consider making a const version of this function also.
    virtual std::vector<std::shared_ptr<VariableF>> get_params() = 0;

    /**
     * Return the parameters map for this node.
     *
     * The parameters map is a map from string-valued parameter name to
     * the corresponding Variable.
     *
     * If a node has parameters that are shared with another node, the shared node
     * will return an empty paramaters map.
     *
     * @return The paramaters map.
     */
    virtual std::map<std::string, std::shared_ptr<VariableF>> get_params_map() = 0;

    /**
     * Return a reference to the n'th Node contained by this node.
     */
    virtual Node &get_node(int n) = 0;

    /**
     * Return a reference to the n'th Node contained by this node.
     */
    virtual const Node &get_node(int n) const = 0;

    /**
     * Return the number of nodes contained by this node, which is equal to the number of times
     * add_node() has been called.
     *
     * @return Number of contained nodes.
     */
    virtual int get_contained_node_count() const {
        return 0;
    }


    /**
     * Copy parameter values from this node to the supplied node.
     *
     * Only the parmaeter values are copied. Thus, after calling this function,
     * if parameter "A" is modified in "other" node, it will not have any effect
     * on parameter "A" of this node.
     *
     * @param other The node that this node's parameters will be copied into.
     */
    virtual void copy_params_to(Node& other) = 0;

    /**
     * Print some statistics for the parameters of this node.
     *
     * Subclasses may override to provide additional or different information.
     */
    virtual void print_paramater_stats();

    /**
     * Save parameters to a file withe the prefix given
     * by the supplied name.
     *
     * This base class function saves the weights and bias parameters only.
     * Subclasses may override to save additional parameters if desired.
     * In general, calling this function will produce several files, one
     * for each parameter matrix in the network.
     *
     * @param name The name to use as a prefix of the filenames of the saved
     * parameters. This same name can then be supplied to load_parameters().
     *
     */
    virtual void save_parameters(std::string name);

    /**
     * Load saved parameters from a file. The string name should be
     * the same that was used to save the parameters when
     * save_learning_info() was called.
     *
     * Note: this function should only be called after the network has
     * been initialized by calling forward().
     *
     * This base class function loads the weights and bias parameters only.
     * Subclasses may override to load additional parameters if desired.
     */
    virtual void load_parameters(std::string name);


    /**
     * Check that the Jacobian computed using the finite differences method agrees with
     * the Jacobian computed using the gradient back-propagation member functions.
     *
     * This function checks only the parameters.
     *
     * @param input_port_extents_map A map from input port name to dimensions (extents) of the matrices
     * associated with that input port.
     */
    void check_jacobian_parameters(std::map<std::string, std::vector<int>> input_port_extents_map);


    /**
     * Check that the Jacobian computed using the finite differences method agrees with
     * the Jacobian computed using the gradient back-propagation member functions.
     *
     * This function checks only the input gradients.
     *
     * @param input_port_extents_map A map from input port name to dimensions (extents) of the matrices
     * associated with that input port.
     */
    void check_jacobian_input_grad(std::map<std::string, std::vector<int>> input_port_extents_map);


    /**
     * Check that the Jacobian computed using the finite differences method agrees with
     * the Jacobian computed using the gradient back-propagation member functions.
     *
     * This is a convinience function that can be used when the Node has only 1 input port.
     *
     * @param input_extents The extents (dimensions) for the input port.
     */
    void check_jacobian_parameters(std::vector<int> input_extents);

    /**
     * Check that the Jacobian computed using the finite differences method agrees with
     * the Jacobian computed using the gradient back-propagation member functions.
     *
     * This is a convinience function that can be used when the Node has only 1 input port.
     *
     * @param input_extents The extents (dimensions) for the input port.
     */
    void check_jacobian_input_grad(std::vector<int> input_extents);

    /**
     * Return the input activations associated with the input port of this node
     * with the specified name.
     *
     * @param name The name of the input port of this node.
     *
     * @return A reference to the associated input activations.
     */
    const MatrixF &get_input_port_data(std::string name) const;

    /**
     * Return the input activations associated with the input port of this node.
     *
     * Provided that this Node has exactly 1 input port, this function will return
     * the input activations without needing to specify a port name. That is, if a Node has
     * only 1 input port, there is only 1 possible input activations Matrix to return and hence
     * no need to supply a port name. However, if
     * this Node does not have exactly 1 input port, exit with an error.
     *
     * @return A reference to the associated input activations.
     */
    const MatrixF &get_input_port_data() const;

    /**
     * Apply the supplied function f(const MatrixF&) to each activations matrix (that
     * is, the data matrix) of each input port. Note that this data matrix
     * is read-only.
     *
     * The function f(const MatrixF&) must accept a single arugument of type "const MatrixF&".
     * It is suggested to use a lambda for f().
     *
     * For a usage example, see the ConcatNode class.
     */
    template<typename Func>
    void for_each_input_port_data(Func f) {
        for (const auto &x : m_input_port_var_map) {
            const MatrixF &cur_mat = x.second.get().data;
            f(cur_mat);
        }
    }

    /**
     * Apply the supplied function f(MatrixF&) to each input gradients matrix
     * (that is, the grad matrix) of each input port.
     *
     * The function f(MatrixF&) must accept a single arugument of type "MatrixF&".
     * It is suggested to use a lambda for f().
     *
     * For a usage example, see the ConcatNode class.
     */
    template<typename Func>
    void for_each_input_port_grad(Func f) {
        for (const auto &x : m_input_port_var_map) {
            MatrixF &cur_mat = x.second.get().grad;
            f(cur_mat);
        }
    }

    /**
     * Apply the supplied function f(const MatrixF& input_data, MatrixF& input_grad)
     * to each (input_data, input_grad) Matrix pair of each input port.
     *
     * It is suggested to use a lambda for f().
     */
    // todo: make this use Variable instead of 2 matrices.
    template<typename Func>
    void for_each_input_port(Func f) {
        for (const auto &x : m_input_port_var_map) {
            const MatrixF &data_mat = x.second.get().data;
            MatrixF &grad_mat = x.second.get().grad;
            f(data_mat, grad_mat);
        }
    }

    /**
     * Return the input gradients associated with the input port of this node
     * with the specified name.
     *
     * @param name The name of the input port of this node.
     *
     * @return A reference to the associated input gradients.
     */
    const MatrixF &get_input_port_grad(std::string name) const;

    /**
     * Return the input gradients associated with the input port of this node
     * with the specified name.
     *
     * @param name The name of the input port of this node.
     *
     * @return A reference to the associated input gradients.
     */
    MatrixF &get_input_port_grad(std::string name);

    /**
     * Return the input activations associated with the input port of this node
     * with the specified name.
     *
     * @param name The name of the input port of this node.
     *
     * @return A reference to the associated input activations (data and
     * gradients).
     */
    VariableF& get_input_port(std::string name);

    /**
     * Return the input activations associated with the input port of this node
     * with the specified name.
     *
     * @param name The name of the input port of this node.
     *
     * @return A reference to the associated input activations (data and
     * gradients).
     */
    const VariableF& get_input_port(std::string name) const;

    /**
     * Return the input gradients associated with the input port of this node.
     *
     * Provided that this Node has exactly 1 input port, this function will return
     * the input errors (deltas) without needing to specify a port name. That is, if a Node has
     * only 1 input port, there is only 1 possible input activations Matrix to return and hence
     * no need to supply a port name. However, if
     * this Node does not have exactly 1 input port, exit with an error.
     *
     * @return A reference to the associated input gradients.
     */
    MatrixF &get_input_port_grad();

    /**
     * Return the input gradients associated with the input port of this node.
     *
     * Provided that this Node has exactly 1 input port, this function will return
     * the input errors (deltas) without needing to specify a port name. That is, if a Node has
     * only 1 input port, there is only 1 possible input activations Matrix to return and hence
     * no need to supply a port name. However, if
     * this Node does not have exactly 1 input port, exit with an error.
     *
     * @return A reference to the associated input gradients.
     */
    const MatrixF &get_input_port_grad() const;

    /**
     * Return the input activations associated with the input port of this node.
     *
     * Provided that this Node has exactly 1 input port, this function will return
     * the input activations without needing to specify a port name. However, if
     * this Node does not have exactly 1 input port, exit with an error.
     *
     * @return A reference to the associated input activations (data and gradients).
     */
    VariableF& get_input_port();

    /**
     * Return the input activations associated with the input port of this node.
     *
     * Provided that this Node has exactly 1 input port, this function will return
     * the input activations without needing to specify a port name. However, if
     * this Node does not have exactly 1 input port, exit with an error.
     *
     * @return A reference to the associated input activations (data and gradients).
     */
    const VariableF& get_input_port() const;


    /**
     * Recursively initialize this node and all deeply-contained nodes.
     *
     * This function is automatically called by forward() if the extents of the input ports have changed
     * or the node is no longer initialized (i.e., is_initialized() returns false). This function
     * can also be called explicitly if it is desired to initialize a node without perform a
     * forward propagation.
     *
     * This function will first connect input ports to any contained nodes (because the calls
     * to connect_input_to_contained_node() are delayed until this function is called). reinitialize()
     * will then be called on this node. This function will then be recursivly called on any
     * contained nodes.
     *
     */
    virtual void initialize() = 0;

    /**
     * Set the initialization state of this Node.
     *
     * @param is_initialized True if initialized. False if not initialized.
     */
    void set_initialized(bool is_initialized) {
        m_is_initialized = is_initialized;
    }

protected:

    bool m_is_train;
    bool m_is_shared;
    std::map<std::string, std::reference_wrapper<VariableF>> m_input_port_var_map;

private:

    /*
     * Copy the elements of the supplied flat output backward matrix into the individual "output backward" matrices
     * associated with each of the output ports.
     *
     * The size of the supplied matrix must be the same as the total number of elements of all output
     * ports. Otherwise, exit with an error.
     */
    void copy_flat_output_backward_to_individual(const MatrixF &flat_output_backward);

    /*
     * Copy the elements of the individual "output data" matrices into the supplied
     * float output forward matrix.
     *
     * If the size of the supplied matrix is not equal to the total number of elements to be
     * copied, it will be resized.
     */
    void copy_individual_to_flat_output_forward(MatrixF &flat_output_forward);

    /*
     * Copy the elements from a flat Variable into a list of Variables.
     *
     * The size of "flat_mat" must be equal to the total number of elements in the variable list. Otherwise,
     * this function will exit with an error.
     *
     * @param mat_list The list of variables that will be copied to.
     * @param flat_mat The variable that will be copied from.
     */
    template <typename T>
    void copy_flat_variable_to_list(std::vector<Variable<T>>& mat_list, const Variable<T>& flat_mat) {
        // Do an initial pass through all matrices to determine the total number of elements.
        int total_size = 0;
        for (size_t i = 0; i < mat_list.size(); i++) {
            Variable<T>& temp = mat_list[i];
            total_size += temp.size();
        }
        // If the element count is different than size of the current flat_mat, then exit with error.
        if (total_size != flat_mat.size()) {
            error_exit("copy_flat_matrix_to_list(): Supplied matrix list has different element count than supplied flat matrix.");
        }
        int cur_pos = 0;
        for (size_t i = 0; i < mat_list.size(); i++) {
            Variable<T>& temp = mat_list[i];
            for (int backing_index = 0; backing_index < temp.size(); ++backing_index) {
                temp.data[backing_index] = flat_mat.data[cur_pos + backing_index];
                temp.grad[backing_index] = flat_mat.grad[cur_pos + backing_index];
            }
            cur_pos += temp.size();
        }
    }

    /*
     * Copy the elements from a list of Matrix into a single flat Matrix.
     *
     * Since it can be inconvinient to determine to determine the total number of elements in the matrix
     * list before calling this function, it is not necessary to supply a "flat_mat" of the correct size.
     * The supplied "flat_mat" will be resized to match the total number of elements.
     *
     * @param mat_list The list of matrices that will be copied from.
     * @param flat_mat The matrix that will be copied into. This matrix will be resized to the same total
     * number of elements in the matrix list if necessary.
     */
    template <typename T>
    void copy_list_to_flat_variable(const std::vector<Variable<T>>& mat_list, Variable<T>& flat_mat) {
        // Do an initial pass through all matrices to determine the total number of elements.
        int total_size = 0;
        for (size_t i = 0; i < mat_list.size(); i++) {
            const Variable<T>& temp = mat_list[i];
            total_size += temp.size();
        }
        // If the element count is different the size of the current flat_mat, then reinitialize.
        if (total_size != flat_mat.size()) {
            // Create 1-dim matrix of size total_size.
            std::cout << "Resizing flat_mat to size = " << total_size << std::endl;
            flat_mat.resize(total_size);
        }
        int cur_pos = 0;
        for (size_t i = 0; i < mat_list.size(); i++) {
            const Variable<T>& temp = mat_list[i];
            for (int backing_index = 0; backing_index < temp.size(); ++backing_index) {
                flat_mat.data[cur_pos + backing_index] = temp.data[backing_index];
                flat_mat.grad[cur_pos + backing_index] = temp.grad[backing_index];
            }
            cur_pos += temp.size();
        }
    }

    bool m_is_initialized;
    // Input port dictionaries.

    // Store the most recent extents for each input port.
    std::map<std::string, std::vector<int>> m_input_port_extents_map;
    // Store the number of outgoing connections for each input port. This number shoud be 1 for each port. Otherwise,
    // there is a connection error.
    std::map<std::string, int> m_input_port_fan_out_map;

    // Output port dictionaries.
    std::map<std::string, std::reference_wrapper<VariableF>> m_output_port_var_map;
    // Store the number of outgoing connections for each output port. This number shoud be 1 for each port. Otherwise,
    // there is a connection error.
    std::map<std::string, int> m_output_port_fan_out_map;
    std::string m_name;
    const float m_epsilon;
    const float m_pass_relative_error;
    static int node_id; // Number of nodes created so far, starts from 1.
    int m_id ;// Unique id for each node.

};

}

#endif /* _NODE_H */
