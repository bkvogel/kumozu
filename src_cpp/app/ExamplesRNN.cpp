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

#include "ExamplesRNN.h"
#include <cstdlib>
#include <string.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <ctime>
#include <omp.h>
#include <algorithm>
#include <memory>

#include "Matrix.h"
#include "MatrixIO.h"
#include "UnitTests.h"
#include "PlotUtilities.h"

#include "SequentialLayer.h"
#include "BoxActivationFunction.h"
#include "ColumnActivationFunction.h"
#include "ConvLayer2D.h"
#include "BatchNormalization3D.h"
#include "Dropout3D.h"
#include "PoolingLayer.h"
#include "ImageToColumnLayer.h"
#include "Dropout1D.h"
#include "LinearLayer.h"
#include "BatchNormalization1D.h"
#include "CrossEntropyCostFunction.h"
#include "MinibatchTrainer.h"
#include "Accumulator.h"
#include "LinearLayer.h"
#include "AdderNode.h"
#include "MultiplyerNode.h"
#include "SplitterNode.h"
#include "CharRNNMinibatchGetter.h"
#include "MSECostFunction.h"
#include "Utilities.h"
#include "SliceUnroller.h"
#include "ConcatNode.h"
#include "ExtractorNode.h"
#include "Variable.h"


using namespace std;

namespace kumozu {


// Multilayer RNN/LSTMs for text modeling similar to Karpathy's char-rnn Torch implementation from:
// https://github.com/karpathy/char-rnn
//

// Notes:
//
// This file contains a function lstm_example() that will train an RNN such as an LSTM on a text file "input.txt".
// After each training epoch, the test error will be printed and a short sequence of text will be generated.
//
// This file also defines several types of RNN node classes (Vanilla RNN and LSTM). Some of these may seem verbose
// but they are intended to show different ways in which the node can be defined.
//
// This is followed by several "slice" classes which use instances of the building-block RNN nodes to build a single
// time slice such as a multilayer slice.
//
// Finally, inside the lstm_example() function, a SliceUnroller is used to unroll the slice into an RNN that can be trained.
// A SliceSampler class is also defined that takes a single time slice and uses it to sample (i.e., generate) a text sequence.
//
// Things to try:
// Try uncommenting different "using" aliases to change the FullSlice between LSTM2LayerSliceV2, LSTM1LayerSliceV2 etc.
// Also try uncommenting differnt "using" aliases for LSTMNode to see how the runtime performance changes as different
// slice nodes are used, such as LSTMNodeV1, LSTMNodeV2, etc.

//////////////////////////////////////////////////////////////////////////////////////////////////////////


//
// Input ports:
//               "x_t"
//               "h_t"
//
// Output ports:
//               "h_t": hidden output to next node (in next time slice)
//               "y_t": output to next layer or final output (in same time slice)
//
// Note: Since this will be a composite node (that is, a node that contains a subgraph of
// other nodes), all we need to do is the following:
//
// 1. For each contained node, add a corresponding private member variable (see below).
// 2. In the constructor initialization list, call the constructor of each contained node (most of
// them will only need a name, but some will require configuration parameters.
// 3. In the constructor body, do the following in order:
//     i. Connect each input port of this node to desired internal node input port.
//     ii. Connect internal nodes together however you want, adding each of them via add_node() in
//         the order that they should be called in the forward data propagation.
//     iii. Connect outputs of internal nodes to output ports of this node.
class RNNNode : public CompositeNode {

public:

    // rnn_dim: dimension of internal RNN state.
    RNNNode(int rnn_dim, std::string name) :
        CompositeNode(name),
        // Note: Call the constructor of each contained node here, in the same order that
        // they appear as private member variables.
        m_linear_x_t {rnn_dim, "Linear x_t"},
        m_linear_h_t_prev {rnn_dim, "Linear h_t_prev"},
        m_adder {"Adder"},
        m_tanh {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh activation"},
        m_splitter{2, "Splitter"}
    {
        // This node is a "composite node" since it will contain a network subgraph.
        // We now spcecify the subgraph:

        // Note: add_node() must add the contained nodes in the same order that they should
        // be called in the forward data pass.

        // Connect inputs of this node to internal nodes:
        connect_input_to_contained_node("x_t", m_linear_x_t);
        schedule_node(m_linear_x_t);
        connect_input_to_contained_node("h_t", m_linear_h_t_prev);
        schedule_node(m_linear_h_t_prev);
        // Sum the outputs of the two linear nodes:
        m_adder.create_input_port_this_name(m_linear_x_t, "0");
        m_adder.create_input_port_this_name(m_linear_h_t_prev, "1");
        schedule_node(m_adder);

        // Connect the output of the adder to the input of a tanh activation:
        m_tanh.connect_parent(m_adder);
        schedule_node(m_tanh);

        // Connect the output of the tanh into the splitter node.
        m_splitter.connect_parent(m_tanh);
        schedule_node(m_splitter);

        // create output ports:
        create_output_port(m_splitter, "0", "h_t"); // Internal state to send to next time slice.
        create_output_port(m_splitter, "1", "y_t"); // Output for this time slice.
    }

    void initialize_params() {
        for (auto& param : m_linear_x_t.get_params()) {
            randomize_uniform(param->data, -0.08f, 0.08f);
        }
        for (auto& param : m_linear_h_t_prev.get_params()) {
            randomize_uniform(param->data, -0.08f, 0.08f);
        }
    }

private:
    // Each contain node should be a private member of this class:
    LinearLayer m_linear_x_t;
    LinearLayer m_linear_h_t_prev;
    AdderNode m_adder;
    ColumnActivationFunction m_tanh;
    SplitterNode m_splitter;
};


// This version uses 8 linear layers.
//
// Create a class that represents an LSTM node. This node can be replicated vertically to
// create an RNN with multiple layers and can be replicated horizontally (i.e., unrolled) to
// train the network.
//
// Input ports:
//               "x_t"
//               "h_t"
//               "c_t"
//
// Output ports:
//               "h_t": hidden output to next node (in next time slice)
//               "c_t": hidden output to next node (in next time slice)
//               "y_t": output to next layer or final output (in same time slice). This is the same value as h_t.
//
// Note: Since this will be a composite node (that is, a node that contains a subgraph of
// other nodes), all we need to do is the following:
//
// 1. For each contained node, add a corresponding private member variable (see below).
// 2. In the constructor initialization list, call the constructor of each contained node (most of
// them will only need a name, but some will require configuration parameters.
// 3. In the constructor body, do the following in order:
//     i. Connect each input port of this node to desired internal node input port.
//     ii. Connect internal nodes together however you want, adding each of them via add_node() in
//         the order that they should be called in the forward data propagation.
//     iii. Connect outputs of internal nodes to output ports of this node.
class LSTMNodeV1 : public CompositeNode {

public:

    // rnn_dim: dimension of internal RNN state.
    LSTMNodeV1(int rnn_dim, std::string name) :
        CompositeNode(name),
        // Note: Call the constructor of each contained node here, in the same order that
        // they appear as private member variables.
        m_splitter_4output_x_t {4, "Splitter: x_t"},
        m_splitter_4output_h_t {4, "Splitter: h_t"},
        m_linear_x_t_i {rnn_dim, "Linear x_t input"},
        m_linear_h_t_i {rnn_dim, "Linear h_t input"},
        m_linear_x_t_f {rnn_dim, "Linear x_t forget"},
        m_linear_h_t_f {rnn_dim, "Linear h_t forget"},
        m_linear_x_t_o {rnn_dim, "Linear x_t output"},
        m_linear_h_t_o {rnn_dim, "Linear h_t output"},
        m_linear_x_t_g {rnn_dim, "Linear x_t g"},
        m_linear_h_t_g {rnn_dim, "Linear h_t g"},
        m_adder_i {"Adder: i"},
        m_adder_f {"Adder: f"},
        m_adder_o {"Adder: o"},
        m_adder_g {"Adder: g"},
        m_sigmoid_i{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: input gate"},
        m_sigmoid_f{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: forget gate"},
        m_sigmoid_o{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: output gate"},
        m_tanh_g {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh g activation"},
        m_multiply_f_and_c_t {"Multiplyer: f * c_t"},
        m_multiply_i_g {"Multiplyer: i * g"},
        m_adder_c_t {"Adder: c_t"},
        m_splitter_2_output{2, "Splitter 2 outputs"},
        m_tanh_c_t {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh c_t"},
        m_multiply_h_t {"Multiplyer: h_t"},
        m_splitter_h_t_y_t {2, "Splitter h_t, y_t"}
    {
        // This node is a "composite node" since it will contain a network subgraph.
        // We now spcecify the subgraph:

        // Note: add_node() must add the contained nodes in the same order that they should
        // be called in the forward data pass.

        // Connect inputs of this node to internal nodes:
        connect_input_to_contained_node("x_t", m_splitter_4output_x_t);
        schedule_node(m_splitter_4output_x_t);
        connect_input_to_contained_node("h_t", m_splitter_4output_h_t);
        schedule_node(m_splitter_4output_h_t);

        m_linear_x_t_i.create_input_port_parent_name(m_splitter_4output_x_t, "0");
        schedule_node(m_linear_x_t_i);

        m_linear_x_t_f.create_input_port_parent_name(m_splitter_4output_x_t, "1");
        schedule_node(m_linear_x_t_f);

        m_linear_x_t_o.create_input_port_parent_name(m_splitter_4output_x_t, "2");
        schedule_node(m_linear_x_t_o);

        m_linear_x_t_g.create_input_port_parent_name(m_splitter_4output_x_t, "3");
        schedule_node(m_linear_x_t_g);

        m_linear_h_t_i.create_input_port_parent_name(m_splitter_4output_h_t, "0");
        schedule_node(m_linear_h_t_i);

        m_linear_h_t_f.create_input_port_parent_name(m_splitter_4output_h_t, "1");
        schedule_node(m_linear_h_t_f);

        m_linear_h_t_o.create_input_port_parent_name(m_splitter_4output_h_t, "2");
        schedule_node(m_linear_h_t_o);

        m_linear_h_t_g.create_input_port_parent_name(m_splitter_4output_h_t, "3");
        schedule_node(m_linear_h_t_g);

        // Sum the outputs of the two linear nodes:
        m_adder_i.create_input_port_this_name(m_linear_x_t_i, "0");
        m_adder_i.create_input_port_this_name(m_linear_h_t_i, "1");
        schedule_node(m_adder_i);

        m_adder_f.create_input_port_this_name(m_linear_x_t_f, "0");
        m_adder_f.create_input_port_this_name(m_linear_h_t_f, "1");
        schedule_node(m_adder_f);

        m_adder_o.create_input_port_this_name(m_linear_x_t_o, "0");
        m_adder_o.create_input_port_this_name(m_linear_h_t_o, "1");
        schedule_node(m_adder_o);

        m_adder_g.create_input_port_this_name(m_linear_x_t_g, "0");
        m_adder_g.create_input_port_this_name(m_linear_h_t_g, "1");
        schedule_node(m_adder_g);

        m_sigmoid_i.connect_parent(m_adder_i);
        schedule_node(m_sigmoid_i);

        m_sigmoid_f.connect_parent(m_adder_f);
        schedule_node(m_sigmoid_f);

        m_sigmoid_o.connect_parent(m_adder_o);
        schedule_node(m_sigmoid_o);

        m_tanh_g.connect_parent(m_adder_g);
        schedule_node(m_tanh_g);

        // Element-wise multiply of m_sigmoid_f and c_t_prev:
        m_multiply_f_and_c_t.create_input_port_this_name(m_sigmoid_f, "0");
        connect_input_to_contained_node("c_t", m_multiply_f_and_c_t, "1");
        schedule_node(m_multiply_f_and_c_t);

        // Element-wise multiply of i and g:
        m_multiply_i_g.create_input_port_this_name(m_sigmoid_i, "0");
        m_multiply_i_g.create_input_port_this_name(m_tanh_g, "1");
        schedule_node(m_multiply_i_g);

        // Add the outputs of m_multiply_f_and_c_t and m_multiply_i_g:
        m_adder_c_t.create_input_port_this_name(m_multiply_f_and_c_t, "0");
        m_adder_c_t.create_input_port_this_name(m_multiply_i_g, "1");
        schedule_node(m_adder_c_t);

        // Split c_t into two signals:
        m_splitter_2_output.connect_parent(m_adder_c_t);
        schedule_node(m_splitter_2_output);

        // First output of m_splitter_2_output will be output port "c_t"
        create_output_port(m_splitter_2_output, "0", "c_t");

        // Second output of m_splitter_2_output will be input to a tanh layer.
        m_tanh_c_t.create_input_port_parent_name(m_splitter_2_output, "1");
        schedule_node(m_tanh_c_t);

        // Multiply output gate and m_tanh_c_t to get h_t.
        m_multiply_h_t.create_input_port_this_name(m_sigmoid_o, "0");
        m_multiply_h_t.create_input_port_this_name(m_tanh_c_t, "1");
        schedule_node(m_multiply_h_t);

        // Split the h_t signal into 2 signals and send to h_t and y_t output ports.
        m_splitter_h_t_y_t.connect_parent(m_multiply_h_t);
        schedule_node(m_splitter_h_t_y_t);

        create_output_port(m_splitter_h_t_y_t, "0", "h_t");
        create_output_port(m_splitter_h_t_y_t, "1", "y_t");
    }

    // Optionally initialize parameters using the values recommended by Karpathy in [1].
    void initialize_params() {
        // fixme
        //MatrixF& W = get_weights();
        //randomize_uniform(W, -0.08f, 0.08f);
        //MatrixF& bias = get_bias();
        //randomize_uniform(bias, -0.08f, 0.08f);
        //MatrixF& bias_f = m_linear_x_t_f.get_bias();
        //set_value(bias_f, 1.0f);
    }

private:
    // Each contained node should be a private member of this class:
    SplitterNode m_splitter_4output_x_t;
    SplitterNode m_splitter_4output_h_t;
    LinearLayer m_linear_x_t_i;
    LinearLayer m_linear_h_t_i;
    LinearLayer m_linear_x_t_f;
    LinearLayer m_linear_h_t_f;
    LinearLayer m_linear_x_t_o;
    LinearLayer m_linear_h_t_o;
    LinearLayer m_linear_x_t_g;
    LinearLayer m_linear_h_t_g;
    AdderNode m_adder_i;
    AdderNode m_adder_f;
    AdderNode m_adder_o;
    AdderNode m_adder_g;
    ColumnActivationFunction m_sigmoid_i; // "input gate"
    ColumnActivationFunction m_sigmoid_f; // "forget gate"
    ColumnActivationFunction m_sigmoid_o; // "output gate"
    ColumnActivationFunction m_tanh_g; // "g gate"
    MultiplyerNode m_multiply_f_and_c_t; // Multiplyer to compute f * c_t_prev
    MultiplyerNode m_multiply_i_g; // Multiplyer to compute i * g
    AdderNode m_adder_c_t;
    SplitterNode m_splitter_2_output;
    ColumnActivationFunction m_tanh_c_t;
    MultiplyerNode m_multiply_h_t;
    SplitterNode m_splitter_h_t_y_t;
};


// LSTN node. This version uses 4 linear layers.
//
// Create a class that represents an LSTM node. This node can be replicated vertically to
// create an RNN with multiple layers and can be replicated horizontally (i.e., unrolled) to
// train the network.
//
// Input ports:
//               "x_t"
//               "h_t"
//               "c_t"
//
// Output ports:
//               "h_t": hidden output to next node (in next time slice)
//               "c_t": hidden output to next node (in next time slice)
//               "y_t": output to next layer or final output (in same time slice). This is the same value as h_t.
//
// Note: Since this will be a composite node (that is, a node that contains a subgraph of
// other nodes), all we need to do is the following:
//
// 1. For each contained node, add a corresponding private member variable (see below).
// 2. In the constructor initialization list, call the constructor of each contained node (most of
// them will only need a name, but some will require configuration parameters.
// 3. In the constructor body, do the following in order:
//     i. Connect each input port of this node to desired internal node input port.
//     ii. Connect internal nodes together however you want, adding each of them via add_node() in
//         the order that they should be called in the forward data propagation.
//     iii. Connect outputs of internal nodes to output ports of this node.
class LSTMNodeV2 : public CompositeNode {

public:

    // rnn_dim: dimension of internal RNN state.
    LSTMNodeV2(int rnn_dim, std::string name) :
        CompositeNode(name),
        // Note: Call the constructor of each contained node here, in the same order that
        // they appear as private member variables.
        m_concat_x_t_and_h_t {0, "ConcatNode: x_t concat h_t"},
        m_splitter_4output_1 {4, "Splitter: 4output 1"},
        m_linear_i {rnn_dim, "Linear: input"},
        m_linear_f {rnn_dim, "Linear: forget"},
        m_linear_o {rnn_dim, "Linear: output"},
        m_linear_g {rnn_dim, "Linear: g"},
        m_sigmoid_i{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: input gate"},
        m_sigmoid_f{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: forget gate"},
        m_sigmoid_o{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: output gate"},
        m_tanh_g {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh g activation"},
        m_multiply_f_and_c_t {"Multiplyer: f * c_t"},
        m_multiply_i_g {"Multiplyer: i * g"},
        m_adder_c_t {"Adder: c_t"},
        m_splitter_2_output{2, "Splitter 2 outputs"},
        m_tanh_c_t {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh c_t"},
        m_multiply_h_t {"Multiplyer: h_t"},
        m_splitter_h_t_y_t {2, "Splitter h_t, y_t"}
    {
        // This node is a "composite node" since it will contain a network subgraph.
        // We now spcecify the subgraph:

        // Note: add_node() must add the contained nodes in the same order that they should
        // be called in the forward data pass.

        // Concatenate x_t and h_t:
        connect_input_to_contained_node("x_t", m_concat_x_t_and_h_t, "0");
        connect_input_to_contained_node("h_t", m_concat_x_t_and_h_t, "1");
        schedule_node(m_concat_x_t_and_h_t);

        // Feed [x_t h_t] into splitter with 4 outputs (optional).
        m_splitter_4output_1.connect_parent(m_concat_x_t_and_h_t);
        schedule_node(m_splitter_4output_1);

        m_linear_i.create_input_port_parent_name(m_splitter_4output_1, "0");
        schedule_node(m_linear_i);

        m_linear_f.create_input_port_parent_name(m_splitter_4output_1, "1");
        schedule_node(m_linear_f);

        m_linear_o.create_input_port_parent_name(m_splitter_4output_1, "2");
        schedule_node(m_linear_o);

        m_linear_g.create_input_port_parent_name(m_splitter_4output_1, "3");
        schedule_node(m_linear_g);

        m_sigmoid_i.connect_parent(m_linear_i);
        schedule_node(m_sigmoid_i);

        m_sigmoid_f.connect_parent(m_linear_f);
        schedule_node(m_sigmoid_f);

        m_sigmoid_o.connect_parent(m_linear_o);
        schedule_node(m_sigmoid_o);

        m_tanh_g.connect_parent(m_linear_g);
        schedule_node(m_tanh_g);

        // Element-wise multiply of m_sigmoid_f and c_t_prev:
        m_multiply_f_and_c_t.create_input_port_this_name(m_sigmoid_f, "0");
        connect_input_to_contained_node("c_t", m_multiply_f_and_c_t, "1");
        schedule_node(m_multiply_f_and_c_t);

        // Element-wise multiply of i and g:
        m_multiply_i_g.create_input_port_this_name(m_sigmoid_i, "0");
        m_multiply_i_g.create_input_port_this_name(m_tanh_g, "1");
        schedule_node(m_multiply_i_g);

        // Add the outputs of m_multiply_f_and_c_t and m_multiply_i_g:
        m_adder_c_t.create_input_port_this_name(m_multiply_f_and_c_t, "0");
        m_adder_c_t.create_input_port_this_name(m_multiply_i_g, "1");
        schedule_node(m_adder_c_t);

        // Split c_t into two signals:
        m_splitter_2_output.connect_parent(m_adder_c_t);
        schedule_node(m_splitter_2_output);

        // First output of m_splitter_2_output will be output port "c_t"
        create_output_port(m_splitter_2_output, "0", "c_t");

        // Second output of m_splitter_2_output will be input to a tanh layer.
        m_tanh_c_t.create_input_port_parent_name(m_splitter_2_output, "1");
        schedule_node(m_tanh_c_t);

        // Multiply output gate and m_tanh_c_t to get h_t.
        m_multiply_h_t.create_input_port_this_name(m_sigmoid_o, "0");
        m_multiply_h_t.create_input_port_this_name(m_tanh_c_t, "1");
        schedule_node(m_multiply_h_t);

        // Split the h_t signal into 2 signals and send to h_t and y_t output ports.
        m_splitter_h_t_y_t.connect_parent(m_multiply_h_t);
        schedule_node(m_splitter_h_t_y_t);

        create_output_port(m_splitter_h_t_y_t, "0", "h_t");
        create_output_port(m_splitter_h_t_y_t, "1", "y_t");
    }

    // Optionally initialize parameters using the values recommended by Karpathy in [1].
    void initialize_params() {
        // fixme
        //MatrixF& W = get_weights();
        //randomize_uniform(W, -0.08f, 0.08f);
        //MatrixF& bias = get_bias();
        //randomize_uniform(bias, -0.08f, 0.08f);
        //MatrixF& bias_f = m_linear_f.get_bias();
        //set_value(bias_f, 1.0f);
    }

private:
    // Each contained node should be a private member of this class:
    ConcatNode m_concat_x_t_and_h_t;
    SplitterNode m_splitter_4output_1;
    LinearLayer m_linear_i;
    LinearLayer m_linear_f;
    LinearLayer m_linear_o;
    LinearLayer m_linear_g;
    ColumnActivationFunction m_sigmoid_i; // "input gate"
    ColumnActivationFunction m_sigmoid_f; // "forget gate"
    ColumnActivationFunction m_sigmoid_o; // "output gate"
    ColumnActivationFunction m_tanh_g; // "g gate"
    MultiplyerNode m_multiply_f_and_c_t; // Multiplyer to compute f * c_t_prev
    MultiplyerNode m_multiply_i_g; // Multiplyer to compute i * g
    AdderNode m_adder_c_t;
    SplitterNode m_splitter_2_output;
    ColumnActivationFunction m_tanh_c_t;
    MultiplyerNode m_multiply_h_t;
    SplitterNode m_splitter_h_t_y_t;
};


// LSTM node. This version uses 1 linear layer.
//
// Create a class that represents an LSTM node. This node can be replicated vertically to
// create an RNN with multiple layers and can be replicated horizontally (i.e., unrolled) to
// train the network.
//
// Input ports:
//               "x_t"
//               "h_t"
//               "c_t"
//
// Output ports:
//               "h_t": hidden output to next node (in next time slice)
//               "c_t": hidden output to next node (in next time slice)
//               "y_t": output to next layer or final output (in same time slice). This is the same value as h_t.
//
// Note: Since this will be a composite node (that is, a node that contains a subgraph of
// other nodes), all we need to do is the following:
//
// 1. For each contained node, add a corresponding private member variable (see below).
// 2. In the constructor initialization list, call the constructor of each contained node (most of
// them will only need a name, but some will require configuration parameters.
// 3. In the constructor body, do the following in order:
//     i. Connect each input port of this node to desired internal node input port.
//     ii. Connect internal nodes together however you want, adding each of them via add_node() in
//         the order that they should be called in the forward data propagation.
//     iii. Connect outputs of internal nodes to output ports of this node.
class LSTMNodeV3 : public CompositeNode {

public:

    // rnn_dim: dimension of internal RNN state.
    LSTMNodeV3(int rnn_dim, std::string name) :
        CompositeNode(name),
        // Note: Call the constructor of each contained node here, in the same order that
        // they appear as private member variables.
        m_concat_x_t_and_h_t {0, "ConcatNode: x_t concat h_t"},
        m_linear {4*rnn_dim, "Linear: i,f,o,g"},
        m_extractor_4 {{rnn_dim, rnn_dim, rnn_dim, rnn_dim}, "Extractor: i,f,o.g"},
        m_sigmoid_i{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: input gate"},
        m_sigmoid_f{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: forget gate"},
        m_sigmoid_o{ColumnActivationFunction::ACTIVATION_TYPE::sigmoid, "Sigmoid: output gate"},
        m_tanh_g {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh g activation"},
        m_multiply_f_and_c_t {"Multiplyer: f * c_t"},
        m_multiply_i_g {"Multiplyer: i * g"},
        m_adder_c_t {"Adder: c_t"},
        m_splitter_2_output{2, "Splitter 2 outputs"},
        m_tanh_c_t {ColumnActivationFunction::ACTIVATION_TYPE::tanh, "tanh c_t"},
        m_multiply_h_t {"Multiplyer: h_t"},
        m_splitter_h_t_y_t {2, "Splitter h_t, y_t"}
    {
        // This node is a "composite node" since it will contain a network subgraph.
        // We now spcecify the subgraph:

        // Note: add_node() must add the contained nodes in the same order that they should
        // be called in the forward data pass.

        // Concatenate x_t and h_t:
        connect_input_to_contained_node("x_t", m_concat_x_t_and_h_t, "0");
        connect_input_to_contained_node("h_t", m_concat_x_t_and_h_t, "1");
        schedule_node(m_concat_x_t_and_h_t);

        m_linear.connect_parent(m_concat_x_t_and_h_t);
        schedule_node(m_linear);

        m_extractor_4.connect_parent(m_linear);
        schedule_node(m_extractor_4);

        m_sigmoid_i.create_input_port_parent_name(m_extractor_4, "0");
        schedule_node(m_sigmoid_i);

        m_sigmoid_f.create_input_port_parent_name(m_extractor_4, "1");
        schedule_node(m_sigmoid_f);

        m_sigmoid_o.create_input_port_parent_name(m_extractor_4, "2");
        schedule_node(m_sigmoid_o);

        m_tanh_g.create_input_port_parent_name(m_extractor_4, "3");
        schedule_node(m_tanh_g);

        // Element-wise multiply of m_sigmoid_f and c_t_prev:
        m_multiply_f_and_c_t.create_input_port_this_name(m_sigmoid_f, "0");
        connect_input_to_contained_node("c_t", m_multiply_f_and_c_t, "1");
        schedule_node(m_multiply_f_and_c_t);

        // Element-wise multiply of i and g:
        m_multiply_i_g.create_input_port_this_name(m_sigmoid_i, "0");
        m_multiply_i_g.create_input_port_this_name(m_tanh_g, "1");
        schedule_node(m_multiply_i_g);

        // Add the outputs of m_multiply_f_and_c_t and m_multiply_i_g:
        m_adder_c_t.create_input_port_this_name(m_multiply_f_and_c_t, "0");
        m_adder_c_t.create_input_port_this_name(m_multiply_i_g, "1");
        schedule_node(m_adder_c_t);

        // Split c_t into two signals:
        m_splitter_2_output.connect_parent(m_adder_c_t);
        schedule_node(m_splitter_2_output);

        // First output of m_splitter_2_output will be output port "c_t"
        create_output_port(m_splitter_2_output, "0", "c_t");

        // Second output of m_splitter_2_output will be input to a tanh layer.
        m_tanh_c_t.create_input_port_parent_name(m_splitter_2_output, "1");
        schedule_node(m_tanh_c_t);

        // Multiply output gate and m_tanh_c_t to get h_t.
        m_multiply_h_t.create_input_port_this_name(m_sigmoid_o, "0");
        m_multiply_h_t.create_input_port_this_name(m_tanh_c_t, "1");
        schedule_node(m_multiply_h_t);

        // Split the h_t signal into 2 signals and send to h_t and y_t output ports.
        m_splitter_h_t_y_t.connect_parent(m_multiply_h_t);
        schedule_node(m_splitter_h_t_y_t);

        create_output_port(m_splitter_h_t_y_t, "0", "h_t");
        create_output_port(m_splitter_h_t_y_t, "1", "y_t");
    }

    // Optionally initialize parameters using the values recommended by Karpathy in [1].
    void initialize_params() {
        for (auto& param : m_linear.get_params()) {
            randomize_uniform(param->data, -0.08f, 0.08f);
        }
        // todo: set only the forget gate part of bias to 1.0f
    }

private:
    // Each contained node should be a private member of this class:
    ConcatNode m_concat_x_t_and_h_t;
    LinearLayer m_linear;
    ExtractorNode m_extractor_4; // extract output of linear layer into 4 vectors of equal size: i, f, o, g
    ColumnActivationFunction m_sigmoid_i; // "input gate"
    ColumnActivationFunction m_sigmoid_f; // "forget gate"
    ColumnActivationFunction m_sigmoid_o; // "output gate"
    ColumnActivationFunction m_tanh_g; // "g gate"
    MultiplyerNode m_multiply_f_and_c_t; // Multiplyer to compute f * c_t_prev
    MultiplyerNode m_multiply_i_g; // Multiplyer to compute i * g
    AdderNode m_adder_c_t;
    SplitterNode m_splitter_2_output;
    ColumnActivationFunction m_tanh_c_t;
    MultiplyerNode m_multiply_h_t;
    SplitterNode m_splitter_h_t_y_t;
};


// Choose which kind of LSTM node to use:

//using LSTMNode = LSTMNodeV1; // Basic naive LSTM node.
//using LSTMNode = LSTMNodeV2; // Might be slightly faster than V1.
using LSTMNode = LSTMNodeV3; // Might be slightly faster than V2..


// Contains an RNNNode + cost function representing 1 time slice.
//
// Input ports:
//               "x_t"
//               "h_t"
//
// Output ports:
//               "h_t": hidden output to next node (in next time slice)
//               "y_t": output of cost function

class RNN1LayerSlice : public CompositeNode {

public:

    RNN1LayerSlice(int rnn_dim, int output_dim, std::string name):
        CompositeNode(name),
        m_rnn_node {rnn_dim, "RNNNode"},
        m_linear_output {output_dim, "Linear output"},
        m_cost_func {"Cost function"},
        m_rnn_dim {rnn_dim},
        m_in_out_dim {output_dim}
    {
        connect_input_to_contained_node("x_t", m_rnn_node, "x_t");
        connect_input_to_contained_node("h_t", m_rnn_node, "h_t");
        schedule_node(m_rnn_node);
        create_output_port(m_rnn_node, "h_t", "h_t");

        m_linear_output.create_input_port_parent_name(m_rnn_node, "y_t");
        schedule_node(m_linear_output);

        m_cost_func.connect_parent(m_linear_output);
        schedule_node(m_cost_func);
        create_output_port_this_name(m_cost_func, "y_t");

        m_input_port_names = {"x_t"}; // Should not include the hidden input ports.
        m_hidden_port_names = {"h_t"};
        m_output_port_names = {"y_t"}; // Should not include the hidden output ports.
    }

    // Needed by SliceUnroller
    const std::vector<string>& get_hidden_port_names() const {
        return m_hidden_port_names;
    }

    // Needed by SliceUnroller
    // Includes all input ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_input_port_names() const {
        return m_input_port_names;
    }

    // Needed by SliceUnroller
    // Includes all output ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_output_port_names() const {
        return m_output_port_names;
    }

    void set_target_activations(const MatrixI& target_activations) {
        m_cost_func.set_target_activations(target_activations);
    }

    int get_rnn_dim() const {
        return m_rnn_dim;
    }

    int get_in_out_dim() const {
        return m_in_out_dim;
    }

    const MatrixF& get_softmax_output() const {
        return m_cost_func.get_softmax_output();
    }

    void initialize_params() {
        m_rnn_node.initialize_params();
    }

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
    std::map<std::string, std::unique_ptr<MatrixF>> make_hidden_state_matrices(int minibatch_size) {
        std::map<std::string, std::unique_ptr<MatrixF>> hidden_mat_map;
        for (std::string hidden_port : m_hidden_port_names) {
            hidden_mat_map.emplace(hidden_port, std::make_unique<MatrixF>(m_rnn_dim, minibatch_size));
        }
        return hidden_mat_map;
    }

private:

    RNNNode m_rnn_node;
    LinearLayer m_linear_output;
    CrossEntropyCostFunction m_cost_func;
    int m_rnn_dim;
    int m_in_out_dim;
    std::vector<string> m_input_port_names;
    std::vector<string> m_hidden_port_names;
    std::vector<string> m_output_port_names;
};



// Contains an LSTMNode + linear layer + cost function representing 1 time slice.
//
// Input ports:
//               "x_t"
//               "h_t"
//               "c_t"
//
// Output ports:
//               "h_t": hidden output to next node (in next time slice)
//               "c_t": hidden output to next node (in next time slice)
//               "y_t": output of cost function

class LSTM1LayerSlice : public CompositeNode {

public:

    LSTM1LayerSlice(int rnn_dim, int output_dim, std::string name):
        CompositeNode(name),
        m_rnn_node {rnn_dim, "LSTMNode"},
        m_linear_output {output_dim, "Linear output"},
        m_cost_func {"Cost function"},
        m_rnn_dim {rnn_dim},
        m_in_out_dim {output_dim}
    {
        connect_input_to_contained_node("x_t", m_rnn_node, "x_t");
        connect_input_to_contained_node("h_t", m_rnn_node, "h_t");
        connect_input_to_contained_node("c_t", m_rnn_node, "c_t");
        schedule_node(m_rnn_node);
        create_output_port(m_rnn_node, "h_t", "h_t");
        create_output_port(m_rnn_node, "c_t", "c_t");

        m_linear_output.create_input_port_parent_name(m_rnn_node, "y_t");
        schedule_node(m_linear_output);

        m_cost_func.connect_parent(m_linear_output);
        schedule_node(m_cost_func);
        create_output_port_this_name(m_cost_func, "y_t");

        m_input_port_names = {"x_t"}; // Should not include the hidden input ports.
        m_hidden_port_names = {"h_t", "c_t"};
        m_output_port_names = {"y_t"}; // Should not include the hidden output ports.
    }

    // Needed by SliceUnroller
    const std::vector<string>& get_hidden_port_names() const {
        return m_hidden_port_names;
    }

    // Needed by SliceUnroller
    // Includes all input ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_input_port_names() const {
        return m_input_port_names;
    }

    // Needed by SliceUnroller
    // Includes all output ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_output_port_names() const {
        return m_output_port_names;
    }

    void set_target_activations(const MatrixI& target_activations) {
        m_cost_func.set_target_activations(target_activations);
    }

    int get_rnn_dim() const {
        return m_rnn_dim;
    }

    int get_in_out_dim() const {
        return m_in_out_dim;
    }

    const MatrixF& get_softmax_output() const {
        return m_cost_func.get_softmax_output();
    }

    void initialize_params() {
        m_rnn_node.initialize_params();
    }

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
    std::map<std::string, std::unique_ptr<MatrixF>> make_hidden_state_matrices(int minibatch_size) {
        std::map<std::string, std::unique_ptr<MatrixF>> hidden_mat_map;
        for (std::string hidden_port : m_hidden_port_names) {
            hidden_mat_map.emplace(hidden_port, std::make_unique<MatrixF>(m_rnn_dim, minibatch_size));
        }
        return hidden_mat_map;
    }

private:

    LSTMNode m_rnn_node;
    LinearLayer m_linear_output;
    CrossEntropyCostFunction m_cost_func;
    int m_rnn_dim;
    int m_in_out_dim;
    std::vector<string> m_input_port_names;
    std::vector<string> m_hidden_port_names;
    std::vector<string> m_output_port_names;
};

// Contains an LSTMNode + dropout + linear layer + cost function representing 1 time slice.
//
// Input ports:
//               "x_t"
//               "h_t"
//               "c_t"
//
// Output ports:
//               "h_t": hidden output to next node (in next time slice)
//               "c_t": hidden output to next node (in next time slice)
//               "y_t": output of cost function

class LSTM1LayerSliceV2 : public CompositeNode {

public:

    //LSTM1LayerSliceV2(int rnn_dim, int output_dim, float dropout_prob_keep, std::string name):
    LSTM1LayerSliceV2(int rnn_dim, int output_dim, std::string name):
        CompositeNode(name),
        m_rnn_node {rnn_dim, "LSTMNode"},
        m_dropout1d_1 {0.5, "Dropout1D: 1"},
        m_linear_output {output_dim, "Linear output"},
        m_cost_func {"Cost function"},
        m_rnn_dim {rnn_dim},
        m_in_out_dim {output_dim}
    {
        connect_input_to_contained_node("x_t", m_rnn_node, "x_t");
        connect_input_to_contained_node("h_t", m_rnn_node, "h_t");
        connect_input_to_contained_node("c_t", m_rnn_node, "c_t");
        schedule_node(m_rnn_node);
        create_output_port(m_rnn_node, "h_t", "h_t");
        create_output_port(m_rnn_node, "c_t", "c_t");

        m_dropout1d_1.create_input_port_parent_name(m_rnn_node, "y_t");
        schedule_node(m_dropout1d_1);

        m_linear_output.connect_parent(m_dropout1d_1);
        schedule_node(m_linear_output);

        m_cost_func.connect_parent(m_linear_output);
        schedule_node(m_cost_func);
        create_output_port_this_name(m_cost_func, "y_t");

        m_input_port_names = {"x_t"}; // Should not include the hidden input ports.
        m_hidden_port_names = {"h_t", "c_t"};
        m_output_port_names = {"y_t"}; // Should not include the hidden output ports.
    }

    // Needed by SliceUnroller
    const std::vector<string>& get_hidden_port_names() const {
        return m_hidden_port_names;
    }

    // Needed by SliceUnroller
    // Includes all input ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_input_port_names() const {
        return m_input_port_names;
    }

    // Needed by SliceUnroller
    // Includes all output ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_output_port_names() const {
        return m_output_port_names;
    }

    void set_target_activations(const MatrixI& target_activations) {
        m_cost_func.set_target_activations(target_activations);
    }

    int get_rnn_dim() const {
        return m_rnn_dim;
    }

    int get_in_out_dim() const {
        return m_in_out_dim;
    }

    const MatrixF& get_softmax_output() const {
        return m_cost_func.get_softmax_output();
    }

    void initialize_params() {
        m_rnn_node.initialize_params();
    }

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
    std::map<std::string, std::unique_ptr<MatrixF>> make_hidden_state_matrices(int minibatch_size) {
        std::map<std::string, std::unique_ptr<MatrixF>> hidden_mat_map;
        for (std::string hidden_port : m_hidden_port_names) {
            hidden_mat_map.emplace(hidden_port, std::make_unique<MatrixF>(m_rnn_dim, minibatch_size));
        }
        return hidden_mat_map;
    }

private:

    LSTMNode m_rnn_node;
    Dropout1D m_dropout1d_1;
    LinearLayer m_linear_output;
    CrossEntropyCostFunction m_cost_func;
    int m_rnn_dim;
    int m_in_out_dim;
    std::vector<string> m_input_port_names;
    std::vector<string> m_hidden_port_names;
    std::vector<string> m_output_port_names;
};



// Contains two LSTMNode's + linear layer + cost function representing 1 time slice.
//
// Input ports:
//               "x_t"
//               "h_t_0"
//               "c_t_0"
//               "h_t_1"
//               "c_t_1"
//
//
// Output ports:
//               "h_t_0": hidden output to next node (in next time slice)
//               "c_t_0": hidden output to next node (in next time slice)
//               "h_t_1": hidden output to next node (in next time slice)
//               "c_t_1": hidden output to next node (in next time slice)
//               "y_t": output of cost function

class LSTM2LayerSlice : public CompositeNode {

public:

    LSTM2LayerSlice(int rnn_dim, int output_dim, std::string name):
        CompositeNode(name),
        m_rnn_node_0 {rnn_dim, "LSTMNode 0"},
        m_rnn_node_1 {rnn_dim, "LSTMNode 1"},
        m_linear_output {output_dim, "Linear output"},
        m_cost_func {"Cost function"},
        m_rnn_dim {rnn_dim},
        m_in_out_dim {output_dim}
    {
        // Layer 0 (bottom):
        connect_input_to_contained_node("x_t", m_rnn_node_0, "x_t");
        connect_input_to_contained_node("h_t_0", m_rnn_node_0, "h_t");
        connect_input_to_contained_node("c_t_0", m_rnn_node_0, "c_t");
        schedule_node(m_rnn_node_0);
        create_output_port(m_rnn_node_0, "h_t", "h_t_0");
        create_output_port(m_rnn_node_0, "c_t", "c_t_0");

        // Connect output of layer 0 to input of layer 1:
        m_rnn_node_1.create_input_port(m_rnn_node_0, "y_t", "x_t");

        // Layer 1 (stacked on top of layer 0):
        connect_input_to_contained_node("h_t_1", m_rnn_node_1, "h_t");
        connect_input_to_contained_node("c_t_1", m_rnn_node_1, "c_t");
        schedule_node(m_rnn_node_1);
        create_output_port(m_rnn_node_1, "h_t", "h_t_1");
        create_output_port(m_rnn_node_1, "c_t", "c_t_1");


        m_linear_output.create_input_port_parent_name(m_rnn_node_1, "y_t");
        schedule_node(m_linear_output);

        m_cost_func.connect_parent(m_linear_output);
        schedule_node(m_cost_func);
        create_output_port_this_name(m_cost_func, "y_t");

        m_input_port_names = {"x_t"}; // Should not include the hidden input ports.
        m_hidden_port_names = {"h_t_0", "c_t_0", "h_t_1", "c_t_1"};
        m_output_port_names = {"y_t"}; // Should not include the hidden output ports.
    }

    // Needed by SliceUnroller
    const std::vector<string>& get_hidden_port_names() const {
        return m_hidden_port_names;
    }

    // Needed by SliceUnroller
    // Includes all input ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_input_port_names() const {
        return m_input_port_names;
    }

    // Needed by SliceUnroller
    // Includes all output ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_output_port_names() const {
        return m_output_port_names;
    }

    void set_target_activations(const MatrixI& target_activations) {
        m_cost_func.set_target_activations(target_activations);
    }

    int get_rnn_dim() const {
        return m_rnn_dim;
    }

    int get_in_out_dim() const {
        return m_in_out_dim;
    }

    const MatrixF& get_softmax_output() const {
        return m_cost_func.get_softmax_output();
    }

    void initialize_params() {
        m_rnn_node_0.initialize_params();
        m_rnn_node_1.initialize_params();
    }


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
    std::map<std::string, std::unique_ptr<VariableF>> make_hidden_state_variables(int minibatch_size) {
        std::map<std::string, std::unique_ptr<VariableF>> hidden_mat_map;
        for (std::string hidden_port : m_hidden_port_names) {
            hidden_mat_map.emplace(hidden_port, std::make_unique<VariableF>(m_rnn_dim, minibatch_size));
        }
        return hidden_mat_map;
    }



private:

    LSTMNode m_rnn_node_0; // bottom node
    LSTMNode m_rnn_node_1; // top node
    LinearLayer m_linear_output;
    CrossEntropyCostFunction m_cost_func;
    int m_rnn_dim;
    int m_in_out_dim;
    std::vector<string> m_input_port_names;
    std::vector<string> m_hidden_port_names;
    std::vector<string> m_output_port_names;
};

// Contains two LSTMNode's + dropout + linear layer + cost function representing 1 time slice.
//
// Input ports:
//               "x_t"
//               "h_t_0"
//               "c_t_0"
//               "h_t_1"
//               "c_t_1"
//
//
// Output ports:
//               "h_t_0": hidden output to next node (in next time slice)
//               "c_t_0": hidden output to next node (in next time slice)
//               "h_t_1": hidden output to next node (in next time slice)
//               "c_t_1": hidden output to next node (in next time slice)
//               "y_t": output of cost function

class LSTM2LayerSliceV2 : public CompositeNode {

public:

    //LSTM2LayerSliceV2(int rnn_dim, int output_dim, float dropout_prob_keep, std::string name):
    LSTM2LayerSliceV2(int rnn_dim, int output_dim, std::string name):
        CompositeNode(name),
        m_rnn_node_0 {rnn_dim, "LSTMNode 0"},
        m_dropout1d_1 {0.5, "Dropout1D: 1"},
        m_rnn_node_1 {rnn_dim, "LSTMNode 1"},
        m_dropout1d_2 {0.5, "Dropout1D: 2"},
        m_linear_output {output_dim, "Linear output"},
        m_cost_func {"Cost function"},
        m_rnn_dim {rnn_dim},
        m_in_out_dim {output_dim}
    {
        // Layer 0 (bottom):
        connect_input_to_contained_node("x_t", m_rnn_node_0, "x_t");
        connect_input_to_contained_node("h_t_0", m_rnn_node_0, "h_t");
        connect_input_to_contained_node("c_t_0", m_rnn_node_0, "c_t");
        schedule_node(m_rnn_node_0);
        create_output_port(m_rnn_node_0, "h_t", "h_t_0");
        create_output_port(m_rnn_node_0, "c_t", "c_t_0");

        // Connect output of layer 0 to input of dropout layer:
        m_dropout1d_1.create_input_port_parent_name(m_rnn_node_0, "y_t");
        schedule_node(m_dropout1d_1);

        // Connect output of dropout layer to input of layer 1:
        m_rnn_node_1.create_input_port_this_name(m_dropout1d_1, "x_t");

        // Layer 1 (stacked on top of layer 0):
        connect_input_to_contained_node("h_t_1", m_rnn_node_1, "h_t");
        connect_input_to_contained_node("c_t_1", m_rnn_node_1, "c_t");
        schedule_node(m_rnn_node_1);
        create_output_port(m_rnn_node_1, "h_t", "h_t_1");
        create_output_port(m_rnn_node_1, "c_t", "c_t_1");

        m_dropout1d_2.create_input_port_parent_name(m_rnn_node_1, "y_t");
        schedule_node(m_dropout1d_2);

        m_linear_output.connect_parent(m_dropout1d_2);
        schedule_node(m_linear_output);

        m_cost_func.connect_parent(m_linear_output);
        schedule_node(m_cost_func);
        create_output_port_this_name(m_cost_func, "y_t");

        m_input_port_names = {"x_t"}; // Should not include the hidden input ports.
        m_hidden_port_names = {"h_t_0", "c_t_0", "h_t_1", "c_t_1"};
        m_output_port_names = {"y_t"}; // Should not include the hidden output ports.
    }

    // Needed by SliceUnroller
    const std::vector<string>& get_hidden_port_names() const {
        return m_hidden_port_names;
    }

    // Needed by SliceUnroller
    // Includes all input ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_input_port_names() const {
        return m_input_port_names;
    }

    // Needed by SliceUnroller
    // Includes all output ports except the hidden ports that are connected between
    // time slices.
    const std::vector<string>& get_output_port_names() const {
        return m_output_port_names;
    }

    void set_target_activations(const MatrixI& target_activations) {
        m_cost_func.set_target_activations(target_activations);
    }

    int get_rnn_dim() const {
        return m_rnn_dim;
    }

    int get_in_out_dim() const {
        return m_in_out_dim;
    }

    const MatrixF& get_softmax_output() const {
        return m_cost_func.get_softmax_output();
    }

    void initialize_params() {
        m_rnn_node_0.initialize_params();
        m_rnn_node_1.initialize_params();
    }

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
    std::map<std::string, std::unique_ptr<MatrixF>> make_hidden_state_matrices(int minibatch_size) {
        std::map<std::string, std::unique_ptr<MatrixF>> hidden_mat_map;
        for (std::string hidden_port : m_hidden_port_names) {
            hidden_mat_map.emplace(hidden_port, std::make_unique<MatrixF>(m_rnn_dim, minibatch_size));
        }
        return hidden_mat_map;
    }

private:

    LSTMNode m_rnn_node_0; // bottom node
    Dropout1D m_dropout1d_1;
    LSTMNode m_rnn_node_1; // top node
    Dropout1D m_dropout1d_2;
    LinearLayer m_linear_output;
    CrossEntropyCostFunction m_cost_func;
    int m_rnn_dim;
    int m_in_out_dim;
    std::vector<string> m_input_port_names;
    std::vector<string> m_hidden_port_names;
    std::vector<string> m_output_port_names;
};


// Choose what kind of RNN slice to use:
// Note: Some of these use dropout and some don't. You will need to modify the FullSlice constructor
// arguments below if switching between dropout/no-dropout versions.

//using FullSlice = RNN1LayerSlice; // 1-layer Vanilla RNN
//using FullSlice = LSTM1LayerSlice; // 1-layer LSTM
//using FullSlice = LSTM1LayerSliceV2; // 1-layer LSTM + dropout
using FullSlice = LSTM2LayerSlice; // 2-layer LSTM
//using FullSlice =   LSTM2LayerSliceV2; // 2-layer LSTM + dropout
// Check that gradients are computed correctly for LSTMNode.
void check_gradients_LSTMNode() {
    // Internal size of RNN:
    const int local_minibatch_size = 2;
    const int rnn_dim = 5;
    const int char_dim = 8;
    LSTMNode slice(rnn_dim, "Slice 0");

    const vector<int> x_t_extents = {char_dim, local_minibatch_size};
    MatrixF x_t_foward(x_t_extents);
    MatrixF x_t_backward(x_t_extents);

    const vector<int> h_t_extents = {rnn_dim, local_minibatch_size};
    std::map<std::string, std::vector<int>> input_port_extents_map1;
    input_port_extents_map1["x_t"] = x_t_extents;
    input_port_extents_map1["h_t"] = h_t_extents;
    input_port_extents_map1["c_t"] = h_t_extents;
    slice.check_jacobian_parameters(input_port_extents_map1);
    slice.check_jacobian_input_grad(input_port_extents_map1);
    cout << "Exiting." << endl;
    exit(0);
}

// This class samples from an RNN using a copy of one slice from the unrolled RNN.
//
// Note that for sampling text from an RNN, only one slice is required.
class SliceSampler {
public:

    // slice: A copy of a slice in the RNN to sample from. The parameters
    // should be the same as in the RNN.
    // idx_to_char: A map from character index to character.
    SliceSampler(FullSlice& slice, const std::map<int, char>& idx_to_char):
        m_slice {slice},
        m_idx_to_char {idx_to_char}
    {
        m_slice.set_train_mode(false); // Make sure slice is in evaluation mode.
    }

    // gen_char_count: number of characters to generate.
    void sample(int gen_char_count, float temperature) {
        const int minibatch_size = 1;
        // Create the hidden state matrices and connect them to input ports of the slice.
        auto name_to_hidden_vars = m_slice.make_hidden_state_variables(minibatch_size);
        for (auto& x : name_to_hidden_vars) {
            string port_name = x.first;
            auto& hidden_var = *x.second;
            m_slice.create_input_port(hidden_var, port_name);
        }

        VariableF input_var(m_slice.get_in_out_dim(), minibatch_size);
        m_slice.create_input_port(input_var, "x_t"); // Input port for slice.
        MatrixI target_activations(minibatch_size);
        m_slice.set_target_activations(target_activations);

        // Seed character: Set to whatever character you want.
        input_var.data(0) = 1.0f;

        MatrixF temperature_probs;
        cout << "Generating text:" << endl << endl;
        for (int i = 0; i < gen_char_count; ++i) {
            m_slice.forward();
            const MatrixF& probabilities = m_slice.get_softmax_output();
            // Use Karpathy's method for modifying the distribution temperature:
            temperature_probs = probabilities;
            element_wise_ln(temperature_probs, temperature_probs);
            scale(temperature_probs, temperature_probs, 1.0f/temperature);
            element_wise_exp(temperature_probs, temperature_probs);
            // Normalize:
            float prob_sum = sum(temperature_probs);
            scale(temperature_probs, temperature_probs, 1.0f/prob_sum);

            int index_chosen = sample_multinomial_distribution(temperature_probs);
            auto it = m_idx_to_char.find(index_chosen);
            if (it != m_idx_to_char.end()) {
                //cout << "char index: " << max_ind << endl;
                cout << it->second;
            } else {
                error_exit("Index not in char map!");
            }

            set_value(input_var.data, 0.0f);
            input_var.data(index_chosen) = 1.0f; // Character for next time slice.
            // Copy hidden state output into hidden state input for next slice.
            for (auto& x : name_to_hidden_vars) {
                string port_name = x.first;
                MatrixF& input_forward = (x.second)->data;
                const MatrixF& output_forward = m_slice.get_output_data(port_name);
                input_forward = output_forward;
            }
        }
        cout << endl;
    }

private:

    FullSlice& m_slice;
    const std::map<int, char>& m_idx_to_char;
};


void check_gradients_FullSlice() {
    // Internal size of RNN:
    const int local_minibatch_size = 2;
    const int rnn_dim = 5;
    const int output_dim = 3; // Number of unique characters
    const int local_char_dim = 4;
    FullSlice slice(rnn_dim, output_dim, "Slice 0");
    MatrixI target_activations(local_minibatch_size);
    slice.set_target_activations(target_activations);

    const vector<int> x_t_extents = {local_char_dim, local_minibatch_size};

    const vector<int> h_t_extents = {rnn_dim, local_minibatch_size};
    std::map<std::string, std::vector<int>> input_port_extents_map1;
    input_port_extents_map1["x_t"] = x_t_extents;
    for (auto hidden_port : slice.get_hidden_port_names()) {
        input_port_extents_map1[hidden_port] = h_t_extents;
    }
    slice.check_jacobian_parameters(input_port_extents_map1);
    slice.check_jacobian_input_grad(input_port_extents_map1);
    cout << "Exiting." << endl;
    exit(0);
}

void check_gradients_SliceUnroller_FullSlice() {
    const int local_minibatch_size = 2;
    const int local_rnn_dim = 4;
    const int local_char_dim = 3;
    const int local_num_slices = 5;
    const float dropout_prob_keep = 0.0f;
    //SliceUnroller<FullSlice> rnn(local_num_slices, "RNN", local_rnn_dim, local_char_dim, dropout_prob_keep);
    SliceUnroller<FullSlice> rnn(local_num_slices, "RNN", local_rnn_dim, local_char_dim);
    std::map<std::string, std::vector<int>> input_port_extents_map;
    input_port_extents_map.clear();
    const vector<int> h_t_extents = {local_rnn_dim, local_minibatch_size};
    FullSlice& slice = rnn.get_slice(0);
    for (auto hidden_port : slice.get_hidden_port_names()) {
        input_port_extents_map[hidden_port] = h_t_extents;
    }
    const vector<int> x_t_extents = {local_char_dim, local_minibatch_size};
    for (int i = 0; i < local_num_slices; ++i) {
        input_port_extents_map["x_t_" + std::to_string(i)] = x_t_extents;
    }
    MatrixI target_activations(local_minibatch_size);
    for (int i = 0; i < local_num_slices; ++i) {
        auto& slice = rnn.get_slice(i);
        slice.set_target_activations(target_activations);
    }
    rnn.check_jacobian_parameters(input_port_extents_map);
    rnn.check_jacobian_input_grad(input_port_extents_map);
    cout << "Exiting." << endl;
    exit(0);
}

// Train, evaluate and sample text from an RNN, such as Vanilla RNN, 1 and 2-layer LSTM.
void lstm_example() {
    cout << "lstm_example()" << endl;

    // Load text from file:
    ifstream text_stream("input.txt");
    stringstream buffer;
    buffer << text_stream.rdbuf();

    // This will contain the content of the file as one string:
    string text_data = buffer.str();
    if (text_data.size() == 0) {
        error_exit("Unable to read any text from the supplied input file!");
    }
    cout << "Number of characters in input text file: " << text_data.size() << endl;

    // Create the character->index map for the input text:
    map<char, int> char_to_idx = create_char_idx_map(text_data);

    // Now create the training and test sets.
    const float test_fraction = 0.05f; // Fraction of file to use for the test set.
    const int partition_index = static_cast<int>((1-test_fraction)*text_data.size());
    //const int partition_index = text_data.size() - 10000;
    cout << "Using characters 0 through " << partition_index << " for training set." << endl;
    cout << "Using characters " << partition_index+1 << " to " << text_data.size() << " for test set." << endl;

    const string training_text = text_data.substr(0, partition_index);
    const string testing_text = text_data.substr(partition_index);

    const int char_dim = char_to_idx.size(); // Number of unique characters.
    cout << "Number of unique characters = RNN internal state dimension: " << char_dim << endl;

    ///////////////////////////////////////////////////////////////////////////
    // Use suggested parameters from "Visualizing and Understanding Recurrent Networks" by Karpathy et al:
    //
    const int minibatch_size = 50;
    const int num_slices = 50; // 50 Length of RNN.
    const int rnn_dim = 256; // good size 256 or greater
    // Set up learning rates.
    //float learning_rate_weights = 1e-2f; // 1e-3 -  1e-5
    float rms_prop_rate_weights = 2e-3f; //2e-3f

    float weight_decay = 1e-5f; //
    const bool enable_weight_decay = true;
    //const float dropout_prob_keep = 0.5f; // Dropout probability of keeping an activation.

    // Check gradients:
    const bool check_gradients_slice = false;
    if (check_gradients_slice) {
        check_gradients_LSTMNode();
    }

    // Check gradients:
    const bool check_gradients_full_slice = false;
    if (check_gradients_full_slice) {
        check_gradients_FullSlice();
    }

    // Check gradients:
    const bool check_gradients_rnn = false;
    if (check_gradients_rnn) {
        check_gradients_SliceUnroller_FullSlice();
    }

    // RNN that will be used to training.
    //SliceUnroller<FullSlice> rnn_train(num_slices, "Training RNN", rnn_dim, char_dim, dropout_prob_keep);
    SliceUnroller<FullSlice> rnn_train(num_slices, "Training RNN", rnn_dim, char_dim);

    // RNN that will be used to testing.
    //SliceUnroller<FullSlice> rnn_test(num_slices, "Testing RNN", rnn_dim, char_dim, 0.0f);
    SliceUnroller<FullSlice> rnn_test(num_slices, "Testing RNN", rnn_dim, char_dim);

    // Get mini-batches of training data for the RNN.
    CharRNNMinibatchGetter train_getter(training_text, minibatch_size, num_slices);
    train_getter.set_char_to_idx_map(char_to_idx);

    FullSlice& slice_0 = rnn_train.get_slice(0);
    // Create the hidden state matrices and connect them to input ports of the slice.
    // For training:
    auto name_to_hidden_var_train = slice_0.make_hidden_state_variables(minibatch_size); // todo: make typedef/alias
    //auto name_to_hidden_backward_matrix_train = slice_0.make_hidden_state_matrices(minibatch_size);
    for (auto& x : name_to_hidden_var_train) {
        string port_name = x.first;
        VariableF& hidden_var = *x.second;
        rnn_train.create_input_port(hidden_var, port_name);
    }
    const int minibatch_size_test = 10; // Mini-batch size for testing.
    // For testing:
    auto name_to_hidden_var_test = slice_0.make_hidden_state_variables(minibatch_size_test); // todo: make typedef/alias
    //auto name_to_hidden_backward_matrix_test = slice_0.make_hidden_state_variables(minibatch_size_test);
    for (auto& x : name_to_hidden_var_test) {
        string port_name = x.first;
        VariableF& hidden_var = *x.second;
        rnn_test.create_input_port(hidden_var, port_name);
    }

    // Connect matrices from train_getter to the RNN:
    for (int i = 0; i < num_slices; ++i) {
        //cout << "slice = " << i << endl;
        VariableF& input_var = train_getter.get_input_forward_batch(i);
        rnn_train.create_input_port(input_var, "x_t_" + std::to_string(i));
        const MatrixI& target_activations = train_getter.get_output_class_index_batch(i);
        auto& slice = rnn_train.get_slice(i);
        slice.set_target_activations(target_activations);
    }

    // Get mini-batches of testing data for the RNN.
    CharRNNMinibatchGetter test_getter(testing_text, minibatch_size_test, num_slices);
    test_getter.set_char_to_idx_map(char_to_idx);
    // Connect matrices from test_getter to the RNN:
    for (int i = 0; i < num_slices; ++i) {
        //cout << "slice = " << i << endl;
        VariableF& input_var = test_getter.get_input_forward_batch(i);
        rnn_test.create_input_port(input_var, "x_t_" + std::to_string(i));
        const MatrixI& target_activations = test_getter.get_output_class_index_batch(i);
        auto& slice = rnn_test.get_slice(i);
        slice.set_target_activations(target_activations);
    }

    Accumulator train_loss_accumulator(minibatch_size);
    Accumulator test_loss_accumulator(minibatch_size_test);

    rnn_train.forward(); // Initialize network.
    rnn_test.forward();
    auto params = rnn_train.get_params();
    cout << "params size = " << params.size() << endl;

    Updater weights_updater(params);
    //float learning_rate_weights = 100.0f;
    //weights_updater.set_mode_constant_learning_rate(learning_rate_weights); //
    //weights_updater.set_mode_rmsprop_momentum(rms_prop_rate_weights, 0.9f, 0.9f);
    weights_updater.set_mode_rmsprop(rms_prop_rate_weights, 0.9f);
    weights_updater.set_flag_weight_decay(weight_decay, enable_weight_decay);
    //cout << "W size = " << W.size() << endl;
    
    FullSlice slice_clone(rnn_dim, char_dim, "Slice Clone");
    slice_clone.set_shared(rnn_train.get_slice(0));
    SliceSampler slice_sampler(slice_clone, train_getter.get_idx_to_char_map());

    // The main training loop:
    int train_epochs = 0;
    rnn_train.set_train_mode(true);
    rnn_test.set_train_mode(false);
    int batch_counter = 0;
    const int max_eochs = 50;
    while (train_epochs < max_eochs) {
        cerr << ".";
        bool end_epoch = train_getter.next(); // Get next training mini-batch
        rnn_train.forward();

        // Copy hidden state from end of RNN to the 1st slice.
        for (auto& x : name_to_hidden_var_train) {
            string port_name = x.first;
            MatrixF& input_forward = x.second->data;
            const MatrixF& output_forward = rnn_train.get_output_data(port_name);
            input_forward = output_forward;
        }

        for (int i = 0; i < num_slices; ++i) {
            const MatrixF& cost_output = rnn_train.get_output_data("y_t_" + std::to_string(i));
            train_loss_accumulator.accumulate(cost_output[0]);
        }

        rnn_train.back_propagate();
        for (int i = 0; i < params.size(); ++i) {
            auto& W_grad = params.at(i)->grad;
            scale(W_grad, W_grad, 1.0f/static_cast<float>(num_slices));
            clip_to_range(W_grad, -5.0f, 5.0f);
        }
        weights_updater.update();

        if ((batch_counter % 10) == 0) {
            cout << batch_counter << " (epoch " << train_epochs << ") Train loss: " << train_loss_accumulator.get_mean() << endl;
            train_loss_accumulator.reset();
        }

        if (end_epoch) {
            //print_stats(W_grad, "W_grad");
            cout << endl << "---------------" << endl;
            cout << "Batches processed: " << batch_counter << endl;
            //network.print_paramater_stats(); // enable for debugging info
            test_loss_accumulator.reset();
            // Copy parameters from training RNN to test RNN.
            rnn_train.copy_params_to(rnn_test);
            bool done = false;
            // Zero out the initial hidden state vectors:
            for (auto& x : name_to_hidden_var_test) {
                string port_name = x.first;
                MatrixF& input_forward = x.second->data;
                set_value(input_forward, 0.0f);
            }
            while (!done) {
                done = test_getter.next(); // Get next test mini-batch
                rnn_test.forward();
                // Copy hidden state from end of RNN to the 1st slice.
                for (auto& x : name_to_hidden_var_test) {
                    string port_name = x.first;
                    MatrixF& input_forward = x.second->data;
                    const MatrixF& output_forward = rnn_test.get_output_data(port_name);
                    //copy_matrix(input_forward, output_forward);
                    input_forward = output_forward;
                }

                for (int i = 0; i < num_slices; ++i) {
                    const MatrixF& cost_output = rnn_test.get_output_data("y_t_" + std::to_string(i));
                    test_loss_accumulator.accumulate(cost_output[0]);
                }
            }
            cout << "Test loss/example: " << test_loss_accumulator.get_mean() << endl;
            const float temperature = 0.8f;
            slice_sampler.sample(1000, temperature);
        }

        if (end_epoch) {
            if (train_epochs > 10) {
                // Decay learning rate:
                rms_prop_rate_weights *= 0.95f;
                //rms_prop_rate_bias *= 0.95f;
                //weights_updater.set_mode_rmsprop_momentum(rms_prop_rate_weights, 0.9f, 0.9f);
                weights_updater.set_mode_rmsprop(rms_prop_rate_weights, 0.9f);
                //bias_updater.set_mode_rmsprop_momentum(rms_prop_rate_bias, 0.9f, 0.9f);
                //bias_updater.set_mode_rmsprop(rms_prop_rate_bias, 0.9f);
                cout << endl << "Setting new learning rate: " << rms_prop_rate_weights << endl;
            }
            // Zero out the initial hidden state vectors:
            for (auto& x : name_to_hidden_var_train) {
                string port_name = x.first;
                MatrixF& input_forward = x.second->data;
                set_value(input_forward, 0.0f);
            }
            train_epochs++;
        }
        batch_counter++;
    }
}


}
