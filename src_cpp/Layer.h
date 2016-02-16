#ifndef _LAYER_H
#define _LAYER_H
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
#include "Node.h"
#include <string>
#include <iostream>
#include "Utilities.h"
#include "Constants.h"

namespace kumozu {

  /**
   * This is an abstract base class for a layer in a network.
   *
   * A Layer is a Node that has 1 input and 1 output. Because of this, the use of ports is optional for a Layer.
   *
   * This class is intended to model a layer that has the following characteristics:
   *
   * - A notion of "input" activations. The input typically corresponds to one mini-batch of data. We use the convention
   *   that the input activations are not considered an internal state of a Layer. Since it is common to connect layer
   *   objects in sequence, it is not necessary for each layer to contain both its inputs and outputs as member variables. If this were
   *   the case, it would be necessary to always copy the values from a layer's output into the inputs of the
   *   next layer. This copy step can be eliminated if each layer only contains inputs or outputs as member variables but not both.
   *   We made the arbitrary decision that a layer will only contain its outputs since it can just get a reference
   *   to the output of the previous layer and use this as its "input." Thus, a layers input activations correspond to
   *   to the output activations of the preceding layer or to some other external object.
   *
   * - A notion of "output" activations that are a function of the input and optional parameters:
   *   output = f(input, parameters). This function corresponds to calling "forward()".
   *   The specific functional form is not specified here. Use a subclass
   *   to impose a specific functional relationship.
   *   This base Layer contains a member variable for the output activations and member function get_output_forward() to
   *   obtain a referecne.
   *
   * - Optional weights parameters.
   * - Optional bias parameters.
   * - Gradients of the parameters with respect to the output error can be computed by a calling the
   *   "back_propagate_paramater_gradients()" function. However, it is usually better to call back_propagate() instead.
   * - Gradients of the "input" activations with respect to the output error can be computed by calling a
   *   method of this class. This corresponds to the "back_propagate_deltas()" function.
   *   However, it is usually better to call back_propagate() instead.
   *
   * Input and output ports:
   *
   * By default, a Layer will automatically create its own output port with the name DEFAULT_OUTPUT_PORT_NAME. However, a Layer will not automatically create
   * its input port since a reference to the input activations and errors is needed in order to do so. These references are often not availabe
   * at the time a Layer is created. However, it easy to create the input port by using the function that connects to an output port
   * of a parent Node and automatically creates a corresponding input port with default name DEFAULT_INPUT_PORT_NAME. Alternatively, a user can
   * also choose to give the input port any desired name. To summarize, a Layer will have either 0 or 1 inputs ports and 1 output port.
   * The input port, if it exists will either have the default name DEFAULT_INPUT_PORT_NAME or a user-defined name. The output port will always
   * have the default name DEFAULT_OUTPUT_PORT_NAME. // todo: make it required to always have 1 input port and remove deprecated old api.
   *
   * Correctness checking:
   *
   * Funtions are also provided to check that the numerically-computed Jacobian roughly matches the gradients/Jacobian
   * computed by the member functions. Whenever a new subclass is created, it is advised to create unit tests that
   * call these Jacobian-checking functions to verify that the implementation is correct. Note that a subclass does
   * not need to override the Jacbian checker since it should be able to use the base class version.
   *
   * Typical usage:
   *
   * Create a desired network by instantiating various subclasses of this class and connecting them together, such as in
   * a sequential network, for example. The network is not yet fully defined because the size of the input activations to
   * the various layers has not yet been specified. That is, the constructor of a Layer instance does not take any
   * parameters that say anything about the sizes of the input activations. Rather, in this framework, the network
   * becomes fully specified when a mini-batch of input data is passed through the network in the forward direction
   * for the first time. This corresponds to calling the "forward(input_forward)" function. This has two advantages
   * over the alternative of supplying the input extents to the constructor:
   *
   * 1. The user does not need to be concerned with figuring out the correct size of the input extents when creating a network.
   * Note that deep convolutional
   * networks typically use pooling layers that reduce the sizes of various dimensions, and some forms of convolutional layers can
   * have output extents that differ from the input extents. Some activation function, such as maxout also reduce the
   * output dimensions. Because of this, it can require some effort to
   * manually determine the correct input extents for a given layer, which is error prone. For example, if one decides to remove
   * a pooling layer in the middel of a very deep network, the later layer definitions do not need to be modified at all since the
   * network will automatically determine the required input extents itself when "forward()" is called for the first time.
   *
   * 2. The input extents can change during runtime. For example, one might want to use one mini-batch size (for example,
   * 128) for training and another mini-batch size (for example, 200) for testing. This framework allows the same network
   * to be used for both training and testing since the network will automatically reinitialize itself whenver its
   * input extents change.
   *
   * An CostFunction object "cost" will also be needed in order to compute the output cost and error gradients for back-propagation.
   *
   * To train a network, call
   *
   * network.forward();
   * cost.foward();
   * cost.back_propagate();
   * network.back_propagate();
   * in a loop until learning is finished.
   *
   * Subclasses:
   *
   * A subclass must override forward_propagate(), back_propagate_deltas(), and reinitialize() and my optionally override any of the other virtual
   * functions as well. Note that a subclass should not override back_propagate() since the base class implementation
   * calls both forward_propagate() and back_propagate_deltas().
   *
   * A subclass must provide an implementation of reinitialize(), which will be called by the base class forward() whenever
   * the supplied input activation sizes change. The purpose of the reinitialize() function is to initialize the various
   * matrices (output activations, parameters, temporary storage etc) that are dependent on the size of the input
   * activations. Otherwise, if the implementation does nothing, all matrices (weights, bias, output activations etc.)
   * in this base class will have size 0 by default. This may be fine for layers that do not contain any parameters, but
   * it is expected that most layers will at least want to use their output activations and so initializing them
   * to some nonzero size is recommended.
   */
   class Layer : public Node {

  public:

  Layer(std::string name) :
    Node(name)
          {
	    // Create the 1 output port.
	    if (get_output_port_count() == 0) {
	      // Create the output port with default name.
	      create_output_port(m_output_forward, m_output_backward, DEFAULT_OUTPUT_PORT_NAME); 
	    }
          }

          
          /**
           * Compute the gradients for the paramaters (i.e.,  weights and bias).
           *
           * The convention is that this function should be called after back_propagate_deltas(). However,
           * it is normally not recommended to call this function directly since it will be called by
           * back_propagate().
           *
           * This function has a default implementation that does nothing because not every subclass
           * of Layer will contain parameters. If a subclass contains parameters, it should overrid
           * this function it back-propagate the parameter gradients.
           *
           * The output error (that is, "output deltas") must have already been updated before
           * calling this method. Note that a reference to the output deltas can be obtained by
           * calling get_output_backward(). Otherwise, the error gradients will not be back-propagated
           * correctly.
           *
           * The weights gradients will typically correspond to m_W_grad and the bias gradients
           * will typically correspond to m_bias_grad. However, a subclass is free to use
           * different member variables, if desired.
           *
	   * Layer subclasses should override this function.
           */
          virtual void back_propagate_paramater_gradients(const MatrixF& input_forward) {}

	  /**
	   * Subclasses should not override this function.
	   */
	  virtual void back_propagate_paramater_gradients() {
	    back_propagate_paramater_gradients(get_input_port_forward());
	  }

	  /**
	   * Subclasses should not override this function.
	   */
          virtual void back_propagate_deltas() {
	    // Get input deltas and forward to the function the subclasses will override.
            back_propagate_deltas(get_input_port_backward(), get_input_port_forward());
          }

          /**
           * Back-propagate errors to compute new values for input_backward.
           *
           * The convention is that this function should be called before back_propagate_paramater_gradients().
           * This function can also be called without calling back_propagate_paramater_gradients() if it is
           * only desired to back-propagate the error deltas from the network output to its inputs. However,
           * it is normally not recommended to call this function directly since it will be called by
           * back_propagate().
           *
           * The output error (that is, "output deltas") must have already been updated before
           * calling this method. Note that a reference to the output deltas can be obtained by
           * calling get_output_backward(). Otherwise, the error gradients will not be back-propagated
           * correctly.
           *
           * subclasses must implement this function.
           */
          virtual void back_propagate_deltas(MatrixF& input_backward, const MatrixF& input_forward) = 0;

         
          
   protected: // fixme: remove protected?

          /**
           *
           * Compute the output activations as a function of input activations.
           *
           * The output activations can then be obtained by calling get_output_forward().
           *
           * If this is the first time calling this function, the network will be initialized using
           * the extents of the supplied input activations. The network will also be reinitialized
           * any time that the extents of the supplied input activations change (although it is
           * uncommon and bad for performance for this to occur frequently at runtime).
           *
           * Implementation note:
           * This function will first check to see if the most recent input extents (i.e., sizes of
           * the input activation dimensions from the previous call) match the
           * supplied matrix. If they differ, various internal state matrices will be reinitialized
           * to be consistent with the extents of the supplied input activations. The parameter
           * matrices, if any, will only be reinitialized when they are required to change size.
           * subclasses are advised to implement this behavior.
           */
          virtual void forward_propagate(const MatrixF& input_forward) = 0;

	  /**
	   * Note: subclasses of Layer should not override this function but instead should
	   * implement the other forward_propagate() that takes an argument.
	   */
          virtual void forward_propagate() {
            // Just forward to input activations from the inport port.
	    // Because of this, a subclass only needs to implement the forward_propagate() that takes an argument.
	    forward_propagate(get_input_port_forward());
          }

          /*
           * Reinitialize the layer based on the supplied new input extent values.
           * This must be called before the layer can be used and must also be called whenever the
           * input extents change during runtime.
           *
           * Every subclass must provide an implementation for this function.
           *
           * Note that a call to this function can be somewhat expensive since several matrices (such
           * as the output activations and possibly the parameters as well) might need to
           * be reallocated and initialized. As a general convention, parameter matrices should only
           * be reallocated and/or reinitialized if their size changes.
           *
           * By default the weights (m_W), weight gradient (m_W_grad), bias (m_bias), bias gradient (m_bias_grad),
           * output activations (m_output_forward), and output error (m_output_backward) will be initialized
           * by this base class to have size 0. Thus, a subclass should use this function to realocate
           * and initialize any of these matrices that is wishes to use.
           */
          virtual void reinitialize(std::vector<int> input_extents) = 0;

	  /**
	   * Reinitialize the Layer.
	   *
	   * If this Layer does not already have an output port, create an output port with
	   * the default name that will be associated with the output activations and output errors matrices.
	   *
	   * Note: subclasses should not override this version of reinitialize() but should instead implement
	   * the other version that takes an input_extents argument.
	   */
          virtual void reinitialize() {
            // Just forward the input extents from the input port.
	    reinitialize(get_input_port_forward().get_extents());
          }

          MatrixF m_output_forward; // associated with the default output port
          MatrixF m_output_backward; // associated with the default output port

   private:
	  std::vector<int> m_input_extents;

  };

}


#endif /* _LAYER_H */
