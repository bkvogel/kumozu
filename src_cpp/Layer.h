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
#include <string>
#include <iostream>

namespace kumozu {

  /*
   * This is an abstract base class for a layer in a network.
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
   *   This base Layer contains a member variable for the output activations and member function get_output() to
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
   * for the first time. This corresponds to calling the "forward(input_activations)" function. This has two advantages
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
  class Layer {

  public:

  Layer(std::string name) : m_layer_name {name},
      m_is_train {false},
        m_is_initialized {false},
          m_enable_bias {true},
            m_epsilon {1e-4f},
              m_pass_relative_error {2e-2f} // Minimum relative error needed to pass Jacobian checker.

              {

              }

              virtual ~Layer(){}



              /*
               * Compute the output activations as a function of input activations.
               *
               * The output activations can then be obtained by calling get_output().
               *
               * If this is the first time calling this function, the network will be initialized using
               * the extents of the supplied input activations. The network will also be reinitialized
               * any time that the extents of the supplied input activations change (although it is
               * uncommon and bad for performance for this to occur frequently at runtime).
               *
               * A more common motivation of this feature is that the mini-batch size might ocassionally
               * change. For example, one might decide to train the network (which is an instance of Layer)
               * with some mini-batch size M >> 1.
               * Once the network has been trained, one might then wish to supply examples to the network
               * one at a time, in which case the new M will be 1. In such a case, when the mini-batch
               * size changes from M to 1, the network/layer will automatically be reinitialized during the
               * call to this function.
               *
               * The layer is considered to be "initialized" after this function returns.
               *
               * This function will call reinitialize() and forward_propagate(), which every subclass must
               * imlement. Note that this function is not virtual because it should not be overriden by a subclass.
               *
               * Implementation note:
               * This function will first check to see if the most recent input extents (i.e., sizes of
               * the input activation dimensions from the previous call) match the
               * supplied matrix. If they differ, various internal state matrices will be reinitialized
               * to be consistent with the extents of the supplied input activations.
               * Important: The parameter
               * matrices, if any, should only be reinitialized when they are required to change size.
               * subclasses are advised to implement this behavior.
               */
              void forward(const MatrixF& input_activations) {
                if (input_activations.get_extents() != m_input_extents) {
                  std::cout << std::endl << "Initializing " << m_layer_name << ":" << std::endl;
                  m_input_extents = input_activations.get_extents();
                  reinitialize(input_activations.get_extents());
                }
                forward_propagate(input_activations);
                m_is_initialized = true;
              }

              /*
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
               * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
               * correctly.
               *
               * The weights gradients will typically correspond to m_W_grad and the bias gradients
               * will typically correspond to m_bias_grad. However, a subclass is free to use
               * different member variables, if desired.
               *
               */
              virtual void back_propagate_paramater_gradients(const MatrixF& input_activations) {}

              /*
               * Back-propagate errors to compute new values for input_error.
               *
               * The convention is that this function should be called before back_propagate_paramater_gradients().
               * This function can also be called without calling back_propagate_paramater_gradients() if it is
               * only desired to back-propagate the error deltas from the network output to its inputs. However,
               * it is normally not recommended to call this function directly since it will be called by
               * back_propagate().
               *
               * The output error (that is, "output deltas") must have already been updated before
               * calling this method. Note that a reference to the output deltas can be obtained by
               * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
               * correctly.
               *
               * subclasses must implement this function.
               */
              virtual void back_propagate_deltas(MatrixF& input_error) = 0;

              /*
               * Perform a full back-propagation pass through the network.
               *
               * It is assumed that forward_propagate() has already been called with the same
               * "input_activations".
               *
               * This updates the input error gradients (i.e., input deltas) and also updates the parameter gradients
               * (weights and bias) for layers that contain and use such parameters.
               *
               * Note: This function is not virtual becuase subclasses should have no need to override it.
               * subclasses should instead override back_propagate_deltas() and back_propagate_paramater_gradients() since
               * they will be called by this function.
               *
               */
              void back_propagate(MatrixF& input_error, const MatrixF& input_activations) {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": back_propagate() called before being initialized." << std::endl;
                  exit(1);
                }
                back_propagate_deltas(input_error);
                back_propagate_paramater_gradients(input_activations);
              }

              /*
               * Set the mode of this layer to either train or test/evaluate.
               *
               * Some layers, such as dropout layers, behave differently between training
               * and evaluation modes. Most other sub-layers can ignore this mode, however.
               *
               * The default value is false (that is, use evaluation mode be default).
               */
              virtual void set_train_mode(bool is_train) {
                m_is_train = is_train;
                //std::cout << m_layer_name << ": set_train_mode() = " << m_is_train << std::endl;
              }


              /*
               * Return a reference to the weights matrix.
               *
               * This funtion should not be called until a forward pass has been performed.
               */
              MatrixF& get_weights() {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_weights() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_W;
              }

              /*
               * Return a reference to the weights matrix.
               *
               * This funtion should not be called until a forward pass has been performed.
               */
              const MatrixF& get_weights() const {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_weights() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_W;
              }

              /*
               * Return a reference to the bias vector.
               *
               * This funtion should not be called until a forward pass has been performed.
               */
              MatrixF& get_bias() {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_bias() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_bias;
              }

              /*
               * Return a reference to the bias vector.
               *
               * This funtion should not be called until a forward pass has been performed.
               */
              const MatrixF& get_bias() const {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_bias() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_bias;
              }

              /*
               * Return a reference to the weight gradient matrix.
               */
              MatrixF& get_weight_gradient() {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_weight_gradient() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_W_grad;
              }

              /*
               * Return a reference to the weight gradient matrix.
               */
              const MatrixF& get_weight_gradient() const {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_weight_gradient() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_W_grad;
              }

              /*
               * Return a reference to the bias gradient vector.
               */
              MatrixF& get_bias_gradient() {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_bias_gradient() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_bias_grad;
              }

              /*
               * Return a reference to the bias gradient vector.
               */
              const MatrixF& get_bias_gradient() const {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_bias_gradient() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_bias_grad;
              }

              /*
               * For modules that contain bias parameters, enable or disable the use of such parameters.
               *
               * By default, bias is enabled.
               *
               * is_enable: If true, bias will be enabled. If false, modules that contain bias parameters will
               * ignore such parameters and/or use 0-valued parameters even if an outside class attempts to modify the
               * bias values.
               */
              virtual void enable_bias(bool is_enable) {
                m_enable_bias = is_enable;
              }

              /*
               * Return true if the layer has already been initialized. Otherwise return false.
               *
               * Note that initialized is performed by the first call to forward().
               */
              bool is_initialized() const {
                return m_is_initialized;
              }

              /*
               * Return a reference to this layer's output activations.
               */
              virtual const MatrixF& get_output() const {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_output() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_output_activations;
              }

              /*
               * Return a reference to this layer's output activations.
               */
              virtual MatrixF& get_output() {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_output() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_output_activations;
              }

              /*
               * Return a reference to this layer's output deltas. These activations represent the
               * gradient of the output activations (that is, errors for the output activations)
               * that are computed during the back-propagation step.
               */
              virtual const MatrixF& get_output_deltas() const {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_output_deltas() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_output_error;
              }

              /*
               * Return a reference to this layer's output deltas. These activations represent the
               * gradient of the output activations (that is, errors for the output activations)
               * that are computed during the back-propagation step.
               */
              virtual MatrixF& get_output_deltas() {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_output_deltas() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_output_error;
              }

              /*
               * Return the extents of the output activations.
               * This information is typically passed to the constructor of the next layer in the network.
               */
              virtual std::vector<int> get_output_extents() const {
                if (!m_is_initialized) {
                  std::cerr << m_layer_name <<  ": get_output_extents() called before being initialized." << std::endl;
                  exit(1);
                }
                return m_output_activations.get_extents();
              }

              /*
               * Check that the Jacobian computed using the finite differences method agrees with
               * the Jacobian computed using the gradient back-propagation member functions.
               *
               * Note: not virtual because derived classes should be able to use the base implementation.
               */
              void check_jacobian_weights(std::vector<int> input_extents);

              /*
               * Check that the Jacobian computed using the finite differences method agrees with
               * the Jacobian computed using the gradient back-propagation member functions.
               *
               * Note: not virtual because derived classes should be able to use the base implementation.
               */
              void check_jacobian_bias(std::vector<int> input_extents);

              /*
               * Check that the Jacobian computed using the finite differences method agrees with
               * the Jacobian computed using the gradient back-propagation member functions.
               *
               * Note: not virtual because derived classes should be able to use the base implementation.
               */
              void check_jacobian_input_error(std::vector<int> input_extents);

              /*
               * This base implementation prints information for weights, bias, output activations,
               * and corresponding gradients.
               * Subclasses may override to provide additional or different information.
               */
              virtual void print_paramater_stats() const;

              std::string get_name() const {
                return m_layer_name;
              }

              /*
               * Save parameters to a file withe the prefix given
               * by the supplied name.
               *
               * This base class function saves the weights and bias parameters only.
               * Subclasses may override to save additional parameters if desired.
               *
               */
              virtual void save_parameters(std::string name) const;

              /*
               * Load learned parameters from a file. The string name should be
               * the same that was used to save the parameters when
               * save_learning_info() was called.
               *
               * This base class function loads the weights and bias parameters only.
               * Subclasses may override to load additional parameters if desired.
               */
              virtual void load_parameters(std::string name);



  protected:



              /*
               *
               * Compute the output activations as a function of input activations.
               *
               * The output activations can then be obtained by calling get_output().
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
              virtual void forward_propagate(const MatrixF& input_activations) = 0;


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
               * output activations (m_output_activations), and output error (m_output_error) will be initialized
               * by this base class to have size 0. Thus, a subclass should use this function to realocate
               * and initialize any of these matrices that is wishes to use.
               */
              virtual void reinitialize(std::vector<int> input_extents) = 0;

              std::string m_layer_name;
              MatrixF m_output_activations;
              MatrixF m_output_error;
              MatrixF m_W; // Parameter weights matrix.
              MatrixF m_W_grad; // Gradient of parameter weights matrix.
              MatrixF m_bias; // Bias parameters.
              MatrixF m_bias_grad; // Gradient of bias parameters.
              std::vector<int> m_input_extents;
              bool m_is_train;
              bool m_is_initialized;
              bool m_enable_bias;

  private:

              const float m_epsilon;
              const float m_pass_relative_error;

  };

}


#endif /* _LAYER_H */
