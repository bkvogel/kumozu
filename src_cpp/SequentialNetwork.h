#ifndef _SEQUENTIALNETWORK_H
#define _SEQUENTIALNETWORK_H
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
#include "PlotUtilities.h"
#include "Layer.h"
#include <functional>

namespace kumozu {

  /*
   * This is a base class for a multi-layer feed-forward network.
   *
   * This class corresponds to a network in which one or more contained layers are connected serially to form
   * a feed-forward network.
   *
   * Usage:
   *
   * Create an instance of this class and add layers using the add_layer() function.
   *
   * This network can then be used just as any other instance of Layer.
   *
   */
  class SequentialNetwork : public Layer {

  public:

  SequentialNetwork(std::string name) :
    Layer(name)
    {

    }

    virtual ~SequentialNetwork(){}

    /*
     * Add a layer to the network.
     *
     * The network is initially empty. Layers are connected in series in the
     * order that they were added. Thus, the first layer added becomes the input
     * layer and the final layer added becomes the output layer.
     */
    void add_layer(Layer& layer) {
      //m_layers.push_back(std::ref(layer)); // This also works
      m_layers.push_back(layer);
    }

    /*
     * Return a reference to the n'th Layer object in the network.
     */
    Layer& get_layer(int n);

    const Layer& get_layer(int n) const;

    /*
     * Compute the gradients for W and bias.
     *
     * This updates m_W_grad and m_bias_grad.
     *
     * The output error (that is, "output deltas") must have already been updated before
     * calling this method. Note that a reference to the output deltas can be obtained by
     * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
     * correctly.
     *
     * Also, back_propagate_deltas() should have already been called before calling this function.
     */
    virtual void back_propagate_paramater_gradients(const MatrixF& input_activations);


    /*
     * Back-propagate errors to compute new values for input_error.
     *
     * The output error (that is, "output deltas") must have already been updated before
     * calling this method. Note that a reference to the output deltas can be obtained by
     * calling get_output_deltas(). Otherwise, the error gradients will not be back-propagated
     * correctly.
     */
    virtual void back_propagate_deltas(MatrixF& input_error);

    /*
     * Set the mode of this layer to either train or test/evaluate.
     *
     * Some layers, such as dropout layers, behave differently between training
     * and evaluation modes. Most other sub-layers can ignore this mode, however.
     *
     * The default value is false (that is, use evaluation mode be default).
     */
    virtual void set_train_mode(bool is_train);

    /*
     * Return a reference to this layer's output activations, which consists of a reference
     * to the outputs of the last layer in the sequence of layers.
     *
     * Note: The output is undefined until layers have been added to the network and
     * forward_propagate() has been called.
     */
    virtual const MatrixF& get_output() const {
      if (!m_is_initialized) {
        std::cerr << m_layer_name <<  ": get_output() called before being initialized." << std::endl;
        exit(1);
      }
      return m_layers[m_layers.size() - 1].get().get_output();
    }

    /*
     * Return a reference to this layer's output activations, which consists of a reference
     * to the outputs of the last layer in the sequence of layers.
     *
     * Note: The output is undefined until layers have been added to the network and
     * forward_propagate() has been called.
     */
    virtual MatrixF& get_output() {
      if (!m_is_initialized) {
        std::cerr << m_layer_name <<  ": get_output() called before being initialized." << std::endl;
        exit(1);
      }
      return m_layers[m_layers.size() - 1].get().get_output();
    }

    /*
     * Return a reference to this layer's output deltas. These activations represent the
     * gradient of the output activations (that is, errors for the output activations)
     * that are computed during the back-propagation step.
     *
     * Note: The output deltas is undefined until layers have been added to the network and
     * forward_propagate() has been called.
     */
    virtual const MatrixF& get_output_deltas() const {
      if (!m_is_initialized) {
        std::cerr << m_layer_name <<  ": get_output_deltas() called before being initialized." << std::endl;
        exit(1);
      }
      return m_layers[m_layers.size() - 1].get().get_output_deltas();
    }

    /*
     * Return a reference to this layer's output deltas. These activations represent the
     * gradient of the output activations (that is, errors for the output activations)
     * that are computed during the back-propagation step.
     *
     * Note: The output deltas is undefined until layers have been added to the network and
     * forward_propagate() has been called.
     */
    virtual MatrixF& get_output_deltas() {
      if (!m_is_initialized) {
        std::cerr << m_layer_name <<  ": get_output_deltas() called before being initialized." << std::endl;
        exit(1);
      }
      return m_layers[m_layers.size() - 1].get().get_output_deltas();
    }

    /*
     * Return the extents of the output activations.
     * This information is typically passed to the constructor of the next layer in the network.
     *
     * Note: The output extents is undefined until layers have been added to the network and
     * forward_propagate() has been called.
     */
    virtual std::vector<int> get_output_extents() const {
      if (!m_is_initialized) {
        std::cerr << m_layer_name <<  ": get_output_extents() called before being initialized." << std::endl;
        exit(1);
      }
      return m_layers[m_layers.size() - 1].get().get_output_extents();
    }

    /*
     * Print information for weights and bias for each sub-layer in this network.
     */
    virtual void print_paramater_stats() const;


    /*
     * Save parameters to a file withe the prefix given
     * by the supplied name.
     *
     * Note: Before this function can be called, the network must first be initialized by calling "forward()."
     * Otherwise, this function will exit with an error.
     *
     * Note that since the wieght and bias paramters for this container layer represent the
     * corresponding parameters of all contained layers, this function takes care of saving
     * the parameters of all contained layers.
     *
     */
    virtual void save_parameters(std::string name) const;

    /*
     * Load learned parameters from a file, copying them into all layers contained
     * by this layer. The string name should be
     * the same that was used to save the parameters when
     * save_learning_info() was called.
     *
     * Note: Before this function can be called, the network must first be initialized by calling "forward()."
     * Otherwise, this function will exit with an error.
     *
     * This implementation first loads the supplied file into the weights and bias parameter matrices of
     * this container layer. Then, the values in the weight and bias matrices are copied into the corresponding
     * weight and bias matrices of any contained layers. That is, this function takes care of loading the
     * parameters into all contained layers.
     */
    virtual void load_parameters(std::string name);


  protected:

    /*
     * Compute the output activations as a function of input activations.
     *
     * The output activations can then be obtained by calling get_output().
     *
     * Implementation note:
     * This function will first check to see if the current input extents match the
     * supplied matrix. If they differ, various internal state matrices may be reinitialized
     * to be consistent with the extents of the supplied input activations. The parameter
     * matrices, if any, will only be reinitialized if they are required to change size.
     * Note that any such initialization will typically only occur once since the extents
     * of the input activations typically do not change at runtime. Any sub classes are advised
     * to implement this behavior.
     */
    virtual void forward_propagate(const MatrixF& input_activations);

    /*
     * This does nothing.
     */
    virtual void reinitialize(std::vector<int> input_extents);

  private:

    /*
     * Copy weights from contained Layers into the m_W matrix of this container Layer.
     */
    void copy_weights_contained_layers_to_this();

    /*
     * This function cause the weights matrix for this composite layer to get copied
     * into the corresponding matrices of the contained layers.
     */
    void copy_weights_this_to_contained_layers();

    /*
     * This function cause the weights gradient matrices for this contained layers to get copied
     * into the corresponding matrix of this composite layer.
     */
    void copy_weights_gradients_contained_layers_to_this();

    /*
     * Copy bias from contained Layers into the m_bias matrix of this container Layer.
     */
    void copy_bias_contained_layers_to_this();

    /*
     * This function cause the m_bias matrix for this composite layer to get copied
     * into the corresponding matrices of the contained layers.
     */
    void copy_bias_this_to_contained_layers();

    /*
     * This function cause the bias gradient matrices for this contained layers to get copied
     * into the corresponding matrix of this composite layer.
     */
    void copy_bias_gradients_contained_layers_to_this();

    // Holds a reference to each layer in the network in the order in which they were added.
    std::vector<std::reference_wrapper<Layer>> m_layers;

  };

}


#endif /* _SEQUENTIALNETWORK_H */
