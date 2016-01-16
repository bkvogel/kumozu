#ifndef _COSTFUNCTION_H
#define _COSTFUNCTION_H
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
   * This is an abstract base class for a cost function in a neural network.
   *
   * A cost function is typically used during the training process to compare the output
   * activations of a network to the corresponding "true" training output values. The degree to
   * which the outputs of the network differ from the true output values will be represented by
   * the "cost" or "loss" value computed by a sub-class of this class.
   *
   * This class is intended to model a cost function that has the following characteristics:
   *
   * - A notion of "input" activations. The input typically corresponds to one mini-batch of data. We use the convention
   *   that the input activations are not considered an internal state of a Layer. That is, the input activations
   *   can be thought of as existing in some toher "external" object, such as the output activations of another
   *   Layer instance, for example.
   * - A cost function value that is computed when forward_propagate() is called. Since this class operates on a
   *   mini-batch of data at a time, the cost function values are returned in output_activations, which contains one
   *   scalar cost function value for each batch index.
   *
   * - Gradients of the "input" activations with respect to the cost function can be computed by calling a
   *   method of this class. This corresponds to the "back_propagate()" function.
   *
   * A function is also provided to check that the numerically-computed Jacobian roughly matches the gradients/Jacobian
   * computed by the member functions.
   *
   * Usage:
   *
   * A sub-class should implement a specific cost function by overriding the member functions
   *
   * forward_propagate() and
   * back_propagate().
   *
   * and also be sure to add a unit test that calls check_gradients() to verify that the implementation is correct.
   *
   * Notes:
   *
   * A CostFunction sub-class is typically placed after a LinearLayer instance in a network. Note that the activations
   * in a LinearLayer correspond to a mini-batch of data represented by a 2D matrix of dimension layer_units x minibatch_size.
   * This class therefore assumes that the second dimension (number of columns) is the mini-batch size.
   */
  class CostFunction {

  public:

  CostFunction(std::string name) : m_layer_name {name},
      m_epsilon {1e-4f},
        m_pass_relative_error {1e-2f}, // Minimum relative error needed to pass Jacobian checker.
          m_is_initialized {false}
          {

          }

          virtual ~CostFunction(){}

          /*
           * Compute the cost function using the supplied input activations and target activations, which
           * typically corresponds to one mini-batch of data.
           *
           * The input activations correspond to the "output" of the network which is connected to the
           * "input" of this class. Thus, the supplied "input_activations" and "target"activations"
           * must have the same dimensions. The cost function output (one scalar value per example) will
           * be stored in the output activations, which can then be obtained by calling get_output().
           *
           */
          //virtual float forward_propagate(const MatrixF& input_activations, const MatrixF& target_activations) = 0;
          virtual float forward_propagate(const MatrixF& input_activations, const MatrixF& target_activations) {
            return 0.0f;
          }

          virtual float forward_propagate(const MatrixF& input_activations, const Matrix<int>& target_activations) {
            return 0.0f;
          }


          float forward(const MatrixF& input_activations, const MatrixF& target_activations) {
            if (input_activations.get_extents() != m_input_extents) {
              std::cout << std::endl << "Initializing " << m_layer_name << ":" << std::endl;
              m_input_extents = input_activations.get_extents();
              reinitialize(input_activations.get_extents());
            }
            float val = forward_propagate(input_activations, target_activations);
            m_is_initialized = true;
            return val;
          }

          float forward(const MatrixF& input_activations, const Matrix<int>& target_activations) {
            if (input_activations.get_extents() != m_input_extents) {
              std::cout << std::endl << "Initializing " << m_layer_name << ":" << std::endl;
              m_input_extents = input_activations.get_extents();
              reinitialize(input_activations.get_extents());
            }
            float val = forward_propagate(input_activations, target_activations);
            m_is_initialized = true;
            return val;
          }


          /*
           * Compute the error gradients for the input layer, which correspond to the error gradients for
           * the output layer of the network that is connected to this class.
           *
           */
          virtual void back_propagate(MatrixF& input_error, const MatrixF& input_activations,
                                      const MatrixF& target_activations) = 0;



          /*
           * Check that the gradients computed using the finite differences method agrees with
           * the gradients computed using the gradient back-propagation member function.
           *
           */
          virtual void check_gradients(std::vector<int> input_extents);

          std::string get_name() const {
            return m_layer_name;
          }

          /*
           * Return the cost for the current mini-batch.
           *
           * Be sure to call forward_propagate() before calling this function.
           */
          //virtual float get_cost() = 0;


  protected:
          /*
           * Set the extents of the input activations.
           */
          virtual void reinitialize(std::vector<int> input_extents) = 0;

          std::string m_layer_name;
          MatrixF m_output_activations;
          std::vector<int> m_input_extents;

          const float m_epsilon;
          const float m_pass_relative_error;

  private:

          bool m_is_initialized;

  };

}


#endif /* _COSTFUNCTION_H */
