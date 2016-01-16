#ifndef _CROSSENTROPYCOSTFUNCTION_H
#define _CROSSENTROPYCOSTFUNCTION_H
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
#include "CostFunction.h"

namespace kumozu {

  /*
   * This is a cross entropy cost function in a neural network.
   *
   * It is assumed that the layer before this cost function is in instance of LinearLayer, so that
   * the activaions input to this cost function have dimensions unit_count x minibatch_size.
   *
   * Usage:
   *
   * To compute the cost in the forward direction, call
   * forward_propagate().
   *
   * Optional: If the actual cost value is required, then call get_cost() to get the value.
   *
   * To compute the gradients with respect to the supplied activations (i.e., delta activations), call
   * back_propagate().
   *
   */
  class CrossEntropyCostFunction : public CostFunction {

  public:

  CrossEntropyCostFunction(std::string name) :
    CostFunction(name)
    {

    }




    virtual float forward_propagate(const MatrixF& input_activations, const MatrixF& target_activations) {
      std::cerr << "Wrong function." << std::endl;
      exit(1);
    }

    /*
     * Compute and return the cost function value using the supplied input activations and target activations, which
     * typically corresponds to one mini-batch of data.
     *
     * The input activations correspond to the "output" of the network which is connected to the
     * "input" of this class. Thus, the supplied "input_activations" and "target"activations"
     * must have the same dimensions. The cost function output (one scalar value per example) will
     * be stored in the output activations, which can then be obtained by calling get_output().
     *
     * input_activations: A 2-D matrix of dimension class_count x minibatch_size.
     *
     * target_activations: A 1-D matrix of size minibatch_size. target_activations(i) contains the integer-valued
     *                     class label in the range [0, class_count-1].
     */
    virtual float forward_propagate(const MatrixF& input_activations, const Matrix<int>& target_activations);


    virtual void back_propagate(MatrixF& input_error, const MatrixF& input_activations,
                                const MatrixF& target_activations) {
      std::cerr << "Wrong function." << std::endl;
      exit(1);
    }

    /*
     * Compute the error gradients for the input layer, which correspond to the error gradients for
     * the output layer of the network that is connected to this class.
     *
     */
    virtual void back_propagate(MatrixF& input_error, const MatrixF& input_activations,
                                const Matrix<int>& target_activations);

    /*
     * Check that the gradients computed using the finite differences method agrees with
     * the gradients computed using the gradient back-propagation member function.
     *
     */
    virtual void check_gradients(std::vector<int> input_extents);


  protected:
    /*
     * Set the extents of the input activations.
     */
    virtual void reinitialize(std::vector<int> input_extents);


  private:

    int m_minibatch_size;
    MatrixF m_exp_input;
    MatrixF m_mu;
    MatrixF m_col_sums;
    MatrixF m_temp_input_error;
    //std::vector<int> m_input_extents;
    //float m_cost = 0;// fixme: remove

  };

}


#endif /* _CROSSENTROPYCOSTFUNCTION_H */
