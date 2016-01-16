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

#include "Layer.h"
#include "Utilities.h"
#include "MatrixIO.h"

using namespace std;

namespace kumozu {

  void Layer::check_jacobian_weights(std::vector<int> input_extents) {
    cout << get_name() << ": Checking Jacobian for weights..." << endl;
    //reinitialize(input_extents);
    // Create random input activations for the layer.
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    MatrixF input_deltas(input_extents);
    randomize_uniform(input_deltas, 0.0f, 1.0f);
    forward(input_activations);
    // Now check Jacobian for weights:
    // Size will be total_output_dim x total_weights_dim =
    // (dim_output*minibatch_size) x total_weights_dim
    const int total_output_dim = get_output().size();
    const int total_weights_dim = get_weights().size();
    // This will contain the Jacobian computed using finite differences method.
    MatrixF numerical_jacobian_weights(total_output_dim, total_weights_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(numerical_jacobian_weights, 0.0f, 1.0f);

    // This will contain the Jacobian computed using the back-prop method of the class
    MatrixF backprop_jacobian_weights(total_output_dim, total_weights_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(backprop_jacobian_weights, 0.0f, 1.0f);

    // Now compute the numerical Jacobian:
    // This will be computed one column at a time.
    MatrixF& W = get_weights();
    MatrixF& output_activations = get_output();
    for (int j=0; j < W.size(); ++j) {
      // j is column index int Jacobian matrix.
      float orig = W[j];
      W[j] += m_epsilon;
      // Now compute output of layer -> output_activations
      forward(input_activations);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_activations.size(); ++i) {
        numerical_jacobian_weights(i,j) = output_activations[i];
      }
      W[j] = orig - m_epsilon;
      // Now compute output of layer -> output_activations
      forward(input_activations);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_activations.size(); ++i) {
        numerical_jacobian_weights(i,j) -= output_activations[i];
        numerical_jacobian_weights(i,j) /= 2*m_epsilon;
      }
      // Put back original value.
      W[j] = orig;
    }
    // Now compute the Jacobian using the backprop function.
    MatrixF& output_errors = get_output_deltas();
    MatrixF& grad_W = get_weight_gradient();
    set_value(output_errors, 0.0f);
    for (int i=0; i < output_errors.size(); ++i) {
      output_errors[i] = 1.0f;
      // Now if we perform backprop, the result should be the same is row i of Jacobian.
      //back_propagate_deltas(input_deltas);
      //back_propagate_paramater_gradients(input_activations);
      back_propagate(input_deltas, input_activations);
      for (int j=0; j < W.size(); ++j) {
        backprop_jacobian_weights(i,j) = grad_W[j];
      }
      output_errors[i] = 0.0f;
    }

    const float relative_error_score = relative_error(numerical_jacobian_weights, backprop_jacobian_weights);
    std::cout << "numerical-back-prop gradients relative error = " << relative_error_score << std::endl;
    assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);
    cout << "PASSED" << endl;
  }

  void Layer::check_jacobian_bias(std::vector<int> input_extents) {
    cout << get_name() << ": Checking Jacobian for bias..." << endl;
    //reinitialize(input_extents);
    // Create random input activations for the layer.
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    MatrixF input_deltas(input_extents);
    randomize_uniform(input_deltas, 0.0f, 1.0f);
    forward(input_activations);
    // Now check Jacobian for bias vectors:
    // Size will be total_output_dim x total_bias_dim =
    // (dim_output*minibatch_size) x total_bias_dim
    const int total_output_dim = get_output().size();
    const int total_bias_dim = get_bias().size();
    // This will contain the Jacobian computed using finite differences method.
    MatrixF numerical_jacobian_bias(total_output_dim, total_bias_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(numerical_jacobian_bias, 0.0f, 1.0f);

    // This will contain the Jacobian computed using the back-prop method of the class
    MatrixF backprop_jacobian_bias(total_output_dim, total_bias_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(backprop_jacobian_bias, 0.0f, 1.0f);

    // Now compute the numerical Jacobian:
    // This will be computed one column at a time.
    MatrixF& bias = get_bias();
    MatrixF& output_activations = get_output();
    for (int j=0; j < bias.size(); ++j) {
      // j is column index int Jacobian matrix.
      float orig = bias[j];
      bias[j] += m_epsilon;
      // Now compute output of layer -> output_activations
      forward(input_activations);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_activations.size(); ++i) {
        numerical_jacobian_bias(i,j) = output_activations[i];
      }
      bias[j] = orig - m_epsilon;
      // Now compute output of layer -> output_activations
      forward(input_activations);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_activations.size(); ++i) {
        numerical_jacobian_bias(i,j) -= output_activations[i];
        numerical_jacobian_bias(i,j) /= 2*m_epsilon;
      }
      // Put back original value.
      bias[j] = orig;
    }
    // Now compute the Jacobian using the backprop function.
    MatrixF& output_errors = get_output_deltas();
    MatrixF& grad_bias = get_bias_gradient();
    set_value(output_errors, 0.0f);
    for (int i=0; i < output_errors.size(); ++i) {
      output_errors[i] = 1.0f;
      // Now if we perform backprop, the result should be the same is row i of Jacobian.
      //back_propagate_deltas(input_deltas);
      //back_propagate_paramater_gradients(input_activations);
      back_propagate(input_deltas, input_activations);
      for (int j=0; j < bias.size(); ++j) {
        backprop_jacobian_bias(i,j) = grad_bias[j];
      }
      output_errors[i] = 0.0f;
    }

    const float relative_error_score = relative_error(numerical_jacobian_bias, backprop_jacobian_bias);
    std::cout << "numerical-back-prop gradients relative error = " << relative_error_score << std::endl;
    assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);
    cout << "PASSED" << endl;
  }

  void Layer::check_jacobian_input_error(std::vector<int> input_extents) {
    cout << get_name() << ": Checking Jacobian for input error gradients..." << endl;
    //reinitialize(input_extents);
    // Create random input activations for the layer.
    MatrixF input_activations(input_extents);
    randomize_uniform(input_activations, 0.0f, 1.0f);
    MatrixF input_deltas(input_extents);
    randomize_uniform(input_deltas, 0.0f, 1.0f);
    forward(input_activations); // Initialize network.
    // Size will be total_output_dim x total_input_dim =
    // (dim_output*minibatch_size) x total_input_dim
    const int total_output_dim = get_output().size();
    const int total_input_dim = input_activations.size();
    // This will contain the Jacobian computed using finite differences method.
    MatrixF numerical_jacobian_input(total_output_dim, total_input_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(numerical_jacobian_input, 0.0f, 1.0f);

    // This will contain the Jacobian computed using the back-prop method of the class
    MatrixF backprop_jacobian_input(total_output_dim, total_input_dim);
    // Randomize to make accidental matches less likely.
    randomize_uniform(backprop_jacobian_input, 0.0f, 1.0f);

    // Now compute the numerical Jacobian:
    // This will be computed one column at a time.
    MatrixF& output_activations = get_output();
    for (int j=0; j < input_activations.size(); ++j) {
      // j is column index int Jacobian matrix.
      float orig = input_activations[j];
      input_activations[j] += m_epsilon;
      // Now compute output of layer -> output_activations
      forward(input_activations);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_activations.size(); ++i) {
        numerical_jacobian_input(i,j) = output_activations[i];
      }
      input_activations[j] = orig - m_epsilon;
      // Now compute output of layer -> output_activations
      forward(input_activations);
      // Copy the output into column j of Jacobian.
      for (int i=0; i < output_activations.size(); ++i) {
        numerical_jacobian_input(i,j) -= output_activations[i];
        numerical_jacobian_input(i,j) /= 2*m_epsilon;
      }
      // Put back original value.
      input_activations[j] = orig;
    }
    // Now compute the Jacobian using the backprop function.
    MatrixF& output_errors = get_output_deltas();

    set_value(output_errors, 0.0f);
    for (int i=0; i < output_errors.size(); ++i) {
      output_errors[i] = 1.0f;
      // Now if we perform backprop, the result should be the same is row i of Jacobian.
      //back_propagate_deltas(input_deltas);
      back_propagate(input_deltas, input_activations);
      for (int j=0; j < input_activations.size(); ++j) {
        backprop_jacobian_input(i,j) = input_deltas[j];
      }
      output_errors[i] = 0.0f;
    }

    const float relative_error_score = relative_error(numerical_jacobian_input, backprop_jacobian_input);
    std::cout << "check_jacobian_input_error(): relative error = " << relative_error_score << std::endl;
    //cout << "numerical_jacobian_input = " << endl << numerical_jacobian_input << endl;
    //cout << "backprop_jacobian_input = " << endl << backprop_jacobian_input << endl;
    assert_almost_equal(relative_error_score, 0.0f, m_pass_relative_error);
    cout << "PASSED" << endl;
  }

  void Layer::print_paramater_stats() const {
    if (!m_is_initialized) {
      std::cerr << m_layer_name <<  ": print_paramater_stats() called before being initialized." << std::endl;
      exit(1);
    }
    if (get_weights().size() > 0) {
      print_stats(get_weights(), m_layer_name + " : weights");
    }
    if (get_bias().size() > 0) {
      print_stats(get_bias(), m_layer_name + " : bias");
    }
    if (get_weight_gradient().size() > 0) {
      print_stats(get_weight_gradient(), m_layer_name + " : weight gradients");
    }
    if (get_bias_gradient().size() > 0) {
      print_stats(get_bias_gradient(), m_layer_name + " : bias gradients");
    }
    if(get_output().size() > 0) {
      print_stats(get_output(), m_layer_name + " : output activations");
    }
    if (get_output_deltas().size() > 0) {
      print_stats(get_output_deltas(), m_layer_name + " : output activations deltas");
    }
  }

  void Layer::save_parameters(std::string name) const {
    if (!m_is_initialized) {
      std::cerr << m_layer_name <<  ": save_parameters() called before being initialized." << std::endl;
      exit(1);
    }
    save_matrix(m_W, name + "_" + m_layer_name + "_W.dat");
    save_matrix(m_bias, name + "_" + m_layer_name + "_bias.dat");
  }

  void Layer::load_parameters(std::string name) {
    m_W = load_matrix(name + "_" + m_layer_name + "_W.dat");
    m_bias = load_matrix(name + "_" + m_layer_name + "_bias.dat");
  }

}
