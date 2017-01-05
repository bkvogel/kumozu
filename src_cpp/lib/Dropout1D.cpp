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
#include "Dropout1D.h"

#include "Utilities.h"
using namespace std;

namespace kumozu {

  void Dropout1D::reinitialize(std::vector<int> input_extents) {
    // Note: input_extents.at(1) is mini-batch size.
    // input_extents.at(0) is dim_input.
    m_output_data.resize(input_extents);
    m_output_grad.resize(input_extents);
    if (m_mode == 0) {
      m_dropout_mask.resize(input_extents.at(0));
    } else if (m_mode == 1) {
      //cout << "Using dropout maks of size: " << input_extents.size() << endl;
      m_dropout_mask.resize(input_extents);
    }
  }

  void Dropout1D::forward_propagate(const MatrixF& input_activations) {
    float prob_keep_current = 1.0f;
    if (is_train_mode()) {
      prob_keep_current = m_prob_keep;
    }

    check_dimensions(input_activations, m_output_data);
    // Compute a new random dropout mask to use on each column in the mini-batch.
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    // Repeat minibatch_size copies of same mask.
    for (int i = 0; i < m_dropout_mask.size(); i++) {
      float x = uni(m_random_engine);
      if (x < prob_keep_current) {
        // Keep the element
        m_dropout_mask[i] = 1;
      } else {
        // Drop the element
        m_dropout_mask[i] = 0;
      }
    }

    if (m_mode == 0) {
      // Apply the dropout mask.
      const float scale = 1.0f/prob_keep_current;
      int minibatch_size = m_output_data.extent(1);
#pragma omp parallel for collapse(2)
      for (int n = 0; n < m_output_data.extent(0); ++n) {
        for (int m = 0; m < minibatch_size; ++m) {
          if (m_dropout_mask[n] == 1) {
            m_output_data(n, m) = input_activations(n, m)*scale; // inverted dropout
          } else {
            m_output_data(n, m) = 0.0f; // works well
          }
        }

      }

    } else if (m_mode == 1) {
      const float scale = 1.0f/prob_keep_current;
      // Apply the dropout mask.
#pragma omp parallel for
      for (int n = 0; n < m_dropout_mask.size(); ++n) {
        if (m_dropout_mask[n] == 1) {
          m_output_data[n] = input_activations[n]*scale; // inverted dropout
        } else {
          m_output_data[n] = 0.0f;
        }
      }

    }


  }

  void Dropout1D::back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) {
    check_dimensions(input_backward, m_output_data);
    float prob_keep_current = 1.0f;
    if (is_train_mode()) {
      prob_keep_current = m_prob_keep;
    }
    const float scale = 1.0f/prob_keep_current;
    // Apply the dropout mask.
    if (m_mode == 0) {
      int minibatch_size = m_output_data.extent(1);
#pragma omp parallel for collapse(2)
      for (int n = 0; n < m_output_data.extent(0); ++n) {
        for (int m = 0; m < minibatch_size; ++m) {
          if (m_dropout_mask[n] == 1) {
            input_backward(n, m) = m_output_grad(n, m)*scale;
          } else {
            input_backward(n, m) = 0.0f; // works well

          }

        }
      }
    } else if (m_mode == 1) {
#pragma omp parallel for
      for (int n = 0; n < m_dropout_mask.size(); ++n) {
        if (m_dropout_mask[n] == 1) {
          input_backward[n] = m_output_grad[n]*scale; // inverted dropout
        } else {
          input_backward[n] = 0.0f;
        }
      }
    }

  }



}
