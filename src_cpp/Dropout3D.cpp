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
#include "Dropout3D.h"

#include "Utilities.h"
using namespace std;

namespace kumozu {

  void Dropout3D::reinitialize(std::vector<int> input_extents) {
    m_minibatch_size = input_extents.at(0);
    m_depth =  input_extents.at(1);
    m_height = input_extents.at(2);
    m_width = input_extents.at(3);
    m_output_activations.resize(m_minibatch_size, m_depth, m_height, m_width);
    m_output_error.resize(m_minibatch_size, m_depth, m_height, m_width);

    if (m_mode == 0) {
      m_dropout_mask.resize(m_depth, m_height, m_width);
    } else if (m_mode == 1) {
      m_dropout_mask.resize(input_extents);
    }
  }

  void Dropout3D::forward_propagate(const MatrixF& input_activations) {
    //reinitialize(input_activations.get_extents());
    float prob_keep_current = 1.0f;
    if (m_is_train) {
      prob_keep_current = m_prob_keep;
    }
    check_dimensions(input_activations, m_output_activations);
    // Compute a new random dropout mask to use on each column in the mini-batch.
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    for (int i = 0; i < m_dropout_mask.size(); i++) {
      const float x = uni(m_random_engine);
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
#pragma omp parallel for collapse(3)
      for (int m = 0; m < m_minibatch_size; ++m) {
        for (int d = 0; d < m_depth; ++d) {
          for (int h = 0; h < m_height; ++h) {
            for (int w = 0; w < m_width; ++w) {
              if (m_dropout_mask(d, h, w) == 1) {
                m_output_activations(m, d, h, w) = input_activations(m, d, h, w)/prob_keep_current; // inverted dropout
              } else {
                m_output_activations(m, d, h, w) = 0.0f;
              }

            }
          }
        }
      }
    } else if (m_mode == 1) {
      const float scale = 1.0f/prob_keep_current;
#pragma omp parallel for
      for (int n = 0; n < m_dropout_mask.size(); ++n) {
        if (m_dropout_mask[n] == 1) {
          //m_output_activations[n] = input_activations[n]/prob_keep_current; // inverted dropout
          m_output_activations[n] = input_activations[n]*scale; // inverted dropout
        } else {
          m_output_activations[n] = 0.0f;
        }
      }
    }
  }

  void Dropout3D::back_propagate_deltas(MatrixF& input_error) {
    float prob_keep_current = 1.0f;
    if (m_is_train) {
      prob_keep_current = m_prob_keep;
    }
    if (m_mode == 0) {
#pragma omp parallel for collapse(3)
      for (int m = 0; m < m_minibatch_size; ++m) {
        for (int d = 0; d < m_depth; ++d) {
          for (int h = 0; h < m_height; ++h) {
            for (int w = 0; w < m_width; ++w) {
              if (m_dropout_mask(d, h, w) == 1) {
                input_error(m, d, h, w) = m_output_error(m, d, h, w)/prob_keep_current;
              } else {
                input_error(m, d, h, w) = 0.0f;
              }

            }
          }
        }
      }
    } else if (m_mode == 1) {
      const float scale = 1.0f/prob_keep_current;
#pragma omp parallel for
      for (int n = 0; n < m_dropout_mask.size(); ++n) {
        if (m_dropout_mask[n] == 1) {
          input_error[n] = m_output_error[n]*scale; // inverted dropout
        } else {
          input_error[n] = 0.0f;
        }
      }
    }
  }

}
