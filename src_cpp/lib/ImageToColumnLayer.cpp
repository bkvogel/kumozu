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
#include "ImageToColumnLayer.h"
#include "Utilities.h"

using namespace std;

namespace kumozu {




  void ImageToColumnLayer::reinitialize(std::vector<int> input_extents) {
    if (input_extents.size() < 3) {
        error_exit("ImageToColumnLayer: Input extents has too few dimensions. Exiting.");
    }

    m_minibatch_size = input_extents.at(0);
    int column_dim = 1;
    for (int i=1; i < static_cast<int>(input_extents.size()); ++i) {
      column_dim *= input_extents.at(i);
    }
    m_output_data.resize(column_dim, m_minibatch_size);
    m_output_grad.resize(column_dim, m_minibatch_size);
  }


  void ImageToColumnLayer::forward_propagate(const MatrixF& input_activations) {
    multi_dim_minibatch_to_column_minibatch(m_output_data, input_activations);
  }

  void ImageToColumnLayer::back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) {
    column_minibatch_to_multi_dim_minibatch(m_output_grad, input_backward);
  }


}
