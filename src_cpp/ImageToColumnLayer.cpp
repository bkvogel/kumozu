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

  void ImageToColumnLayer::multi_dim_minibatch_to_column_minibatch(MatrixF& A, const MatrixF&B) {
    bool bad_parameters = false;
    if (A.size() != B.size()) {
      bad_parameters = true;
    }
    const int minibatch_size = A.extent(1);
    if (minibatch_size != B.extent(0)) {
      bad_parameters = true;
    }
    if (bad_parameters) {
      cerr << "multi_dim_minibatch_to_column_minibatch(): bad parameters." << endl;
      exit(1);
    }
    // Number of dimensions in B that get flattened into a column of data in A.
    const int combine_dims = B.order() -1;
    if (combine_dims == 2) {
#pragma omp parallel for
      for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
        int col = 0;
        for (int i = 0; i < B.extent(1); ++i) {
          for (int j = 0; j < B.extent(2); ++j) {
            A(col, minibatch_index) = B(minibatch_index, i, j);
            ++col;
          }
        }
      }
    } else if (combine_dims == 3) {
#pragma omp parallel for
      for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
        int col = 0;
        for (int i = 0; i < B.extent(1); ++i) {
          for (int j = 0; j < B.extent(2); ++j) {
            for (int k = 0; k < B.extent(3); ++k) {
              A(col, minibatch_index) = B(minibatch_index, i, j, k);
              ++col;
            }
          }
        }
      }
    } else {
      cerr << "multi_dim_minibatch_to_column_minibatch(): This size is not supported yet. Sorry." << endl;
      exit(1);
    }
  }


  void ImageToColumnLayer::column_minibatch_to_multi_dim_minibatch(const MatrixF& A, MatrixF&B) {
    bool bad_parameters = false;
    if (A.size() != B.size()) {
      bad_parameters = true;
    }
    const int minibatch_size = A.extent(1);
    if (minibatch_size != B.extent(0)) {
      bad_parameters = true;
    }
    if (bad_parameters) {
      cerr << "multi_dim_minibatch_to_column_minibatch(): bad parameters." << endl;
      exit(1);
    }
    // Number of dimensions in B that get flattened into a column of data in A.
    const int combine_dims = B.order() -1;
    if (combine_dims == 2) {
#pragma omp parallel for
      for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
        int col = 0;
        for (int i = 0; i < B.extent(1); ++i) {
          for (int j = 0; j < B.extent(2); ++j) {
            B(minibatch_index, i, j) = A(col, minibatch_index);
            ++col;
          }
        }
      }
    } else if (combine_dims == 3) {
#pragma omp parallel for
      for (int minibatch_index = 0; minibatch_index < minibatch_size; ++ minibatch_index) {
        int col = 0;
        for (int i = 0; i < B.extent(1); ++i) {
          for (int j = 0; j < B.extent(2); ++j) {
            for (int k = 0; k < B.extent(3); ++k) {
              B(minibatch_index, i, j, k) = A(col, minibatch_index);
              ++col;
            }
          }
        }
      }
    } else {
      cerr << "multi_dim_minibatch_to_column_minibatch(): This size is not supported yet. Sorry." << endl;
      exit(1);
    }
  }



  void ImageToColumnLayer::reinitialize(std::vector<int> input_extents) {
    if (input_extents.size() < 3) {
      cerr << "ImageToColumnLayer: Input extents has too few dimensions. Exiting." << endl;
      exit(1);
    }

    m_minibatch_size = input_extents.at(0);
    int column_dim = 1;
    for (int i=1; i < static_cast<int>(input_extents.size()); ++i) {
      column_dim *= input_extents.at(i);
    }
    m_output_activations = MatrixF(column_dim, m_minibatch_size);
    m_output_error = MatrixF(column_dim, m_minibatch_size);
  }


  void ImageToColumnLayer::forward_propagate(const MatrixF& input_activations) {
    multi_dim_minibatch_to_column_minibatch(m_output_activations, input_activations);
  }

  void ImageToColumnLayer::back_propagate_deltas(MatrixF& input_error) {
    column_minibatch_to_multi_dim_minibatch(m_output_error, input_error);
  }


}
