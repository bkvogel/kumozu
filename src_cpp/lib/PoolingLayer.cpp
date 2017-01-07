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
#include "PoolingLayer.h"

#include "Utilities.h"
using namespace std;

namespace kumozu {

  void PoolingLayer::reinitialize(std::vector<int> input_extents) {
    string indent = "    ";
    const int minibatch_size = input_extents.at(0);
    const int dim0_in = input_extents.at(1);
    const int dim1_in = input_extents.at(2);
    const int dim2_in = input_extents.at(3);
    const int step_0 = m_pooling_region_step_sizes.at(0);
    const int step_1 = m_pooling_region_step_sizes.at(1);
    const int step_2 = m_pooling_region_step_sizes.at(2);

    // Use this initialization if we want to require that every pooling region must fit completely
    // inside the input activations matrix.
    //const int out_dim_0 = (dim0_in - region_size_0)/step_0 + 1;
    //const int out_dim_1 = (dim1_in - region_size_1)/step_1 + 1;
    //const int out_dim_2 = (dim2_in - region_size_2)/step_2 + 1;

    // Use this initialization if we want to allow pooling regions to partially slip outside of
    // the input activations matrix, as long as the corner element still lies inside.
    const int out_dim_0 = static_cast<int>(ceil(static_cast<float>(dim0_in)/static_cast<float>(step_0)));
    const int out_dim_1 = static_cast<int>(ceil(static_cast<float>(dim1_in)/static_cast<float>(step_1)));
    const int out_dim_2 = static_cast<int>(ceil(static_cast<float>(dim2_in)/static_cast<float>(step_2)));

    m_output_var.resize(minibatch_size, out_dim_0, out_dim_1, out_dim_2);
    m_state.resize(minibatch_size, out_dim_0, out_dim_1, out_dim_2, 3);


    if (input_extents.size() != 4) {
      std::cerr << get_name() << ": Input extents have wrong size. Exiting." << std::endl;
      exit(1);
    }
    std::cout << indent << "Pooling input extents (minibatch_size, depth_in, height_in, width_in):" << std::endl << indent;
    for (auto i : input_extents) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << indent << "Pooling region extents:" << std::endl << indent;
    for (auto i : m_pooling_region_extents) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

    std::cout << indent << "Pooling output extents (minibatch_size, depth_out, height_out, width_out):" << std::endl << indent;
    for (auto i : m_output_var.get_extents()) {
      std::cout << i << " ";
    }
    std::cout << std::endl;

  }

  void PoolingLayer::forward_propagate(const MatrixF& input_activations) {
    forward_maxout_3d(input_activations, m_output_var.data, m_state,
                      m_pooling_region_extents, m_pooling_region_step_sizes);
  }

  void PoolingLayer::back_propagate_activation_gradients(MatrixF& input_backward, const MatrixF& input_forward) {
    reverse_maxout_3d(input_backward, m_output_var.grad, m_state,
                      m_pooling_region_extents, m_pooling_region_step_sizes);
  }


  void PoolingLayer::forward_maxout_3d(const MatrixF& input, MatrixF& output, Matrix<int>& state,
                                       const std::vector<int>& pooling_region_extents, const std::vector<int>& pooling_region_step_sizes) {
    const int minibatch_size = input.extent(0);
    const int dim0_in = input.extent(1);
    const int dim1_in = input.extent(2);
    const int dim2_in = input.extent(3);
    const int step_0 = pooling_region_step_sizes.at(0);
    const int step_1 = pooling_region_step_sizes.at(1);
    const int step_2 = pooling_region_step_sizes.at(2);
    const int region_size_0 = pooling_region_extents.at(0);
    const int region_size_1 = pooling_region_extents.at(1);
    const int region_size_2 = pooling_region_extents.at(2);

#pragma omp parallel for collapse(4)
    for (int b = 0; b < minibatch_size; ++b) {
      for (int i = 0; i < dim0_in; i += step_0) {
        for (int j = 0; j < dim1_in; j += step_1) {
          for (int k = 0; k < dim2_in; k += step_2) {
            // (b,i,j,k) can be work item on GPU.
            // Iterate over the cube inside "input" to find the max value:
            int max_ind_0 = i;
            int max_ind_1 = j;
            int max_ind_2 = k;
            float max_val = input(b, max_ind_0, max_ind_1, max_ind_2);
            for (int l = 0; l < region_size_0; ++l) {
              for (int m = 0; m < region_size_1; ++m) {
                for (int n = 0; n < region_size_2; ++n) {
                  if (input.is_valid_range(b, i+l, j+m, k+n)) {
                    if (input(b, i+l, j+m, k+n) > max_val) {
                      max_val = input(b, i+l, j+m, k+n);
                      max_ind_0 = i+l;
                      max_ind_1 = j+m;
                      max_ind_2 = k+n;
                    }
                  }
                }
              }
            }
            int box_i = i/step_0;
            int box_j = j/step_1;
            int box_k = k/step_2;
            output(b, box_i, box_j, box_k) = max_val;
            state(b,box_i,box_j,box_k,0) = max_ind_0;
            state(b,box_i,box_j,box_k,1) = max_ind_1;
            state(b,box_i,box_j,box_k,2) = max_ind_2;
          }
        }
      }
    }


  }

  void PoolingLayer::reverse_maxout_3d(MatrixF& input_backward, const MatrixF& output_backward, Matrix<int>& state,
                                       const std::vector<int>& pooling_region_extents, const std::vector<int>& pooling_region_step_sizes) {
    const int minibatch_size = input_backward.extent(0);
    const int dim_out_0 = output_backward.extent(1);
    const int dim_out_1 = output_backward.extent(2);
    const int dim_out_2 = output_backward.extent(3);

    set_value(input_backward, 0.0f);

    // Here we split into 2 parrallelizable nested loops. We can continue to split in
    // this way if more harware threads are available.
#pragma omp parallel for collapse(2)
    for (int b = 0; b < minibatch_size; ++b) {
      for (int i = 0; i < dim_out_0; i += 2) {
        for (int j = 0; j < dim_out_1; j += 1) {
          for (int k = 0; k < dim_out_2; k += 1) {
            float val = output_backward(b,i,j,k);
            int max_ind0 = state(b,i,j,k,0);
            int max_ind1 = state(b,i,j,k,1);
            int max_ind2 = state(b,i,j,k,2);
            input_backward(b, max_ind0, max_ind1, max_ind2) += val;
          }
        }
      }
    }

#pragma omp parallel for collapse(2)
    for (int b = 0; b < minibatch_size; ++b) {
      for (int i = 1; i < dim_out_0; i += 2) {
        for (int j = 0; j < dim_out_1; j += 1) {
          for (int k = 0; k < dim_out_2; k += 1) {
            float val = output_backward(b,i,j,k);
            int max_ind0 = state(b,i,j,k,0);
            int max_ind1 = state(b,i,j,k,1);
            int max_ind2 = state(b,i,j,k,2);
            input_backward(b, max_ind0, max_ind1, max_ind2) += val;
          }
        }
      }
    }



  }

}
