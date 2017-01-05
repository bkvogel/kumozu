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

#include "Utilities.h"
#include "CharRNNMinibatchGetter.h"


using namespace std;

namespace kumozu {

  void CharRNNMinibatchGetter::set_char_to_idx_map(std::map<char, int> char_to_idx_map) {
    // Note: char_to_idx_map is passed by value because it is assumed to be small and
    // this function is typically only called once.
    m_char_to_idx = char_to_idx_map;
    std::cout << "char_to_idx" << std::endl;
    for (auto& x : m_char_to_idx) {
      std::cout << "char: " << x.first << " index: " << x.second << std::endl;
      m_idx_to_char[x.second] = x.first;
    }
    const int unique_char_count = m_char_to_idx.size();
    cout << "Number of unique characters in text file: " << unique_char_count << endl;
    cout << "idx_to_char" << endl;
    for (auto& x : m_idx_to_char) {
      cout << "index: " << x.first << " char: " << x.second << endl;
    }
    if (m_input_forward_slices.size() != 0) {
      error_exit("set_char_to_idx_map(): Error: calling more than once is not allowed.");
    }
    // Create the input and output activations matrices for each time slice of the RNN.
    for (int i = 0; i < m_num_slices; ++i) {
      m_input_forward_slices.push_back(std::make_unique<MatrixF>(unique_char_count, m_minibatch_size));
      m_input_backward_slices.push_back(std::make_unique<MatrixF>(unique_char_count, m_minibatch_size));
      m_output_1_hot_slices.push_back(std::make_unique<MatrixF>(unique_char_count, m_minibatch_size));
      m_output_class_index_slices.push_back(std::make_unique<MatrixI>(m_minibatch_size));
    }

    m_text_mat.resize(m_char_count);
    for (int i = 0; i != m_text_mat.size(); ++i) {
      auto it = m_char_to_idx.find(m_text_data[i]);
      if (it == m_char_to_idx.end()) {
	error_exit("set_char_to_idx_map(): Input text contains character that is not in the char-to_idx map.");
      }
      m_text_mat(i) = it->second; // Store index of character
    }


  }


  bool CharRNNMinibatchGetter::next() {
    // zero matrices
    for (int j = 0; j < m_num_slices; ++j) {
      set_value(*m_input_forward_slices[j], 0.0f);
      //set_value(*m_input_backward_slices[j], 0.0f);
      set_value(*m_output_1_hot_slices[j], 0.0f);
      set_value(*m_output_class_index_slices[j], 0);
    }
    for (int i=0; i < m_minibatch_size; ++i) {
      // Check if there are enough characters left in the text string to fill the RNN.
      if (m_minibatch_indices(i) + m_num_slices >= m_char_count) {
        // Not enough characters left, so reset to a valid index.
        //m_minibatch_indices(i) = 0;
	//m_minibatch_indices(i) = random_offset();
	//m_minibatch_indices(i) = m_minibatch_indices_init(i);
	error_exit("CharRNN... oops next()");
      }
      int start_ind = m_minibatch_indices(i);
      narrow(m_example_input, m_text_mat, 0, start_ind, m_num_slices);
      for (int j = 0; j < m_num_slices; ++j) {
	MatrixF& temp_input_forward = *m_input_forward_slices[j];
	temp_input_forward(m_example_input(j), i) = 1;
      }
      
      narrow(m_example_output, m_text_mat, 0, start_ind+1, m_num_slices);
      for (int j = 0; j < m_num_slices; ++j) {
	MatrixF& temp_output_1_hot = *m_output_1_hot_slices[j];
	temp_output_1_hot(m_example_output(j), i) = 1;
	MatrixI& temp_output_class_index = *m_output_class_index_slices[j];
	temp_output_class_index(i) = m_example_output(j);
      }

      m_minibatch_indices(i) += m_num_slices;
    }
    //narrow(m_input_mini, m_input_full, m_example_index_dimension_input, m_minibatch_start_index, m_minibatch_size);
    //narrow_permuted(m_input_mini, m_input_full, m_example_index_dimension_input, m_minibatch_start_index,
    //              m_minibatch_size, m_example_indices);
    //narrow(m_output_mini, m_output_full, m_example_index_dimension_output, m_minibatch_start_index, m_minibatch_size);
    //narrow_permuted(m_output_mini, m_output_full, m_example_index_dimension_output, m_minibatch_start_index,
    //              m_minibatch_size, m_example_indices);
    /*
      bool end_epoch = false;
      int new_start_index = m_minibatch_start_index + m_minibatch_size;
      if (new_start_index >= m_char_count) {
      new_start_index = 0;
      }
      if ((m_minibatch_start_index != 0) && (new_start_index == 0)) {
      end_epoch = true;
      if (m_enable_shuffling) {
      shuffle_examples(); // Re-shuffle examples
      }
      } else {
      end_epoch = false;
      }
      m_minibatch_start_index = new_start_index;
      return end_epoch;
    */
    m_next_counter++;
    if ((m_next_counter % m_full_iteration_count) == 0) {
      for (int i=0; i < m_minibatch_size; ++i) {
	m_minibatch_indices(i) = m_minibatch_indices_init(i);
      }
      return true;
    } else {
      return false;
    }
  }


  void CharRNNMinibatchGetter::print_current_minibatch() {
    for (int i = 0; i < m_minibatch_size; ++i) {
      cout << "Example: " << i << " Network input:" << endl;
      for (int n = 0; n < m_num_slices; ++n) {
	const MatrixF& input_forward = get_input_forward_batch(n);
	// Find out which row has value 1 for the i'th example.
	int one_hot_row = 0;
	for (int r = 0; r < input_forward.extent(0); ++r) {
	  if (input_forward(r, i) > 0) {
	    one_hot_row = r;
	  }
	}
	char c = m_idx_to_char[one_hot_row];
	cout << c;
      }
      cout << endl;
      cout << "Example: " << i << " Network output:" << endl;
      for (int n = 0; n < m_num_slices; ++n) {
	const MatrixF& output_1_hot = get_output_1_hot_batch(n);
	// Find out which row has value 1 for the i'th example.
	int one_hot_row = 0;
	for (int r = 0; r < output_1_hot.extent(0); ++r) {
	  if (output_1_hot(r, i) > 0) {
	    one_hot_row = r;
	  }
	}
	char c = m_idx_to_char[one_hot_row];
	cout << c;
      }
      cout << endl;
      cout << "--------------------------------" << endl;
    }
  }

  std::map<char, int> create_char_idx_map(const std::string& text) {
    std::map<char, int> char_to_idx;
    int cur_index = 0;
    for (const char& c : text) {
      //cout << c << endl;
      if (char_to_idx.find(c) == char_to_idx.end()) {
        char_to_idx[c] = cur_index;
        ++cur_index;
      }
    }
    return char_to_idx;
  }


}
