#ifndef _CHAR_RNN_MINIBATCHTRAINER_H
#define _CHAR_RNN_MINIBATCHTRAINER_H
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
#include "Constants.h"
#include "Utilities.h"
#include "Utilities.h"
#include <random>
#include <algorithm>
#include <map>
#include <memory>
#include "Variable.h"

namespace kumozu {

/**
   * A Utility class to train a character-based recurrent neural network (RNN) in mini-batches.
   *
   * This class creates matrices for the input activations and target outputs for the RNN. For a character-based
   * RNN, the target output corresponds to the next character in the text sequence.
   *
   * When training a network using SGD or evaluating a network, it is common to supply small batches (called a mini-batch)
   * of examples at a time. This class provides functions that can be called to obtain these mini-batches of
   * examples.
   *
   * An instance of this class is supplied with a string of text which will be used to create the input activations and
   * target values for the RNN. This text string will typically either correspond to the training set or the test set.
   * Therefore, two instances of this class are typically used while training an RNN: one instance that is
   * supplied with a training string and another instance that is supplied with a test string.
   *
   * For an RNN with T time slices, the input activaitons correspond to a list of "input forward" matrices of length T as
   * well as the list of "input backward" matrices of length T. The "input foward" matrices contain the character values
   * that are read on the forward pass and the "input backwards" activations are written to by the network on the
   * backwards pass. The format for an input activations matrix at time slice t corresponds to a mini-batch of
   * activations of size (char_dim x minibatch_size). Thus, the matrix has minibatch_size columns, where the j'th
   * column corresponds to the j'th example in the mini-batch. Each column vector j contains all zeros except for
   * a 1 in row i where index i corresponds to on of the char_dim possible characters.
   *
   * The target outputs "target_outputs" for slice t may either correspond to a matrix of the same size for as the input activations or may
   * instead correspond to a 1-dim matrix of size minibatch_size. In the later case, target_outputs(j) is an integer
   * in the range [0, char_dim) that indicates which one of the char_dim possible characters is the desired character
   * in time slice j.
   *
   * An instance of this class will create the above-mentioned lists of matrices and funtions are provided to obtain a
   * reference to these matrices for each time slice of the RNN.
   *
   * Usage:
   *
   * First create an instance of this class, supplying the filename of the input text along with the
   * desired mini-batch size.
   *
   * A map from char to index value will be needed. To create the map from a text file, call create_char_idx_map().
   *
   * Then call set_char_to_idx_map() to supply the map from character to index value to an instance of this class.
   *
   * Then, optionally call various functions to configure aspects of how the instance will iterate through the examples.
   *
   * Then define or instantiate your desired RNN.
   *
   * Then, in a loop (for each time slice), call get_input_forward_batch(i), get_input_backward_batch(i), get_output_batch(i) to obtain a reference to the corresponding input and target output activations and connect them to the desired nodes in the network.
   *
   * Repeatedly call next() as many times as desired. Each call to next() will obtain a new mini-batch of input and output
   * examples which will be copied into the matrices that were obtained when get_input_forward_batch() and get_output_batch()
   * were called. Note that it is not necessary to call get_input_forward_batch() or get_output_batch() after each call
   * to next() since the same matrices are reused and new data is simply copied into them by calling next().
   *
   */
class CharRNNMinibatchGetter {

public:

    /**
     * Create a new instance using the supplied text.
     *
     * @parm text_data String containing the text data.
     *
     * @param minibatch_size The desired size of one mini-batch of examples that should be returned by the
     *                 get_input_batch()/get_output_batch() functions.
     *
     */
    CharRNNMinibatchGetter(const std::string& text_data, int minibatch_size, int num_slices):
        m_text_data {text_data},
        m_minibatch_size {minibatch_size},
        m_num_slices {num_slices},
        m_minibatch_start_index {0},
        m_is_initialized {false},
        m_is_new_epoch {false},
        m_next_counter {0}
    {
        std::cout << "CharRNNMinibatchGetter:" << std::endl;
        m_char_count = text_data.size();
        std::cout << "Number of characters in input text: " << m_char_count << std::endl;
        std::cout << "Using mini-batch size: " << m_minibatch_size << std::endl;
        std::cout << "Number of RNN slices: " << num_slices << std::endl;

        m_full_iteration_count = m_char_count/(num_slices*minibatch_size);
        std::cout << "Number of calls to next() for one full iteration through text: " << m_full_iteration_count << std::endl;

        int new_char_count = m_full_iteration_count*num_slices*minibatch_size;
        std::cout << "Only using the first: " << new_char_count << " characters of input text." << std::endl;

        //for (int i = 0; i < 10; ++i) {
        //  std::cout << random_offset() << std::endl;
        //}
        //exit(1);
        //

        //m_input_mini = narrow(m_input_full, m_example_index_dimension_input, 0, m_minibatch_size);
        //m_output_mini = narrow(m_output_full, m_example_index_dimension_output, 0, m_minibatch_size);
        // Optional: only allow a minibatch size that evenly divides into the number of examples.
        //if ((m_char_count % m_minibatch_size) != 0) {
        //std::cerr << "Number of examples divided by mini-batch size must have 0 remainder." << std::endl;
        //exit(1);
        //}

        //for (size_t n = 0; n != m_example_indices.size(); ++n) {
        //m_example_indices.at(n) = n;
        //}
        // m_example_indices[n] contains the index to the example to use at the n-th example in the epoch.
        // We will randomly permute these so that the index will change after iterating through an epoch of examples.
        //shuffle_examples();

        const int minibatch_step = new_char_count/minibatch_size;
        m_minibatch_indices.resize(m_minibatch_size);
        m_minibatch_indices_init.resize(m_minibatch_size);
        for (int i = 0; i != m_minibatch_size; ++i) {
            // The starting offset into the full text string for the i'th example in the mini-batch:
            //m_minibatch_indices[i] = i;  // stepped initial offset

            // This makes it so that the starting indices are separated by the total character count divded by mini-batch size.
            m_minibatch_indices[i] = i*minibatch_step;
            m_minibatch_indices_init[i] = i*minibatch_step;
            //m_minibatch_indices[i] = random_offset(); // randomized initial offset
        }

    }

    /**
           * Set the character-to-index map for this instance.
           *
           * It is recommended that all mini-batch getters used for for training/testing etc be supplied
           * with excatly the same char_to_idx_map. The recommended usage is for a single map to be created and
           * then supplied to all mini-batch getter instances by calling this member function.
           */
    void set_char_to_idx_map(std::map<char, int> char_to_idx_map);

    /**
           * Get a reference to the MatrixF for the specified time slice that will be populated with one
       * mini-batch of new input examples after each call to next().
           *
           * This function should only be called once, before the first call to next(). Otherwise, the
       * program will exit with an error.
       *
       * @param slice_index The slice index that specifies which matrix to return. The maximum allowed
       * value is one less than the total number of slices in the RNN.
           */
    const VariableF& get_input_forward_batch(int slice_index) const {
        return *m_input_var_slices.at(slice_index);
    }

    /**
           * Get a reference to the MatrixF for the specified time slice that will be populated with one
       * mini-batch of new input examples after each call to next().
           *
           * This function should only be called once, before the first call to next(). Otherwise, the
       * program will exit with an error.
       *
       * @param slice_index The slice index that specifies which matrix to return. The maximum allowed
       * value is one less than the total number of slices in the RNN.
           */
    VariableF& get_input_forward_batch(int slice_index) {
        return *m_input_var_slices.at(slice_index);
    }





    /**
           * Get a reference to the MatrixF for the specified time slice that will be populated with one
       * mini-batch of new output examples after each call to next().
           *
       * Note that there are two versions of this function. This version returns a float matrix in
       * which each column uses a one-hot encoding (the same encoding used for the input activations).
       *
           * This function should only be called once, before the first call to next(). Otherwise, the
       * program will exit with an error.
       *
       * @param slice_index The slice index that specifies which matrix to return. The maximum allowed
       * value is one less than the total number of slices in the RNN.
           */
    const MatrixF& get_output_1_hot_batch(int slice_index) const {
        return *m_output_1_hot_slices.at(slice_index);
    }

    /**
           * Get a reference to the MatrixF for the specified time slice that will be populated with one
       * mini-batch of new output examples after each call to next().
           *
       * Note that there are two versions of this function. This version returns a float matrix in
       * which each column uses a one-hot encoding (the same encoding used for the input activations).
       *
           * This function should only be called once, before the first call to next(). Otherwise, the
       * program will exit with an error.
       *
       * @param slice_index The slice index that specifies which matrix to return. The maximum allowed
       * value is one less than the total number of slices in the RNN.
           */
    MatrixF& get_output_1_hot_batch(int slice_index) {
        return *m_output_1_hot_slices.at(slice_index);
    }

    /**
           * Get a reference to the MatrixF for the specified time slice that will be populated with one
       * mini-batch of new output examples after each call to next().
           *
       * Note that there are two versions of this function. This version returns an int matrix in
       * which each column uses a class-index encoding. That is, the matrix corresponds to 1 x minibatch_size
       * matrix in which matrix(i) contains the int-valued class label for the i'tch batch where i
       * is in the range [0, class_label_count). Note that class_label_count is simply the number of
       * unique characters in the map that is supplied to set_char_to_idx_map().
       *
           * This function should only be called once, before the first call to next(). Otherwise, the
       * program will exit with an error.
       *
       * @param slice_index The slice index that specifies which matrix to return. The maximum allowed
       * value is one less than the total number of slices in the RNN.
           */
    const MatrixI& get_output_class_index_batch(int slice_index) const {
        return *m_output_class_index_slices.at(slice_index);
    }

    /**
           * Get a reference to the MatrixF for the specified time slice that will be populated with one
       * mini-batch of new output examples after each call to next().
           *
       * Note that there are two versions of this function. This version returns an int matrix in
       * which each column uses a class-index encoding. That is, the matrix corresponds to 1 x minibatch_size
       * matrix in which matrix(i) contains the int-valued class label for the i'tch batch where i
       * is in the range [0, class_label_count). Note that class_label_count is simply the number of
       * unique characters in the map that is supplied to set_char_to_idx_map().
       *
           * This function should only be called once, before the first call to next(). Otherwise, the
       * program will exit with an error.
       *
       * @param slice_index The slice index that specifies which matrix to return. The maximum allowed
       * value is one less than the total number of slices in the RNN.
           */
    MatrixI& get_output_class_index_batch(int slice_index) {
        return *m_output_class_index_slices.at(slice_index);
    }




    /**
           * Obtain a new mini-batch of input and output examples from the full data set and copy into
           * the matrices that were obtained from get_input_batch() and get_output_batch().
           *
           * @return true if calling this function returns the (approximate) last mini-batch in the examples set. Otherwise,
           * return false.
           */
    bool next();

    /*
           * Set to true to enable random shuffling of the examples set. If set to true,
           * the examples set will be randomly permutted after each epoch.
           */
    //void enable_shuffling(bool enable) {
    //m_enable_shuffling = enable;
    //}

    /*
           * Reset the the current mini-batch to the first batch.
           */
    void reset() {
        m_minibatch_start_index = 0;
    }

    /**
       * Print the current mini-batches for the RNN.
       *
       * For each of the minibatch_count examples, print both the network input and output as
       * a character sequence.
       *
       * This member function is intended for debugging purposes.
       */
    void print_current_minibatch();

    /**
       * Return the map from index to character.
       */
    const std::map<int, char>& get_idx_to_char_map() {
        return m_idx_to_char;
    }


private:

    /*
       * Return a random vallid starting offset.
       *
       * @return A random valid starting offset. The return index will be in the
       * range [0, m_char_count - m_num_slices).
       */
    int random_offset() {
        static std::random_device rand_dev;
        static std::mt19937 mersenne_twister_engine(rand_dev());
        std::uniform_int_distribution<int> uni(0, m_char_count - m_num_slices-1);
        return uni(mersenne_twister_engine);
    }

    /*
           * Compute a new random permutation of the examples.
           */
    //void shuffle_examples() {
    //std::cout << "Shuffling..." << std::endl;
    /*
            static std::random_device rand_dev;
            static std::mt19937 mersenne_twister_engine(rand_dev());
            std::shuffle(m_example_indices.begin(), m_example_indices.end(), mersenne_twister_engine);
        */
    //std::cout << "Number of examples = " << m_example_indices.size() << std::endl;
    //for (size_t n = 0; n < 40; ++n) {
    //  std::cout << m_example_indices.at(n) << ", ";
    //}
    //exit(1);
    //}

    const std::string& m_text_data;
    int m_minibatch_size;
    int m_num_slices;
    int m_minibatch_start_index;
    bool m_is_initialized;
    int m_char_count; // number of characters in the input text.
    bool m_is_new_epoch;

    std::vector<int> m_example_indices;
    //bool m_enable_shuffling;

    //
    // char_to_idx will map a character to a unique integer index value.
    std::map<char, int> m_char_to_idx;
    // idx_to_char will is the reverse-direction map that will map a unique integer index value to a character.
    std::map<int, char> m_idx_to_char;

    std::vector<std::unique_ptr<VariableF>> m_input_var_slices;
    //std::vector<std::unique_ptr<MatrixF>> m_input_backward_slices;
    std::vector<std::unique_ptr<MatrixF>> m_output_1_hot_slices;
    std::vector<std::unique_ptr<MatrixI>> m_output_class_index_slices;

    MatrixI m_text_mat; // Contains the text supplied to the constructor.

    MatrixI m_example_input; // Text of same length as RNN.
    MatrixI m_example_output; // Text of same length as RNN.

    // Size is same as minibatch_size.
    // m_minibatch_indices(i) contains the offset into the full text for the i'th example.
    MatrixI m_minibatch_indices;
    MatrixI m_minibatch_indices_init;

    // Approximate number of calls to next() that correspond to 1 full iteration through the input text.
    int m_full_iteration_count;
    int m_next_counter;
};

/**
   * Given a string of text, return the character map.
   *
   * @param text The input text.
   * @return The character to index map, which maps each unique character in the text
   * to an integer in the range [0, unique_char_count).
   */
std::map<char, int> create_char_idx_map(const std::string& text);

// todo: add a utility function to connect an instance of a mini-batch-getter to an RNN, assuming the
// RNN uses input/output ports named "0", "1", etc.

}


#endif /* _CHAR_RNN_MINIBATCHTRAINER_H */
