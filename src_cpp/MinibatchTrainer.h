#ifndef _MINIBATCHTRAINER_H
#define _MINIBATCHTRAINER_H
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


namespace kumozu {

  /*
   * A Utility class to train a network in mini-batches.
   *
   * An instance of this class is supplied with a full training or test set of examples network input and output activations.
   * Functions can then be called on this instance to obtain small chunks of examples (called a mini-batch) for the purpose
   * of training and/or testing a network. This class can be configured so that consecutive calles to obtain another
   * "mini-batch" of examples can either iterate sequentially through the full set or alternately iterate through a
   * shuffled (i.e., randomly permuted) set.
   *
   *
   * Usage:
   *
   * First create an instance of this class, supplying the full set of input and output examples along with the
   * desired mini-batch size. We allow the input and output types to differ since it is common for the output
   * type to be int when the output example represents a class label, or a float when it is used for a regression problem.
   *
   * Optionally call various methods to configure aspects of how the instance will iterate through the examples.
   *
   * Call get_input_batch() and get_output_batch() to obtain a reference to the corresponding input and output
   * mini-batch matrices. These functions only need to be called once because new data will be copied into the same
   * matrices as we iterate through the sample set.
   *
   * Repeatidly call next() as many times as desired. Each call to next() will obtain a new mini-batch of input and output
   * examples which will be copied into the matrices that were obtained when get_input_batch() and get_output_batch()
   * were called. Note that it is not necessary to call get_input_batch() or get_output_batch() after each call
   * to next() since the same matrices are reused and new data is simply copied into them by calling next().
   *
   */
  template <typename T1, typename T2>
    class MinibatchTrainer {

  public:

    /*
     * Create a new instance that will be associated with the supplied network.
     *
     * Parameters:
     *
     * network: an instance of a derived class of Network.
     *
     * input_full: A matrix that represents a full training or test set of examples where minibatch_index_input
     *        specifies the extent that is used to index a particular example.
     *
     * example_index_dimension_input: The dimension of input_full that is used to index the examples.
     *
     * output_full: A matrix that represents a full training or test set of examples where minibatch_index_output
     *        specifies the extent that is used to index a particular example.
     *
     * example_index_dimension_output: The dimension of output_full that is used to index the examples.
     *
     * minibatch_size: The desired size of one mini-batch of examples that should be returned by the
     *                 get_input_batch()/get_output_batch() functions.
     *
     */
  MinibatchTrainer(const Matrix<T1>& input_full, int example_index_dimension_input,
                   const Matrix<T2>& output_full, int example_index_dimension_output, int minibatch_size): // todo: make templated?
    m_minibatch_size {minibatch_size},
      m_input_full {input_full},
        m_output_full {output_full},
          m_example_index_dimension_input {example_index_dimension_input},
            m_example_index_dimension_output {example_index_dimension_output},
              m_minibatch_start_index {0},
                m_is_initialized {false},
                  m_example_count {input_full.extent(example_index_dimension_input)},
                    m_is_new_epoch {false},
                      m_example_indices(m_example_count),
                        m_enable_shuffling {false}
                      {
                        // Check that input and output matrices contain the same number of examples.
                        if (input_full.extent(example_index_dimension_input) != output_full.extent(example_index_dimension_output)) {
                          std::cerr << "MinibatchTrainer: input_full and output_full do not contain the same number of examples. Exiting." << std::endl;
                          exit(1);
                        }
                        if (m_minibatch_size >= m_example_count) {
                          std::cerr << "MinibatchTrainer: Mini-batch size should be smaller than the number of examples. Exiting." << std::endl;
                          exit(1);
                        }
                        std::cout << "MinibatchTrainer:" << std::endl;
                        std::cout << "Number of examples in set = " << m_example_count << std::endl;
                        std::cout << "Using mini-batch size = " << m_minibatch_size << std::endl;
                        /*
                          std::cout << "MinibatchTrainer:" << std::endl;
                          std::cout << "m_input_activations_mini.size() = " << m_input_activations_mini.size() << std::endl;
                          std::cout << "m_input_activations_mini.order() = " << m_input_activations_mini.order() << std::endl;
                          std::cout << "m_input_activations_mini.extent(0) = " << m_input_activations_mini.extent(0) << std::endl;
                          std::vector<int> x = network.get_input_extents();
                          std::cout << "x.size() = " << x.size() << std::endl;
                          std::cout << "x.at(0) = " << x.at(0) << std::endl;
                          std::cout << "x.at(1) = " << x.at(1) << std::endl;
                          std::cout << "x.at(2) = " << x.at(2) << std::endl;
                          exit(1);
                        */
                        m_input_mini = narrow(m_input_full, m_example_index_dimension_input, 0, m_minibatch_size);
                        m_output_mini = narrow(m_output_full, m_example_index_dimension_output, 0, m_minibatch_size);
                        // Optional: only allow a minibatch size that evenly divides into the number of examples.
                        if ((m_example_count % m_minibatch_size) != 0) {
                          std::cerr << "Number of examples divided by mini-batch size must have 0 remainder." << std::endl;
                          exit(1);
                        }

                        for (size_t n = 0; n != m_example_indices.size(); ++n) {
                          m_example_indices.at(n) = n;
                        }
                        // m_example_indices[n] contains the index to the example to use at the n-th example in the epoch.
                        // We will randomly permute these so that the index will change after iterating through an epoch of examples.
                        //shuffle_examples();


                      }

                      /*
                       * Get a reference to the MatrixF that will be populated with one mini-batch of new input examples
                       * after each call to next().
                       *
                       * This function should only be called once, before the first call to next().
                       */
                      const Matrix<T1>& get_input_batch() const {
                        return m_input_mini;
                      }

                      /*
                       * Get a reference to the MatrixF that will be populated with one mini-batch of new output examples
                       * after each call to next().
                       *
                       * This function should only be called once, before the first call to next().
                       */
                      const Matrix<T2>& get_output_batch() const {
                        return m_output_mini;
                      }


                      /*
                       * Obtain a new mini-batch of input and output examples from the full data set and copy into
                       * the matrices that were obtained from get_input_batch() and get_output_batch().
                       *
                       * Return true if calling this function returns the last mini-batch in the examples set. Otherwise,
                       * return false.
                       */
                      bool next() {
                        //narrow(m_input_mini, m_input_full, m_example_index_dimension_input, m_minibatch_start_index, m_minibatch_size);
                        narrow_permuted(m_input_mini, m_input_full, m_example_index_dimension_input, m_minibatch_start_index,
                                        m_minibatch_size, m_example_indices);
                        //narrow(m_output_mini, m_output_full, m_example_index_dimension_output, m_minibatch_start_index, m_minibatch_size);
                        narrow_permuted(m_output_mini, m_output_full, m_example_index_dimension_output, m_minibatch_start_index,
                                        m_minibatch_size, m_example_indices);
                        bool end_epoch = false;
                        int new_start_index = m_minibatch_start_index + m_minibatch_size;
                        if (new_start_index >= m_example_count) {
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
                      }

                      /*
                       * Set to true to enable random shuffling of the examples set. If set to true,
                       * the examples set will be randomly permutted after each epoch.
                       */
                      void enable_shuffling(bool enable) {
                        m_enable_shuffling = enable;
                      }

                      /*
                       * Reset the the current mini-batch to the first batch.
                       */
                      void reset() {
                        m_minibatch_start_index = 0;
                      }

  private:

                      /*
                       * Compute a new random permutation of the examples.
                       */
                      void shuffle_examples() {
                        //std::cout << "Shuffling..." << std::endl;
                        // m_example_indices
                        static std::random_device rand_dev;
                        static std::mt19937 mersenne_twister_engine(rand_dev());
                        std::shuffle(m_example_indices.begin(), m_example_indices.end(), mersenne_twister_engine);
                        //std::cout << "Number of examples = " << m_example_indices.size() << std::endl;
                        //for (size_t n = 0; n < 40; ++n) {
                        //  std::cout << m_example_indices.at(n) << ", ";
                        //}
                        //exit(1);
                      }

                      int m_minibatch_size;
                      const Matrix<T1>& m_input_full;
                      const Matrix<T2>& m_output_full;
                      int m_example_index_dimension_input;
                      int m_example_index_dimension_output;
                      Matrix<T1> m_input_mini;
                      Matrix<T2> m_output_mini;
                      int m_minibatch_start_index;
                      bool m_is_initialized;
                      int m_example_count;
                      bool m_is_new_epoch;

                      std::vector<int> m_example_indices;
                      bool m_enable_shuffling;

  };

}


#endif /* _MINIBATCHTRAINER_H */
