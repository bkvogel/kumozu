#ifndef _NETFLIXRECSYSTEM_H
#define _NETFLIXRECSYSTEM_H
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



#include <string>
#include <iostream>
#include "Constants.h"
#include "Utilities.h"
#include <vector>
#include <map>
#include "Matrix_list.h"

namespace kumozu {


  /*
   * Example of a matrix factorization-based recommendation system.
   *
   * The algorithm uses SGD to learn the W and H matrices in the low-rank
   * approximation:
   *
   * X approx= W * H
   *
   * where X is the sparsely-observed user ratings matrix of dimension
   * item_count x user_count. That is, each row in X corresponds to and
   * item id and each column corresponds to a user id. The value
   * of an element is the user rating: X(item_id, user_id) = rating value.
   *
   * Once the model is learned, W will contain the item parameter and H will
   * contain the user parameters.
   */
  class NetflixRecSystem {

  public:

    /*
     * Read in the user ratings and split into a training set and test
     * set. The test set consists of the ratings specified by the probe file.
     *
     * Good parameter settings:
     * lambda_W = 2e-2 (weight decay for W)
     * lambda_H = 2e-2 (weight decay for H)
     * alpha = 1e-3 (learning rate for both W and H)
     * feature_counter = 80 (40 works almost as well) (The rank of the approximation).
     *
     * Results in Probe set RMSE of 0.9115 using 80 features.
     * Results in Probe set RMSE of 0.910 using 160 features.
     *
     * For comparison, the original Cinimatch score was 0.9525 and
     * the winning team (BellKor's Pragmatic Chaos) scored 0.8567.
     *
     * See:
     * http://www.netflixprize.com/leaderboard
     */
  NetflixRecSystem(std::string training_set, std::string probe):
    m_user_id_count(0),
      m_item_id_count(0), m_train_rating_count(0), m_test_rating_count(0),
      m_ratings_mat_train(0,0), m_ratings_mat_test(0,0), m_lambda_W(2e-2f),
      m_lambda_H(2e-2f), m_alpha(0.001f)
      {
        // Load Netflix Prize data set:
        load_probe_set(probe);
        load_training_set(training_set);
        m_squared_errors_train.resize(m_train_rating_count);
        m_squared_errors_test.resize(m_test_rating_count);
        std::cout << "Number of training ratings = " << m_train_rating_count << std::endl;
        std::cout << "Number of test ratings = " << m_test_rating_count << std::endl;
        std::cout << "Number of distinct items = " << m_item_id_count << std::endl;
        std::cout << "Number of distinct users = " << m_user_id_count << std::endl;
      }

    /*
     * Learn the factor model for all users and items.
     *
     * The RMSE on the probe set (test set) is displayed as the
     * learning progresses.
     */
    void learn_model();

  private:

    /*
     * Load the probe set, which is the subset of (user_id, item_id) pairs
     * in the training set that Netflix suggests using as the validation set.
     */
    void load_probe_set(std::string probe);

    /*
     * Load the Netflix training set, which contains 100M+ user ratings.
     * The subset of these ratings specified in the probe set will be used for a validation set.
     * Since we don't have a separate test set, the training set will thus be partitioned into
     * a new smaller training set and a validation/test set.
     */
    void load_training_set(std::string training_set);

    std::map<std::string, int> m_probe_hashes;
    std::map<int, int> m_map_user_id_ext_to_int;
    std::map<int, int> m_map_item_id_ext_to_int;
    std::vector<float> m_squared_errors_train;
    std::vector<float> m_squared_errors_test;
    int m_user_id_count;
    int m_item_id_count;
    int m_train_rating_count;
    int m_test_rating_count;
    // Sparse matrix of user ratings.
    Matrix_list m_ratings_mat_train; // training set ratings.
    Matrix_list m_ratings_mat_test; // test set ratings (i.e., probe set).
    // L2 weight penalty for W:
    float m_lambda_W; // 0.01 weight decay parameter for W.
    // L2 weight penalty for H:
    float m_lambda_H; // 0.01 weight decay parameter for H.
    // The main learning rate.
    float m_alpha;
  };

}

#endif /* _NETFLIXRECSYSTEM_H */
