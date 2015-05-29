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
#include "NetflixRecSystem.h"
#include "Utilities.h"
#include "MatrixIO.h"

#include <fstream>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;
namespace kumozu {

	void NetflixRecSystem::learn_model() {
		// Model: X = W * H
		// where X is the sparse ratings matrix (list of ratings) of size
		// m_item_id_count x m_user_id_count.

		// Number of "basis" columns in W.
		const int feature_count = 160; // 40-80
		Matrix W(m_ratings_mat_train.rows, feature_count);
		const float mean = 0.0f;
		const float std_dev = 0.01f;
		randomize_normal(W, mean, std_dev);

		Matrix H(feature_count, m_ratings_mat_train.columns);
		randomize_normal(H, mean, std_dev);

		int reps = 200; // 100
		for (int i = 0; i < reps; i++) {
			cout << "-----iteration " << i << endl;

			// Loops through all ratings in training set:
			// This paralle for loop is not safe in general, but does not
			// seem to affect the RMSE scores and runs much faster. It is
			// probably reasonably safe to use for a large sparsely-observed ratings matrix.
			#pragma omp parallel for
			for (int i = 0; i < m_train_rating_count; ++i) {
				
				int row = m_ratings_mat_train.observed_list[i].row; // item id
				int col = m_ratings_mat_train.observed_list[i].col; // user id
				float val_X = m_ratings_mat_train.observed_list[i].val; // rating

				// Compute estimate for val_X and the approximation error.
				float approx_val_X = 0;
				for (int cur_feature = 0; cur_feature < feature_count; ++cur_feature) {
					approx_val_X += W(row, cur_feature) * H(cur_feature, col);
				}
				float cur_error = val_X - approx_val_X;
				m_squared_errors_train[i] = cur_error*cur_error;
				for (int cur_feature = 0; cur_feature < feature_count; ++cur_feature) {
					// Update W
					float w_i = W(row, cur_feature);
					// Same as for typical neural nets with backprop and weight decay.
					w_i = w_i - m_alpha*(H(cur_feature, col)*(-cur_error) + m_lambda_W*w_i);
					W(row, cur_feature) = w_i;
					// Update H
					float h_i = H(cur_feature, col);
					// Same as for typical neural nets with backprop and wieght decay.
					h_i = h_i - m_alpha*(W(row, cur_feature)*(-cur_error) + m_lambda_H*h_i);
					H(cur_feature, col) = h_i;
				}
			}
			// Loops through all ratings in training set:
			float sum_squared_error_train = 0.0f;
			for (int i = 0; i < m_train_rating_count; ++i) {
				sum_squared_error_train += m_squared_errors_train[i];
			}

			float train_rmse = sqrt(sum_squared_error_train/m_train_rating_count);
			cout << "Train RMSE = " << train_rmse << endl;
			// Loops through all ratings in test set to compute test RMSE:
			#pragma omp parallel for
			for (int i = 0; i < m_test_rating_count; ++i) {

				int row = m_ratings_mat_test.observed_list[i].row; // item id
				int col = m_ratings_mat_test.observed_list[i].col; // user id
				float val_X = m_ratings_mat_test.observed_list[i].val; // rating
				
				float approx_val_X = 0;
				for (int cur_feature = 0; cur_feature < feature_count; ++cur_feature) {
					approx_val_X += W(row, cur_feature) * H(cur_feature, col);
				}
				// Force the predicted value to be in [1, 5]:
				if (approx_val_X > 5.0f) {
					approx_val_X = 5.0f;
				} else if (approx_val_X < 1.0f) {
					approx_val_X = 1.0f;
				}
				float cur_error = val_X - approx_val_X;
				m_squared_errors_test[i] = cur_error*cur_error;
			}
			float sum_squared_error_test = 0.0f;
			for (int i = 0; i < m_test_rating_count; ++i) {
				sum_squared_error_test += m_squared_errors_test[i];
			}

			float test_rmse = sqrt(sum_squared_error_test/m_test_rating_count);
			cout << "Test RMSE = " << test_rmse << endl;
		}

		
	}

	void NetflixRecSystem::load_probe_set(std::string probe) {
		cout << "Reading probe file..." << endl;
		ifstream probe_file(probe);
		if (!probe_file) {
			cerr << "The probe file could not be opened for reading!" << endl;
			exit(1);
		}
		string line;
		string item_id_str;
		while (getline(probe_file, line)) {
			string::size_type item_id_pos = line.find_first_of(":");
			int item_id_int;
			if (item_id_pos != string::npos) {
				// This line contains an item id.
				item_id_str = line.substr(0, item_id_pos);
				stringstream item_id_stream(item_id_str);
				item_id_stream >> item_id_int;
			} else {
				// line contains the user id.
				stringstream user_id_stream(line);
				int user_id_int;
				user_id_stream >> user_id_int;
				if (m_map_user_id_ext_to_int.find(user_id_int) == m_map_user_id_ext_to_int.end()) {
					m_map_user_id_ext_to_int[user_id_int] = m_user_id_count++;
            }
				if (m_map_item_id_ext_to_int.find(item_id_int) == m_map_item_id_ext_to_int.end()) {
					m_map_item_id_ext_to_int[item_id_int] = m_item_id_count++;
				}
				m_probe_hashes[item_id_str + ":" + line] = 1;
			}
		}
		cout << "Ratings count in probe file = " << m_probe_hashes.size() << endl;
	}


	void NetflixRecSystem::load_training_set(std::string training_set) {
		cout << "Reading training set..." << endl;
		int counter = 0;
		boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
		for (boost::filesystem::directory_iterator itr(training_set);
			 itr != end_itr;
			 ++itr) {
			if (boost::filesystem::is_regular_file(itr->status())) {
				string cur_file = itr->path().string();
				counter++;
				if ((counter % 500) == 0) {
					cout << "Files read = " << counter << endl;
				}
				// debug:
				//if (counter > 2000) {
				//	cout << "Debug: Ignoring the rest of the ratings." << endl;
				//	break;
				//}
				ifstream infile(cur_file);
				string line;
				string item_id_str;
				//bool stop_early = true;
				while (getline(infile, line)) {
					
					string::size_type item_id_pos = line.find_first_of(":");
					int item_id_int;
					if (item_id_pos != string::npos) {
						item_id_str = line.substr(0, item_id_pos);
						stringstream item_id_stream(item_id_str);
						item_id_stream >> item_id_int;
					} else {
						istringstream line_stream(line);
						vector<string> line_items;
						string temp;
						while (std::getline(line_stream, temp, ',')) {
							line_items.push_back(temp);
						}
						if (line_items.size() != 3) {
							cerr << "Bad count." << endl;
							exit(1);
						}
						string user_id_str = line_items[0];
						stringstream user_id_stream(user_id_str);
						int user_id_int;
						user_id_stream >> user_id_int;
						string rating_str = line_items[1];
						float rating_float;
						stringstream rating_stream(rating_str);
						rating_stream >> rating_float;
						if (m_map_user_id_ext_to_int.find(user_id_int) == m_map_user_id_ext_to_int.end()) {
							m_map_user_id_ext_to_int[user_id_int] = m_user_id_count++;
						}
						if (m_map_item_id_ext_to_int.find(item_id_int) == m_map_item_id_ext_to_int.end()) {
							m_map_item_id_ext_to_int[item_id_int] = m_item_id_count++;
						}
						Matrix_element element;
						//element.row = item_id_int; // row index is internal item id
						element.row = m_map_item_id_ext_to_int[item_id_int]; // row index is internal item id
						//element.col = user_id_int; // column index is internal user id
						element.col = m_map_user_id_ext_to_int[user_id_int]; // column index is internal user id
						element.val = rating_float; // value is the rating.
						
						if (m_probe_hashes.find(item_id_str + ":" + user_id_str) == m_probe_hashes.end()) {
							m_ratings_mat_train.observed_list.push_back(element);
							m_train_rating_count++;
						} else {
							m_ratings_mat_test.observed_list.push_back(element);
							m_test_rating_count++;
						}
					}
			
				}
			
			}
			
		}
		m_ratings_mat_train.rows = m_item_id_count;
		m_ratings_mat_train.columns = m_user_id_count;
		m_ratings_mat_test.rows = m_item_id_count;
		m_ratings_mat_test.columns = m_user_id_count;
		
	}

	
}
